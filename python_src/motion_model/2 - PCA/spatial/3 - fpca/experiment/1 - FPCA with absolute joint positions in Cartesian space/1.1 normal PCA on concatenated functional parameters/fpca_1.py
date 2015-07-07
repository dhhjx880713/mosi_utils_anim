# -*- coding: utf-8 -*-
"""
Created on Mon Dec 01 21:02:47 2014

@author: hadu01
"""
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
import numpy as np
import os
import shutil
import json
from lib.PCA import PCA, Center

ROOT_DIR = os.sep.join([".."] * 7)


def get_input_folder():
    """
    Return input folder path of functional data
    """
    data_dir_name = "data"
    PCA_dir_name = "2 - PCA"
    type_parameter = "spatial"
    step = "2 - function_generation"
    action = 'experiments'
    test_feature = "1 - FPCA with absolute joint positions in Cartesian space"
    input_dir = os.sep.join([ROOT_DIR,
                             data_dir_name,
                             PCA_dir_name,
                             type_parameter,
                             step,
                             action,
                             test_feature])
    return input_dir


def get_input_data_folder():
    """
    Return folder path of feature data without trailing os.sep
    """
    data_dir_name = "data"
    PCA_dir_name = "2 - PCA"
    type_parameter = "spatial"
    step = "1 - preprocessing"
    action = 'experiments'
    test_feature = "1 - FPCA with absolute joint positions in Cartesian space"
    input_dir = os.sep.join([ROOT_DIR,
                             data_dir_name,
                             PCA_dir_name,
                             type_parameter,
                             step,
                             action,
                             test_feature])
    return input_dir


def conventionalPCA():
    input_dir = get_input_data_folder()
    if len(input_dir) > 116:
        input_dir = clean_path(input_dir)
    elementary_action = 'walk'
    motion_primitive = 'leftStance'
    filename = input_dir + os.sep + '%s_%s_featureVector.json' % (elementary_action, motion_primitive)
    with open(filename, 'rb') as handle:
        dic_feature_data = json.load(handle)
    temp = []
#   extract data from dic_feature_data
    for key, value in dic_feature_data.iteritems():
        temp.append(value)
    temp = np.asarray(temp)
    number_samples, number_frames, number_joint, len_point = temp.shape
    print temp.shape
    return temp
#    data = np.zeros((number_frames, number_samples, number_joint * len_point))
#    for i in xrange(number_frames):
#        for j in xrange(number_samples):
#            data[i, j] = np.ravel(temp[j, i, :, :])
#    print data.shape
## reshape the data as samples * dimension
#    test_data = np.zeros((number_samples, number_frames*number_joint*len_point))
#    for i in xrange(number_samples):
#        test_data[i,:] = np.ravel(data[:, i, :])
#    c = Center(test_data, scale = False)
#    p = PCA(test_data, fraction = 0.99)
#    print "number of principal components for 99% information:" + str(p.npc)


def clean_path(path):
    """
    Generate absolute path starting with '\\\\?\\' to avoid failure of loading
    because of long path in windows

    Parameters
    ----------
    * path: string
    \tRelative path

    Return
    ------
    * path: string
    \tAbsolute path starting with '\\\\?\\'
    """
    path = path.replace('/', os.sep).replace('\\', os.sep)
    if os.sep == '\\' and '\\\\?\\' not in path:
        # fix for Windows 260 char limit
        relative_levels = len([directory for directory in path.split(os.sep)
                               if directory == '..'])
        cwd = [directory for directory in os.getcwd().split(os.sep)] if ':' not in path else []
        path = '\\\\?\\' + os.sep.join(cwd[:len(cwd)-relative_levels] + [directory for directory in path.split(os.sep) if directory != ''][relative_levels:])
    return path


def load_functional_data(RDataFile):
    """
    load functional data from pre-stored R data file

    Parameters
    ----------
    * RDataFile: string
    \tFile contains functional data

    Return
    ------
    * coefs: 3d numpy array
    \tCoefs contain the coefficients of functional data. The first dimension is
    the coefficients in time domain, the second dimension is the number of
    samples, the third dimension is the number of dimensions of each frame
    """
    rcode = """
        library(fda)
        fd = readRDS("%s")
        coefs = fd$coefs
    """ % (RDataFile)
    robjects.r(rcode)
    coefs = robjects.globalenv['coefs']
    coefs = np.asarray(coefs)
    print coefs.shape
    return coefs

def sampleFunctionalData(functionalData, n_frames):
    """
    get samples from functional data
    """
    robjects.conversion.py2ri = numpy2ri.numpy2ri
    r_data = robjects.Matrix(np.array(functionalData))
    rcode = '''
        library(fda)
        functionalData = %s
        n_frames = %d
        n_basis = dim(functionalData)[1]
        n_samples = dim(functionalData)[2]
        n_dim = dim(functionalData)[3]
        # define bspline basis functions
        basisobj = create.bspline.basis(c(0, n_frames - 1), nbasis = n_basis)
        samples_mat = array(0, c(n_samples, n_frames, n_dim))
        for (i in 1:n_samples){
            for (j in 1:n_dim){
                fd = fd(functionalData[,i,j], basisobj)
                samples = eval.fd(seq(0, n_frames - 1, len = n_frames), fd)
                samples_mat[i,,j] = samples
            }
        }
    ''' % (r_data.r_repr(), n_frames)   
    robjects.r(rcode)
    new_samples = np.asarray(robjects.globalenv['samples_mat'])
    return new_samples


class fpca(object):
    """
    Apply PCA on concatenated functional coefficients of each motion sample
    """
    def __init__(self, functionalData, fraction=0.9):
        self.functionalData = functionalData
        self.data = self.concatenate_functional_data(self.functionalData)
        print self.data.shape
#        self.test_data = self.data[:,:]
#        print self.test_data[0]
#        print 'dimension of original data: ' 
#        print self.test_data.shape
        c = Center(self.data, scale=False)
        self.mean = np.ravel(c.mean)
        p = PCA(self.data, fraction=fraction)
        self.npc = p.npc
        print "number of principal components: " + str(self.npc)
        self.eigenVecs = p.Vt[:self.npc]
        self.projectTrainingData()
        self.backprojectTrainingData()
        self.data = self.decentrailzed_data(self.data)
        self.backprojectedData = self.decentrailzed_data(self.backprojectedData)
        err = self.MSEbetweenCoefficients()
        self.backprojected_functional_data = self.decouple_functional_data()
        print 'MSE of coefficients is: ' + str(err)

    def concatenate_functional_data(self, data):
        """
        Reorder functional data, concatenaing coefficients for one sample as a
        long vector

        Parameters
        ----------
        * data: 3d numpy array
        \tCoefs contain the coefficients of functional data. The first
        dimension is the coefficients in time domain, the second dimension is
        the number of samples, the third dimension is the number of dimensions
        of each frame.

        Return
        ------
        * reordered_data: 2d numpy array
        \tEach row is corresponding to one motion sample, each column is
        corresponding to one dimension of motion data
        """
        if len(data.shape) != 3:
            raise ValueError('The input data should be a 3 dimension matrix')
        reordered_data = []
        self.n_coefs, self.n_samples, self.n_dim = data.shape
        for i in xrange(self.n_samples):
            tmp = []
            for j in xrange(self.n_dim):
                tmp.append(data[:, i, j])
            tmp = np.ravel(tmp)
            reordered_data.append(tmp)
        reordered_data = np.asarray(reordered_data)
        print reordered_data.shape
        return reordered_data
    
    def decouple_functional_data(self):
        '''
        decouple functional data for each dimension
        '''
        backProjected_functional_data = np.zeros(self.functionalData.shape)
        for i in xrange(self.n_samples):
            tmp = np.reshape(self.backprojectedData[i,:], (self.n_coefs, self.n_dim))
            for j in xrange(self.n_dim):
                backProjected_functional_data[:, i, j] = tmp[:,j]
        return backProjected_functional_data

    def inverse_PCA(self, lowV):
        """
        Project a low dimensional motion data back to functional data
        """
        lowV = np.asarray(lowV)
        motion_vec = np.dot(np.transpose(self.eigenVecs), lowV.T)
        motion_vec = np.ravel(motion_vec)
        motion_vec += self.mean
        # reorder the motion vector to a matrix
        coefs_frame = motion_vec.reshape(self.n_coefs, self.n_dim)
        return coefs_frame
    
    def projectTrainingData(self):
        """
        Project the raw data to low dimensional space
        """
        self.projectedData = np.dot(self.eigenVecs, np.transpose(self.data))
#        print self.projectedData.shape
    
    def backprojectTrainingData(self):
        """
        Back project low dimensional data to original space
        """
        self.backprojectedData = np.dot(np.transpose(self.eigenVecs), 
                                        self.projectedData)
        self.backprojectedData = np.transpose(self.backprojectedData)
        print self.backprojectedData.shape

    def MSEbetweenCoefficients(self):
        """
        Compute MSE between original coefficient and back projected
        coefficients
        """
        n_samples, n_coefficients = self.data.shape
        err = 0
        for i in xrange(n_samples):
            err += np.linalg.norm(self.data[i] - self.backprojectedData[i])
        err = err/(n_samples * n_coefficients )
        return err
    
    def decentrailzed_data(self, data):
        '''
        '''
        tmp = np.zeros(data.shape)
        for i in xrange(self.n_samples):
            tmp[i] = data[i] + self.mean
        return tmp

if __name__ == '__main__':
#    data = conventionalPCA()
    input_dir = get_input_folder()
    if input_dir > 116:  # avoid too long path in windows
        input_dir = clean_path(input_dir)
    elementary_action = 'walk'
    motion_primitive = 'leftStance'
    filename = input_dir + os.sep + '%s_%s_functionalData.RData' % \
        (elementary_action, motion_primitive)
    try:
        shutil.copyfile(filename, 'functionalData.RData')
    except:
        raise IOError('no existing file or file path is wrong')
    functionalData = load_functional_data('functionalData.RData')
    fp = fpca(functionalData, fraction=0.99)
#   get back projected functional training data, it should have the same size 
#   as functional data
    reordered_data = fp.data
    backProjectedFunctionalData = fp.backprojected_functional_data

#    assert functionalData.shape == backProjectedFunctionalData.shape
#   reconstruct functions from functional data, then sampling functions to get
#   discrete trajectory point positions
    new_samples = sampleFunctionalData(backProjectedFunctionalData, 47)
    print new_samples.shape
    os.remove('functionalData.RData')
