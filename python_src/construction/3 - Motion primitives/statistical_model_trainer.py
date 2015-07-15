# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 09:34:02 2015

@author: hadu01
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import os
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
import shutil
from lib.bvh import BVHReader, BVHWriter
ROOT_DIR = os.sep.join([".."] * 3)

class StatisticalModelTrainer(object):

    def __init__(self, spatial_file=None, temporal_file=None, save_path=None):
        self._load_spatial_data(spatial_file)
        if temporal_file is None:
            self.use_temporal_parameter = False
        else:
            self.use_temporal_parameter = True
            self._load_temporal_data(temporal_file)
            self._combine_spatial_temporal_parameters()
        self._train_gmm()
        self._create_gmm()
        self._save_model(save_path=save_path)
    
#    def _check_corresponding_temporal_spatial(self):
#        '''Check the corresponding between temporal and spatial parameters are
#           correct or not
#        '''
        

    def _load_spatial_data(self, data):
        '''
        Load dimensional representation for motion segements from a json file

        Parameters
        ----------
        * data: json file
        \tThe data is stored in a dictionary
        '''
        with open(data, 'rb') as infile:
            tmp = json.load(infile)
        infile.close()
        self._motion_primitive_name = tmp['motion_type']
        self._spatial_parameters = np.array(tmp['spatial_parameters'])
#        self._time_parameters = np.array(tmp['time_parameters'])
        self._spatial_eigenvectors = tmp['spatial_eigenvectors']
        self._n_frames = int(tmp['n_frames'])
        self._scale_vec = tmp['scale_vector']
        self._n_basis = tmp['n_basis']
        self._mean_motion = tmp['mean_motion']
        self._n_dim_spatial = int(tmp['n_dim_spatial'])

    def _load_temporal_data(self, RDataFile):
        '''
        Load temporal parameters from R data
        '''
        rcode = '''
            library(fda)
            pcaobj = readRDS("%s")
        ''' % (RDataFile)
        robjects.r(rcode)
        self._temporal_pca = robjects.globalenv['pcaobj']
        self._temporal_parameters = np.asarray(self._temporal_pca[self._temporal_pca.names.index('scores')])

    def _weight_temporal_parameters(self):
        '''
        Weight low dimensional temporal parameters
        '''
        weight_matrix = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        self._temporal_parameters = np.dot(self._temporal_parameters,
                                           weight_matrix)

    def _combine_spatial_temporal_parameters(self):
        '''
        Concatenate temporal and spatial paramters of same motion sample as a 
        long vector
        '''
        print self._spatial_parameters.shape[0]
        print self._temporal_parameters.shape[0]
        assert self._spatial_parameters.shape[0] == \
               self._temporal_parameters.shape[0], ('Number of samples are not the same for spatial parameters and temporal parameters')
        self._weight_temporal_parameters()
        self._motion_parameters = np.concatenate((self._spatial_parameters,
                                                  self._temporal_parameters,),
                                                 axis=1)

    def _train_gmm(self, n_K=10, DEBUG=0):
        '''
        Find the best number of Gaussian using BIC score
        '''
        if self.use_temporal_parameter:
            print 'dimension of training data: '
            print self._motion_parameters.shape
            obs = np.random.permutation(self._motion_parameters)
        else:
            print 'dimension of training data: '
            print self._spatial_parameters.shape
            obs = np.random.permutation(self._spatial_parameters)
        lowestBIC = np.infty
        BIC = []
        K = range(1, n_K)
        BICscores = []
        for i in K:
            gmm = mixture.GMM(n_components=i, covariance_type='full')
            gmm.fit(obs)
            BIC.append(gmm.bic(obs))
            BICscores.append(BIC[-1])
            if BIC[-1] < lowestBIC:
                lowestBIC = BIC[-1]
        index = min(xrange(n_K - 1), key=BIC.__getitem__)
        print 'number of Gaussian: ' + str(index + 1)
        self.numberOfGaussian = index + 1
        if DEBUG:
            fig = plt.figure()
            plt.plot(BICscores)
            plt.show()

    def _create_gmm(self):
        '''
        Using GMM to model the data with optimized number of Gaussian
        '''
        self.gmm = mixture.GMM(n_components=self.numberOfGaussian,
                               covariance_type='full')
#        self.gmm.fit(self._spatial_parameters)
#        scores = self.gmm.score(self._spatial_parameters)
        if self.use_temporal_parameter:
            self.gmm.fit(self._motion_parameters)
            scores = self.gmm.score(self._motion_parameters)
        else:
            self.gmm.fit(self._spatial_parameters)
            scores = self.gmm.score(self._spatial_parameters)
#        print scores
        averageScore = np.mean(scores)
        print 'average score is:' + str(averageScore)
    
    def _sample_spatial_parameters(self, n, save_path=None):
        '''Generate ranmdon sample from mrophable model based on spatial p
           parameters
        '''
        self.new_ld_vectors = self.gmm.sample(n)
        for i in xrange(n):
            filename = 'generated_motions' + os.sep + str(i) + '.bvh'
            self._backprojection(self.new_ld_vectors[i], filename = filename)
    
    def _sample_fd_spatial_parameters(self, n, save_path=None):
        self.new_fd_ld_vectors = self.gmm.sample(n)
        
    
    def _backprojection(self, ld_vec, filename=None):
        """Back project a low dimensional spatial parameter to motion
        """
        eigenVectors = np.array(self._spatial_eigenvectors)
        backprojected_vector = np.dot(np.transpose(eigenVectors), ld_vec.T)
        backprojected_vector = np.ravel(backprojected_vector)
        backprojected_vector += self._mean_motion
        # reshape motion vector as a 2d array n_frames * n_dim
        assert len(backprojected_vector) == self._n_dim_spatial * self._n_frames, ('the length of back projected motion vector is not correct!')
        if filename is None:
            filename = 'sample.bvh'
        else:
            filename = filename
        frames = np.reshape(backprojected_vector, (self._n_frames, 
                                                   self._n_dim_spatial))
        # rescale root position for each frame
        for i in xrange(self._n_frames):
            frames[i, 0] = frames[i, 0] * self._scale_vec[0]
            frames[i, 1] = frames[i, 1] * self._scale_vec[1]
            frames[i, 2] = frames[i, 2] * self._scale_vec[2]
        skeleton = os.sep.join(('lib', 'skeleton.bvh'))
        reader = BVHReader(skeleton)                                                   
        BVHWriter(filename, reader, frames, frame_time=0.013889,
                  is_quaternion=True)                                                      
                                                           
                                                   
    def _save_model(self, save_path=None):
        '''
        Save model as a json file

        Parameters
        ----------
        *filename: string
        \tGive the file name to json file
        '''
        if save_path is None:
            filename = self._motion_primitive_name + '_quaternion_mm.json'
        else:
            filename = save_path + os.sep + self._motion_primitive_name + '_quaternion_mm.json'
            if len(filename) > 116:
                filename = clean_path(filename)
        weights = self.gmm.weights_.tolist()
        means = self.gmm.means_.tolist()
        covars = self.gmm.covars_.tolist()
        mean_fd = self._temporal_pca[self._temporal_pca.names.index('meanfd')]
        self._mean_time_vector = np.array(mean_fd[mean_fd.names.index('coefs')])
        self._mean_time_vector = np.ravel(self._mean_time_vector)
        n_basis_time = len(self._mean_time_vector)
        harms = self._temporal_pca[self._temporal_pca.names.index('harmonics')]
        self._eigen_vectors_time = np.array(harms[harms.names.index('coefs')])
        data = {'name': self._motion_primitive_name,
                'gmm_weights': weights,
                'gmm_means': means,
                'gmm_covars': covars,
                'eigen_vectors_spatial': self._spatial_eigenvectors,
                'mean_spatial_vector': self._mean_motion,
                'n_canonical_frames': self._n_frames,
                'translation_maxima': self._scale_vec,
                'n_basis_spatial': self._n_basis,
                'eigen_vectors_time': self._eigen_vectors_time.tolist(),
                'mean_time_vector': self._mean_time_vector.tolist(),
                'n_dim_spatial': self._n_dim_spatial,
                'n_basis_time': n_basis_time}
        with open(filename, 'wb') as outfile:
            json.dump(data, outfile)
        outfile.close()

def get_input_folder_spatial():
    """
    Return input folder path of functional data
    """
    data_dir_name = "data"
    PCA_dir_name = "2 - PCA"
    type_parameter = "spatial"
    step = "3 - fpca"
    action = 'experiment'
    feature = '2 - FPCA with quaternion joint angles'
    test_type = '1.1 normal PCA on concatenated functional parameters'
    input_dir = os.sep.join([ROOT_DIR,
                             data_dir_name,
                             PCA_dir_name,
                             type_parameter,
                             step,
                             action,
                             feature,
                             test_type
                             ])
    return input_dir


def get_input_folder_temporal():
    """
    Return input folder path of functional data
    """
    data_dir_name = "data"
    PCA_dir_name = "2 - PCA"
    type_parameter = "temporal"
    step = "3 - fpca__result"
    action = 'experiments'
    input_dir = os.sep.join([ROOT_DIR,
                             data_dir_name,
                             PCA_dir_name,
                             type_parameter,
                             step,
                             action])
    return input_dir


def get_output_folder(elementary_action):
    """Return the save path without trailing os.sep
    """
    data_dir_name = "data"
    motion_primitive_dir = "3 - Motion primitives"
    model_type = "motion_primitives_quaternion_PCA95"
    elementary_action = "elementary_action_" + elementary_action
    output_dir = os.sep.join([ROOT_DIR,
                              data_dir_name,
                              motion_primitive_dir,
                              model_type,
                              elementary_action])
    return output_dir


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

def get_input_dir_standard_PCA():
    """
    Return input folder path of spatial parameters from standard PCA
    """
    data_dir_name = "data"
    PCA_dir_name = "2 - PCA"
    type_parameter = "spatial"
    step = "3 - fpca"
    action = 'experiment'
    feature = '2 - FPCA with quaternion joint angles'
    test_type = '1.2 normal PCA on motion data'
    input_dir = os.sep.join([ROOT_DIR,
                             data_dir_name,
                             PCA_dir_name,
                             type_parameter,
                             step,
                             action,
                             feature,
                             test_type
                             ])
    return input_dir   

def main():
    input_dir_spatial = get_input_folder_spatial()
    if len(input_dir_spatial) > 116:  # avoid too long path in windows
        input_dir_spatial = clean_path(input_dir_spatial)
    elementary_action = 'carryBoth'
    motion_primitive = 'turningRightStance'
    filename_spatial = input_dir_spatial + os.sep + '%s_%s_low_dimensional_data.json' % \
        (elementary_action, motion_primitive)
    input_dir_temporal = get_input_folder_temporal()
    if len(input_dir_temporal) > 116:  # avoid too long path in windows
        input_dir_temporal = clean_path(input_dir_temporal)
    filename_temporal = input_dir_temporal + os.sep + 'b_splines_%s_%s.RData' % \
        (elementary_action, motion_primitive) 
    try:
        shutil.copyfile(filename_temporal, 'functionalData.RData')
    except:
        raise IOError('no existing file or file path is wrong') 
        
#    testFile = 'walk_leftStance_low_dimensional_data.json'
#    rDataFile = 'b_splines_walk_leftStance.RData'
#    temporalDataFile = 'scores_walk_leftStance.npy'
    output_dir = get_output_folder(elementary_action)
    model =  StatisticalModelTrainer(spatial_file=filename_spatial, 
                                     temporal_file='functionalData.RData', 
                                     save_path=output_dir)
    os.remove('functionalData.RData')

def test_standPCA_on_quaternion_spatial():
    input_dir_spatial_standard_PCA = get_input_dir_standard_PCA()
    elementary_action = "pick"
    motion_primitive = 'first'
    filename_spatial = input_dir_spatial_standard_PCA + os.sep + '%s_%s_low_dimensional_data.json' % \
        (elementary_action, motion_primitive)
    if len(filename_spatial) > 116:  # avoid too long path in windows
        filename_spatial = clean_path(filename_spatial)    
    model = StatisticalModelTrainer(spatial_file=filename_spatial)
    model._sample_spatial_parameters(30)

def test_fpca_on_quaternion_spatial():
    input_dir_spatial = get_input_folder_spatial()
    elementary_action = 'carry'
    motion_primitive = 'leftStance'  
    filename_spatial = input_dir_spatial + os.sep + '%s_%s_low_dimensional_data.json' % \
        (elementary_action, motion_primitive)
    if len(filename_spatial) > 116:
        filename_spatial = clean_path(filename_spatial)
    model = StatisticalModelTrainer(spatial_file=filename_spatial)
    
    
if __name__ == '__main__':
#    test_standPCA_on_quaternion_spatial()
    main()
