# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 14:12:02 2014

@author: hadu01
"""

import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
import os
import json
import numpy as np
import shutil
import sys
ROOT_DIR = os.sep.join([".."] * 6)
src_dir = os.sep.join([".."] * 5)
sys.path.insert(1, src_dir + r'/3 - Motion primitives/2 - motion_primitive/')
from motion_sample import MotionSample 
from lib.bvh import BVHReader, BVHWriter


def get_input_data_folder():
    """
    Return folder path of feature data without trailing os.sep
    """
    data_dir_name = "data"
    PCA_dir_name = "2 - PCA"
    type_parameter = "spatial"
    step = "1 - preprocessing"
    action = 'experiments'
    test_feature = "2 - FPCA with quaternion joint angles"

    input_dir = os.sep.join([ROOT_DIR,
                             data_dir_name,
                             PCA_dir_name,
                             type_parameter,
                             step,
                             action,
                             test_feature])
    return input_dir


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


def load_input_data(elementary_action, motion_primitive):
    """
    Load input feature data from json file

    Parameters
    ----------
     * elementary_action: String
    \tElementary action of the motion primitive
     * motion_primitive: String
    \tSpecified motion primitive

    Return
    ------
    data: 3d np.array
    \tThe first dimension is number of frames,
    second dimension is number of samples,
    third dimension is number of joints * length of position point
    """
    input_dir = get_input_data_folder()
    if len(input_dir) > 116 or True:
        input_dir = clean_path(input_dir)
    filename = input_dir + os.sep + '%s_%s_featureVector.json' % (elementary_action, motion_primitive)
    with open(filename, 'rb') as handle:
        dic_feature_data = json.load(handle)
    temp = []
    fileorder = []
#   extract data from dic_feature_data
    for key, value in dic_feature_data.iteritems():
        fileorder.append(key)
        temp.append(value)
        
    temp = np.asarray(temp)
    
    number_samples, number_frames, number_channels = temp.shape

    quaternions = number_channels - 1
    len_quat = 4
    len_root = 3
    number_dimensions = quaternions * len_quat + len_root

    data = np.zeros((number_frames, number_samples, number_dimensions))
    for i in xrange(number_frames):
        for j in xrange(number_samples):
            temp_j_i = []
            for k in xrange(len(temp[j,i])):
                for elem in temp[j,i,k]:
                    temp_j_i.append(elem)
            data[i, j, :] = np.array(temp_j_i)
            
    return data, fileorder


def function_generator(elementary_action,
                       motion_primitive,
                       save_path,
                       nharm=3,
                       numknots=7):
    """
    Generate functional data representation for motion data usnig fda library
    from R

    Parameters
    ----------
     * elementary_action: String
    \tElementary action of the motion primitive
     * motion_primitive: String
    \tSpecified motion primitive
    * save_path: String
    \tTarget folder to store functional data
    * nharm: Integer
    \tNumber of harmonic functions
    * numknots: Interger
    \tNumber of knots
    Return
    ------

    """
    data, fileorder = load_input_data(elementary_action, motion_primitive)
    data = np.array(data)
    print data.shape
#    print fileorder
#    print data[:,1,3]
#    print np.min(data)
#    print np.argmin(data)
#    #raw_input()
#    #print "continue"
##    print fileorder[1]
##    return
##    print 'the dimension of data: '
##    print data.shape
    robjects.conversion.py2ri = numpy2ri.numpy2ri
    r_data = robjects.Matrix(np.array(data))  # convert numpy to r data
    number_frames = data.shape[0]
    maxX = number_frames - 1
    nharm = min(nharm, numknots)
    filename = save_path + os.sep + '%s_%s_functionalData.RData' % \
        (elementary_action, motion_primitive)
#    print filename
    rcode = '''
        library(fda)
        # initialize parameters' value
        data = %s
        maxX = %d
        length = %d
        numknots = %d
        nharm = %d
        basisobj = create.bspline.basis(c(0,maxX),numknots)
        ys = smooth.basis(argvals=seq(0, maxX, len = length), y = data,
                          fdParobj = basisobj)
        fd = ys$fd
        coefs = fd$coefs
        saveRDS(fd, 'functionalData.RData')
        saveRDS(data, 'data.RData')
    ''' % (r_data.r_repr(), maxX, number_frames, numknots, nharm)
    robjects.r(rcode)
#    with open('temp.txt', 'w+') as f:
#        f.write(r_data.r_repr())
        
#    data2 = np.array(robjects.r['data'])
#    print np.min(data2)
#    print np.argmin(data2)
    coefs = robjects.globalenv['coefs']
#    coefs = robjects.r['coefs']
    coefs = np.array(coefs)
#    print np.max(coefs)
#    print np.argmax(coefs) 
    
    try:
        shutil.copyfile('functionalData.RData', filename)
        os.remove('functionalData.RData')
    except:
        raise IOError('no existing file or file path is wrong')
    return coefs


def get_output_folder():
    """
    Return folder path to store result without trailing os.sep
    """
    data_dir_name = "data"
    PCA_dir_name = "2 - PCA"
    type_parameter = "spatial"
    step = "2 - function_generation"
    action = 'experiments'
    test_feature = "2 - FPCA with quaternion joint angles"
    
    output_dir = os.sep.join([ROOT_DIR,
                              data_dir_name,
                              PCA_dir_name,
                              type_parameter,
                              step,
                              action,
                              test_feature])
    return output_dir


def from_fd_to_motion_data(coefs, n_frames):
    """Sample the motion data from functional representation
    
    Parameters
    ----------
    * coefs: 3d numpy array
    /tThe shape of coefs should be n_basis * n_samples * n_dims
    
    * n_frames: integer
    /tThe number of samples should be taken
    """
    n_basis, n_samples, n_dims = coefs.shape
    robjects.conversion.py2ri = numpy2ri.numpy2ri
    r_data = robjects.Matrix(np.asarray(coefs))
    rcode = '''
        library(fda)
        data = %s
        n_frames = %d
        n_basis = dim(data)[1]
        n_samples = dim(data)[2]
        n_dims = dim(data)[3]
        # create basis object
        basisobj = create.bspline.basis(c(0, n_frames-1), nbasis = n_basis)
        samples_mat = array(0, c(n_samples, n_frames, n_dim))
        for (i in 1:n_samples){
            for (j in 1:n_dim){
                fd = fd(data[,i,j], basisobj)
                samples = eval.fd(seq(0, n_frames - 1, len = n_frames), fd)
                samples_mat[i,,j] = samples
            }
        }
    ''' % (r_data.r_repr(), n_frames)
    robjects.r(rcode)
    reconstructed_data = np.asarray(robjects.globalenv['samples_mat'])
    return reconstructed_data
    

def gen_motion_from_feature_vector(data):
    """Reconstruct motion data from feature vector. to verify the extracted 
       data is correct or not
      
    Parameters
    ----------
    * data: 3d numpy array
    \tThe shape of data should be n_samples * n_frames * n_dims
    
    """
    n_samples, n_frames, n_dims = data.shape
    skeleton = os.sep.join(('lib', 'skeleton.bvh'))
    reader = BVHReader(skeleton)
    for i in xrange(n_samples):
        filename = 'generated motions' + os.sep + str(i) + '.bvh'
        frames = data[i,:,:]
        BVHWriter(filename, reader, frames, frame_time=0.013889,
                  is_quaternion=True)     
    


#def evaluate_functional_representation(coefs, canonical_frame_number):
#    """Evaluate functional representation of motion data by generating motion 
#       from coefficients
#    """
#    print "shape of coefs: "
#    print coefs.shape
#    n_basis, n_samples, n_dims = coefs.shape
#    for i in xrange(n_samples):
#        sample = MotionSample(coefs[:,i,:], 
#                              canonical_frame_number,
#                              np.arange(canonical_frame_number))
#        filename =  str(i) + '.bvh'
#        sample.save_motion_vector(filename)                            


if __name__ == '__main__':
    save_path = get_output_folder()
    if len(save_path) > 116 or True:
        save_path = clean_path(save_path)
    elementary_action = 'place'
    motion_primitive = 'second'
    coefs = function_generator(elementary_action, motion_primitive, save_path)
    print coefs.shape
#    evaluate_functional_representation(coefs, 132)
    reconstructed_data = from_fd_to_motion_data(coefs, 132)
#    print reconstructed_data.shape
    gen_motion_from_feature_vector(reconstructed_data)
    