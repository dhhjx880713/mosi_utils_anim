# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 10:48:30 2015

@author: Markus, hadu01
"""
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
import os
import json
import numpy as np
import shutil
import matplotlib.pyplot as plt
import copy
from scipy.ndimage.filters import gaussian_filter1d
from lib.bvh import BVHReader, BVHWriter
#from lib.PCA import PCA, Center
from PCA_fd import standardPCA, PCA_fd
ROOT_DIR = os.sep.join([".."] * 7)

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


def get_output_folder():
    """Return folder path to store result without trailing os.sep
    """
    data_dir_name = "data"
    PCA_dir_name = "2 - PCA"
    type_parameter = "spatial"
    step = "3 - fpca"
    action = 'experiment'
    test_feature = "2 - FPCA with quaternion joint angles"
    experiment_type = "1.1 normal PCA on concatenated functional parameters"
    output_dir = os.sep.join([ROOT_DIR,
                              data_dir_name,
                              PCA_dir_name,
                              type_parameter,
                              step,
                              action,
                              test_feature,
                              experiment_type])
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
##   extract data from dic_feature_data
#    for key, value in dic_feature_data.iteritems():
#        fileorder.append(key)
##        value = np.asarray(value)
#        temp.append(value)
##        print value.shape
    fileorder = sorted(dic_feature_data.keys())
    for filename in fileorder:
        temp.append(dic_feature_data[filename])
        
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

def load_scale_vector(elementary_action, motion_primitive):
    """
    Load scale vector from json file

    Parameters
    ----------
     * elementary_action: String
    \tElementary action of the motion primitive
     * motion_primitive: String
    \tSpecified motion primitive

    Return
    ------

    """   
    input_dir = get_input_data_folder()
    if len(input_dir) > 116 or True:
        input_dir = clean_path(input_dir)
    filename = input_dir + os.sep + '%s_%s_maxVector.json' % (elementary_action, motion_primitive)   
    with open(filename, 'rb') as infile:
        scale_vector = json.load(infile)
        infile.close()
    return scale_vector

def reshape_data_for_PCA(data):
    '''
    Reshape data for standard PCA
    '''
    data = np.asarray(data)
    assert len(data.shape) == 3, ('Data matrix should be a 3d array')
    n_frames, n_samples, n_dims = data.shape
    reshaped_data = np.zeros((n_samples, n_frames * n_dims))
    for i in xrange(n_samples):
        tmp = np.reshape(data[:,i,:], (1, n_frames * n_dims))
        reshaped_data[i,:] = tmp
    return reshaped_data, (n_frames, n_samples, n_dims)

def reshape_data_for_fpca(data):
    '''
    Reshape data for functional PCA. The input data for functional PCA should 
    be a 3d array: n_samples * n_frames * n_dims 
    '''
    data = np.asarray(data)
    assert len(data.shape) == 3, ('Data matrix should be a 3d array')
    n_frames, n_samples, n_dims = data.shape
    reshaped_data = np.zeros((n_samples, n_frames, n_dims))
    for i in xrange(n_samples):
        for j in xrange(n_frames):
            reshaped_data[i, j, :] = data[j, i, :]
    return reshaped_data

def backprojection_to_motion_data(data, original_shape):
    '''
    Reshape backprojection data from PCA to motion data
    
    * reconstructed_data: 3d numpy array
    \tThe dimension of matrix should be n_samples * n_frames * n_dims
    '''
#    reconstructed_data = np.zeros(original_shape)
    n_frames, n_samples, n_dims = original_shape
    reconstructed_data = np.zeros((n_samples, n_frames, n_dims))
    for i in xrange(n_samples):
        reconstructed_data[i, :, :] = np.reshape(data[i, :],
                                                 (n_frames, n_dims))
    return reconstructed_data

def test_standardPCA(data):
    '''
    Experiments on standard PCA: compare the number of principal components
    regarding keep different ratio of variance
    '''
    reshaped_data, original_shape = reshape_data_for_PCA(data)
    # test setting
    variances = [0.80, 0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99]
    n_pcs = []
    MSEs = []
    for variance in variances:
        tmp = copy.deepcopy(reshaped_data)
        pcaobj, backprojection = standardPCA(tmp, 
                                             fraction = variance)
        reconstructed_data = backprojection_to_motion_data(backprojection,
                                                           original_shape)
        err = MSE(data, reconstructed_data)
        print err
        npc = pcaobj.npc
        n_pcs.append(npc)
        MSEs.append(err)
#    n_pcs = np.ravel(n_pcs)
#    MSEs = np.ravel(MSEs)
    fig1 = plt.figure(1)
    plt.plot(variances, n_pcs)
    plt.title('Number of principal components vs. variance ratio')
    plt.xlabel('percetage of variance to keep')
    plt.ylabel('number of principal components')
    plt.show()   
    fig2 = plt.figure(2)
    plt.plot(variances, MSEs)
    plt.title('Mean square error vs. variance ration')
    plt.xlabel('percetage of variance to keep') 
    plt.ylabel('mean square errors')
    plt.show() 

def MSE(raw_data, reconstructed_data):
    '''
    Compute the mean squared error bewteen original data and reconstructed
    data
    '''
    diff = raw_data - reconstructed_data
    n_frames, n_samples, n_dims = diff.shape
    err = 0
    # convert diff matrix to a long vector
    diff = np.ravel(diff)
#    for i in xrange(n_samples):
#        for j in xrange(n_frames):
#            for k in xrange(n_dims):
#                err += np.linalg.norm(diff[j, i, k])
    err = np.sum(diff * diff)
    err = err/(n_samples * n_frames * n_dims)
    return err


def smooth_data(data, sigma=2):
    """Smooth one-dimensional signal using Gaussian Kernel
    """
    n_frames, n_samples, n_dims = data.shape
    for i in xrange(n_samples):
        for j in xrange(n_dims):
            data[:,i,j] = gaussian_filter1d(data[:,i,j], sigma)
    return data
    
    
def test_functionalPCA(data):                                                     
    '''
    Experiments on functional PCA
    '''
    variances = [0.80, 0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99]
    reshaped_data = reshape_data_for_fpca(data)
    n_pcs = []
    MSEs = []
    for variance in variances:
        fpcaobj = PCA_fd(reshaped_data, n_basis = 7, fraction = variance)
        n_pcs.append(fpcaobj.pcaobj.npc)
        reconstructed_data = fpcaobj.reconstructed_data
        err = MSE(reshaped_data, reconstructed_data)
        print 'mean square error is:' + str(err)
        MSEs.append(err)
    fig1 = plt.figure(1)
    plt.plot(variances, n_pcs)
    plt.title('Number of principal components vs. variance ratio')
    plt.xlabel('percetage of variance to keep')
    plt.ylabel('number of principal components')
    plt.show()   
    fig2 = plt.figure(2)
    plt.plot(variances, MSEs)
    plt.title('Mean square error vs. variance ration')
    plt.xlabel('percetage of variance to keep') 
    plt.ylabel('mean square errors')
    plt.show() 


def gen_motion_from_feature_vector(data):
    """Reconstruct motion data from feature vector. to verify the extracted 
       data is correct or not
      
    Parameters
    ----------
    * data: 3d numpy array
    \tThe shape of data should be (n_samples, n_frames, n_dims)
    
    """
    n_samples, n_frames, n_dims = data.shape
    skeleton = os.sep.join(('lib', 'skeleton.bvh'))
    reader = BVHReader(skeleton)
    for i in xrange(n_samples):
        filename = 'generated_motions' + os.sep + str(i) + '.bvh'
        frames = data[i,:,:]
        BVHWriter(filename, reader, frames, frame_time=0.013889,
                  is_quaternion=True)   
                  

def plot_data(data, joint_order):
    """Plot data to check the smoothness of data
    plot the quaternion of left shoulder, the index of should is 6
    3 + 5*4 = 23, [24: 28]
    """
    print data.shape
    n_samples, n_frames, n_dims = data.shape
#    n_frames, n_samples, n_dims = data.shape
    fig, axarr = plt.subplots(2,2)
    for i in xrange(n_samples):
        axarr[0,0].plot(data[i, :, 3+ 4* (joint_order-1) + 0])
        axarr[0,1].plot(data[i, :, 3+ 4* (joint_order-1) + 1])
        axarr[1,0].plot(data[i, :, 3+ 4* (joint_order-1) + 2])
        axarr[1,1].plot(data[i, :, 3+ 4* (joint_order-1) + 3])
    plt.suptitle('LeftShoulder')
    plt.show()

def applyStandardPCAonFeatureData(elementary_action, motion_primitive):
    # extract feature data
    data, fileorder = load_input_data(elementary_action, motion_primitive)
    # data is 3d array: n_frames * n_samples * n_dims
    reshaped_data, original_shape = reshape_data_for_PCA(data)
    # reshaped_data: n_samples * n_frames * n-dims
    pcaobj, backprojection = standardPCA(reshaped_data, fraction = 0.95)

def fpca_spatial(spatial_data, root_scale, n_basis):
    reshaped_data = reshape_data_for_fpca(spatial_data)
    fpcaobj = PCA_fd(reshaped_data, n_basis, fraction=0.95, fpca=True)
    return fpcaobj
    

def main():
    elementary_action = 'walk'
    motion_primitive = 'leftStance'
    data, fileorder = load_input_data(elementary_action, motion_primitive)
    print data.shape

    n_frames, n_samples, n_dims = data.shape
#    smoothed_data = smooth_data(data, sigma=5)
    scale_vector = load_scale_vector(elementary_action, motion_primitive)
#    print scale_vector
    n_basis = 7
#    reshaped_data, original_shape = reshape_data_for_PCA(data)
#    print reshaped_data.shape
    
#    test_standardPCA(data)
#    test_functionalPCA(data)
    
     # standard PCA
#    pcaobj, backprojection = standardPCA(reshaped_data, fraction = 0.95)
#    reconstructed_data = backprojection_to_motion_data(backprojection, 
#                                                       original_shape)
    
#    err = MSE(data, reconstructed_data)
#    print err
  
    reshaped_data = reshape_data_for_fpca(data)                                                       
    fpcaobj = PCA_fd(reshaped_data, n_basis = n_basis, fraction = 0.95, fpca=True) 
    low_dimensional_functional_data = fpcaobj.lowVs

    spatial_eigenvectors = fpcaobj.eigenvectors    
    fdata = {}
    fdata['motion_type'] = elementary_action + '_' + motion_primitive
    fdata['spatial_parameters'] = low_dimensional_functional_data.tolist()
    fdata['file_order'] = fileorder
    fdata['spatial_eigenvectors'] = spatial_eigenvectors.tolist()
    fdata['n_frames'] = n_frames
    fdata['n_basis'] = n_basis
    fdata['scale_vector'] = scale_vector
    fdata['mean_motion'] = fpcaobj.centerobj.mean.tolist()
    fdata['n_dim_spatial'] = n_dims
    output_dir = get_output_folder()
    filename = output_dir + os.sep + elementary_action + '_'\
               + motion_primitive + '_low_dimensional_data.json'
    if len(filename) > 116:
        filename = clean_path(filename)
    with open(filename, 'wb') as outfile:
        json.dump(fdata, outfile)
        outfile.close()

                                          
    
    
if __name__ == '__main__':
#    test()
    main()
#    reconstructed_data = main()
#    with open('reconstructed_data.json', 'wb') as outfile:
#        json.dump(reconstructed_data.tolist(), outfile)
#    outfile.close()