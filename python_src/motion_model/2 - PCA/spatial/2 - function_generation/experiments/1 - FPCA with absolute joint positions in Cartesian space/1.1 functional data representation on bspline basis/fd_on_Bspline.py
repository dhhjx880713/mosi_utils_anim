# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 11:00:42 2015

@author: hadu01
"""

import numpy as np
import json
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import os
from lib.bvh import BVHReader

ROOT_DIR = os.sep.join([".."] * 7)


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

def get_joint_sequence():
    """
    Return the list of joints in skeleton
    Joint sequence:
    'Hips', 'Spine', 'Spine_1', 'Neck', 'Head', 'LeftShoulder',
    'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 
    'RightForeArm', 'RightHand', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 
    'RightUpLeg', 'RightLeg', 'RightFoot'
    """
    skeletonFile = r'data\skeleton.bvh'
    bvhFile = BVHReader(skeletonFile)
    joints = []
    for joint_name in bvhFile.node_names.keys():
        if not joint_name.startswith("Bip"):
            joints.append(joint_name)
    return joints

def plot_joint_trajectory(data, joint_name):
    """
    Plot 3d root trajectories in data

    Parameters:
    -----------
    *data: 3d array: n_sample * n_frames * dim_point

    """
    if len(data.shape) == 3:
        # data contains multiple trajectories
        n_samples, n_frames, dim = data.shape
        # change coordinate
        temp = np.zeros((n_samples, n_frames, dim))
        temp[:, :, 0] = data[:, :, 0]
        temp[:, :, 2] = data[:, :, 1]
        temp[:, :, 1] = data[:, :, 2]
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x_max = np.max(temp[:, :, 0])
        x_min = np.min(temp[:, :, 0])
        y_max = np.max(temp[:, :, 1])
        y_min = np.min(temp[:, :, 1])
        z_max = np.max(temp[:, :, 2])
        z_min = np.min(temp[:, :, 2])
        x_mean = np.mean(temp[:, :, 0])
        y_mean = np.mean(temp[:, :, 1])
        z_mean = np.mean(temp[:, :, 2])
#        print x_max
#        print x_min
#        print y_max
#        print y_min
#        print z_max
#        print z_min
#        print x_mean
#        print y_mean
#        print z_mean
        for i in xrange(n_samples):
            tmp = temp[i, :, :]
            tmp = np.transpose(tmp)
            ax.plot(*tmp)
    elif len(data.shape) == 2:
        # data constains one trajectory
        n_frames, dim = data.shape
        temp = np.zeros((n_frames, dim))
        temp[:,0] = data[:,0]
        temp[:,2] = data[:,1]
        temp[:,1] = data[:,2]
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x_max = np.max(temp[:, 0])
        x_min = np.min(temp[:, 0])
        y_max = np.max(temp[:, 1])
        y_min = np.min(temp[:, 1])
        z_max = np.max(temp[:, 2])
        z_min = np.min(temp[:, 2])
        x_mean = np.mean(temp[:, 0])
        y_mean = np.mean(temp[:, 1])
        z_mean = np.mean(temp[:, 2])
        tmp = np.transpose(temp)
        ax.plot(*tmp)
    else:
        raise ValueError('The shape of data is not correct!')
    max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0
#    print max_range
    ax.set_xlim(x_mean - max_range, x_mean + max_range)
    ax.set_ylim(y_mean - max_range, y_mean + max_range)
    ax.set_zlim(z_mean - max_range, z_mean + max_range)
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Z [cm]')
    ax.set_zlabel('Y [cm]')
    plt.title(joint_name)
    plt.show()

def get_joint_data(joint_index):
    """
    Return 3d root joint position
    
    Return:
    -------
    * data: 3d array
    \tData contains 3d position for joints of samples
    """
    input_dir = get_input_data_folder()
    if len(input_dir) > 116:
        input_dir = clean_path(input_dir)
    elementary_action = 'walk'
    motion_primitive = 'leftStance'
    filename = input_dir + os.sep + '%s_%s_featureVector.json' % \
                                    (elementary_action, motion_primitive)

    with open(filename, 'rb') as handle:
        dic_feature_data = json.load(handle)
    temp = []
#   extract data from dic_feature_data
    for key, value in dic_feature_data.iteritems():
        temp.append(value)
    temp = np.asarray(temp)
    number_samples, number_frames, number_joint, len_point = temp.shape
#    print temp.shape
#    return temp    
    data = temp[:, :, joint_index, :]
    print data.shape
    return data

def convert_to_fd(input_data, n_basis):
    '''
    represent data as a linear combination of basis function, and return 
    weights of functions
    
    Parameters
    ----------
    * input_data: 3d array (n_samples * n_frames *n_dim)
    * n_basis: integer
    \tThe number of basis functions to be used
    
    Return
    ------
    * coefs: 3d array (n_coefs * n_samples * n_dim)
    '''
    input_data = np.asarray(input_data)
    assert len(input_data.shape) == 3, ('input data should be a 3d array')
    n_samples, n_frames, n_dim = input_data.shape
    # reshape the data matrix for R library fda
    tmp = np.zeros((n_frames, n_samples, n_dim))
    for i in xrange(n_frames):
        for j in xrange(n_samples):
            tmp[i, j] = input_data[j, i, :]
    robjects.conversion.py2ri = numpy2ri.numpy2ri
    r_data = robjects.Matrix(np.asarray(tmp))
    rcode = '''
        library(fda)
        data = %s
        n_basis = %d
        n_samples = dim(data)[2]
        n_frames = dim(data)[1]
        n_dim = dim(data)[3]
        basisobj = create.bspline.basis(c(0, n_frames - 1),
                                        nbasis = n_basis)
        smoothed_tmp = smooth.basis(argvals=seq(0, {n_frames-1},
                        len = {n_frames}),y = {data}, fdParobj = basisobj)
        fd = smoothed_tmp$fd                                                  
    ''' % (r_data.r_repr(), n_basis)
    robjects.r(rcode)
    fd = robjects.globalenv['fd']
    coefs = fd[fd.names.index('coefs')]
    coefs = np.asarray(coefs)   
    print coefs.shape
    return coefs

def from_fd_to_data(fd, n_frames):
    '''
    generate data from weights of basis functions
    
    Parameter
    ---------
    * fd: 3d array (n_weights * n_samples * n_dim)
    \tThe weights of basis functions
    
    * n_frames: integer
    \tDefine the number of samples to be sample from function
    '''
    assert len(fd.shape) == 3, ('weights matrix should be a 3d array')
#        n_basis, n_samples, n_dim = coefs.shape
    robjects.conversion.py2ri = numpy2ri.numpy2ri
    r_data = robjects.Matrix(np.asarray(fd))
    rcode = '''
        library(fda)
        data = %s
        n_frames = %d
        n_basis = dim(data)[1]
        n_samples = dim(data)[2]
        n_dim = dim(data)[3]
        basisobj = create.bspline.basis(c(0, n_frames - 1), nbasis = n_basis)
        samples_mat = array(0, c(n_samples, n_frames, n_dim))
        for (i in 1:n_samples){
            for (j in 1:n_dim){
                fd = fd(data[,i,j], basisobj)
                samples = eval.fd(seq(0, n_frames -1, len = n_frames), fd)
                samples_mat[i,,j] = samples
            }
        }
    ''' % (r_data.r_repr(), n_frames)
    robjects.r(rcode)
    reconstructed_data = np.asarray(robjects.globalenv['samples_mat'])
    return reconstructed_data

def MSE(raw_data, reconstructed_data):
    '''
    Compute the mean squared error bewteen original data and reconstructed 
    data
    '''
    diff = raw_data - reconstructed_data
    n_samples, n_frames, n_dim = diff.shape
    err = 0
    for i in xrange(n_samples):
        for j in xrange(n_frames):
            err += np.linalg.norm(diff[i, j])
    err = err/(n_samples * n_frames)
    return err

def main():
    test_n_basis = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    test_joints = ['Hips', 'Head', 'LeftHand','RightHand']
    joints = get_joint_sequence()
    errs = {}
    for joint in test_joints:
        joint_index = joints.index(joint)
        trajectory_data = get_joint_data(joint_index)
#        plot_joint_trajectory(trajectory_data, joint)
        n_samples, n_frames, n_dims = trajectory_data.shape
        temp = []
        for n_basis in test_n_basis:
            coefs = convert_to_fd(trajectory_data, n_basis)
            reconstructed_data = from_fd_to_data(coefs, n_frames)
#            plot_joint_trajectory(reconstructed_data, joint)
            err = MSE(trajectory_data, reconstructed_data)
            temp.append(err)
        errs[joint] = temp
    fig = plt.figure()
    for joint, err in errs.iteritems():
        plt.plot(test_n_basis, err, label = joint)
    plt.legend()
    plt.title('reconstruction errors of 4 joints based on bspline basis')
    plt.xlabel('number of basis functions')
    plt.ylabel('mean squared error: ')
    plt.show()
            
def test():
#    data = get_root_joint_data()
#    get_root_joint_data()
    joints = get_joint_sequence()
    print joints
    print 'number of joints: ' + str(len(joints))
#    testjoint = 'Hips'
#    joint_index = joints.index(testjoint)
#    data = get_joint_data(joint_index)
#    n_samples, n_frames, n_dims = data.shape
#    plot_joint_trajectory(data)


if __name__ == '__main__':
    main()
#    test()
