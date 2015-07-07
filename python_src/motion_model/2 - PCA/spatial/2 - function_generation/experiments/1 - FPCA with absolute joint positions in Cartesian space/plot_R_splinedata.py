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
import matplotlib.pylab as plt
ROOT_DIR = os.sep.join([".."] * 6)


def get_input_data_folder():
    """
    Return folder path of feature data without trailing os.sep
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
    if len(input_dir) > 116:
        input_dir = clean_path(input_dir)
    filename = input_dir + os.sep + '%s_%s_featureVector.json' % (elementary_action, motion_primitive)
    with open(filename, 'rb') as handle:
        dic_feature_data = json.load(handle)
    temp = []
#   extract data from dic_feature_data
    for key, value in dic_feature_data.iteritems():
        temp.append(value)
    temp = np.asarray(temp)
    number_samples, number_frames, number_joint, len_point = temp.shape
    data = np.zeros((number_frames, number_samples, number_joint * len_point))
    for i in xrange(number_frames):
        for j in xrange(number_samples):
            data[i, j] = np.ravel(temp[j, i, :, :])
#     print data.shape
    return data

    
    

def function_generator(elementary_action,
                       motion_primitive,
                       save_path,
                       nharm=3,
                       numknots=8):
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
    data = load_input_data(elementary_action, motion_primitive)
    nsamples = data.shape[1]
    
    # Align Root joint
    for i in xrange(nsamples):
        data[:, i, 0] -= data[0, i, 0]
        data[:, i, 1] -= data[0, i, 1]
        data[:, i, 2] -= data[0, i, 2]
    data = data[:, :, :3]
    
#    print 'the dimension of data: '
#    print data.shape
    robjects.conversion.py2ri = numpy2ri.numpy2ri
    r_data = robjects.Matrix(np.array(data))  # convert numpy to r data
    number_frames = data.shape[0]
    maxX = number_frames - 1
    nharm = min(nharm, numknots)
    filename = save_path + os.sep + '%s_%s_fd_root.RData' % \
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
        basisobj = create.bspline.basis(c(0,{maxX}),{numknots})
        ys = smooth.basis(argvals=seq(0, {maxX}, len = {length}), y = {data},
                          fdParobj = basisobj)
        rootdata = ys$fd
        
        pcobj <- pca.fd(rootdata) # get meanfd
        mean <- pcobj$mean
        new_coefs = rootdata$coefs
        nsamples = 620
        for (i in i:nsamples){
        new_coefs[,i,] = new_coefs[,i,] - mean$coefs[,1,]
        }
        centered_root = fd(new_coefs, rootdata$basis)
        t = seq(0, 46, 0.1)
        original_samples = eval.fd(t, rootdata)
        mean_samples = eval.fd(t, mean)
        centered_samples = eval.fd(t, centered_root)
        saveRDS(original_samples, 'original_samples.RData')
        saveRDS(mean_samples, 'mean_samples.RData')
        saveRDS(centered_samples, 'centered_samples.RData')

    ''' % (r_data.r_repr(), maxX, number_frames, numknots, nharm)
    robjects.r(rcode)
    try:
        shutil.copyfile('functionalData.RData', filename)
        os.remove('functionalData.RData')
    except:
        raise IOError('no existing file or file path is wrong')


def plot_all():
    readRDS = robjects.r['readRDS']
    
    dirdata = get_output_folder()
    dirdata = clean_path(dirdata)
    
    original_samples = readRDS(os.sep.join((dirdata, 'original_samples.RData')))
    original_samples = np.array(original_samples)
    mean_samples = readRDS(os.sep.join((dirdata, 'mean_samples.RData')))
    mean_samples = np.array(mean_samples)
    centered_samples = readRDS(os.sep.join((dirdata, 'centered_samples.RData')))
    centered_samples = np.array(centered_samples)

    # plot original  
    plt.figure()
    plt.plot(original_samples[:,:,0],original_samples[:,:,1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('origial Splines, X vs Y')
    plt.show()

    plt.figure()
    plt.plot(original_samples[:,:,0],original_samples[:,:,2])
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('origial Splines, X vs Z')
    plt.show()

    plt.figure()
    plt.plot(original_samples[:,:,1],original_samples[:,:,2])
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title('origial Splines, Y vs Z')
    plt.show()
    
    
    # plot mean    
    plt.figure()
    plt.plot(mean_samples[:,:,0],mean_samples[:,:,1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('mean Spline, X vs Y')
    plt.show()

    plt.figure()
    plt.plot(mean_samples[:,:,0],mean_samples[:,:,2])
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('mean Spline, X vs Z')
    plt.show()

    plt.figure()
    plt.plot(mean_samples[:,:,1],mean_samples[:,:,2])
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title('mean Spline, Y vs Z')
    plt.show()

    
    
    # plot centered    
    plt.figure()
    plt.plot(centered_samples[:,:,0],centered_samples[:,:,1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('centered Spline, X vs Y')
    plt.show()

    plt.figure()
    plt.plot(centered_samples[:,:,0],centered_samples[:,:,2])
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('centered Spline, X vs Z')
    plt.show()

    plt.figure()
    plt.plot(centered_samples[:,:,1],centered_samples[:,:,2])
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title('centered Spline, Y vs Z')
    plt.show()

def get_output_folder():
    """
    Return folder path to store result without trailing os.sep
    """
    data_dir_name = "data"
    PCA_dir_name = "2 - PCA"
    type_parameter = "spatial"
    step = "2 - function_generation"
    action = 'experiments'
    test_feature = "1 - FPCA with absolute joint positions in Cartesian space"
    output_dir = os.sep.join([ROOT_DIR,
                              data_dir_name,
                              PCA_dir_name,
                              type_parameter,
                              step,
                              action,
                              test_feature])
    return output_dir


if __name__ == '__main__':
    save_path = get_output_folder()
    if len(save_path) > 116:
        save_path = clean_path(save_path)
    elementary_action = 'walk'
    motion_primitive = 'leftStance'
    #function_generator(elementary_action, motion_primitive, save_path)
    plot_all()