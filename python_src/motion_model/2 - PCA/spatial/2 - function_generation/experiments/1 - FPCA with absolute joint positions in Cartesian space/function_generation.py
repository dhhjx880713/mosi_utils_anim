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
ROOT_DIR = os.sep.join([".."] * 6)


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
                       numknots=5):
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
#    print 'the dimension of data: '
#    print data.shape
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
        basisobj = create.bspline.basis(c(0,{maxX}),{numknots})
        ys = smooth.basis(argvals=seq(0, {maxX}, len = {length}), y = {data},
                          fdParobj = basisobj)
        fd = ys$fd
        coefs = fd$coefs
        saveRDS(fd, 'functionalData.RData')
    ''' % (r_data.r_repr(), maxX, number_frames, numknots, nharm)
    robjects.r(rcode)
    coefs = robjects.globalenv['coefs']

    print type(coefs)
    try:
        shutil.copyfile('functionalData.RData', filename)
        os.remove('functionalData.RData')
    except:
        raise IOError('no existing file or file path is wrong')


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
    elementary_action = 'place'
    motion_primitive = 'second'
    function_generator(elementary_action, motion_primitive, save_path)