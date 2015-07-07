'''
Created on Nov 19, 2014

@author: hadu01
'''
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 12:47:13 2014

@author: mamauer, han
"""
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
import numpy as np
import os
import json
import shutil
# import Math.timeVariances
# import Math.misc
#import matplotlib.pylab as plt
#from mpl_toolkits.mplot3d import Axes3D
ROOT_DIR = os.sep.join([".."] * 6)


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

               
def fpca(RDataFile, nharm = 3):
    """Load functional data from RData file
    
    Parameters
    ----------
    * RDataFile: string
    \tFile contains functional data
    
    Return
    ------
    * fd: robjects
    \tFd contains functional data object
    """
    rcode = '''
        library(fda)
        fd = readRDS("%s")
        print('test')
        print(fd$rep1)
        nharm = %d
        pca = pca.fd(fd, nharm={nharm})
        pcaVarmax <- varmx.pca.fd(pca)
        scores = pcaVarmax$scores
    ''' % (RDataFile, nharm)
    robjects.r(rcode)
    pca = robjects.globalenv['pcaVarmax']
    scores = np.asarray(pca[pca.names.index('scores')])
    print 'dimension of scores: '
    print scores.shape
    return scores, pca 



# def inverse_pca(pcaobj, numberOfFrames):
#     robjects.r('library(fda)')
#     eigenfd = pcaobj[pcaobj.names.index('harmonics')]
#     meanfd = pcaobj[pcaobj.names.index('meanfd')]
#     fdeval = robjects.r['eval.fd']
     

    
if __name__== '__main__':
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
    scores, pca = fpca('functionalData.RData')
    os.remove('functionalData.RData')

    