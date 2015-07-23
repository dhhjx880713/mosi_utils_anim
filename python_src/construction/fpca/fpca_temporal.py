# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 12:47:13 2014

@author: mamauer, han
"""
import rpy2.robjects.numpy2ri as numpy2ri
import shutil
import rpy2.robjects as robjects
import numpy as np
import os
ROOT_DIR = os.sep.join([".."] * 5)

    
def get_input_folder():
    """
    Return input folder path of functional data
    """
    data_dir_name = "data"
    PCA_dir_name = "2 - PCA"
    type_parameter = "temporal"
    step = "2 - b_splines"
    action = 'experiments'
    input_dir = os.sep.join([ROOT_DIR,
                              data_dir_name,
                              PCA_dir_name,
                              type_parameter,
                              step,
                              action])
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
#        print(fd)
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

def fpca_temporal(fd, nharm = 3):
    rcode = '''
        library(fda)
        fd = %s
#        print(fd)
        nharm = %d
        pca = pca.fd(fd, nharm={nharm})
        pcaVarmax <- varmx.pca.fd(pca)
        scores = pcaVarmax$scores
    ''' % (fd, nharm)
    robjects.r(rcode)
    fpca = robjects.globalenv['pcaVarmax']
    return fpca    


def get_output_folder():
    """
    Return folder path to store result without trailing os.sep
    """
    data_dir_name = "data"
    PCA_dir_name = "2 - PCA"
    type_parameter = "temporal"
    step = "3 - fpca__result"
    action = 'experiments'
    output_dir = os.sep.join([ROOT_DIR,
                              data_dir_name,
                              PCA_dir_name,
                              type_parameter,
                              step,
                              action])
    return output_dir
    
def main():    
    input_dir = get_input_folder()
    if input_dir > 116:  # avoid too long path in windows
        input_dir = clean_path(input_dir)
    elementary_action = 'carryBoth'
    motion_primitive = 'turningRightStance'
    filename = input_dir + os.sep + 'b_splines_%s_%s.rds' % \
        (elementary_action, motion_primitive)
    print filename
    try:
        shutil.copyfile(filename, 'functionalData.RData')
    except:
        raise IOError('no existing file or file path is wrong')
    scores, pca = fpca('functionalData.RData')
    os.remove('functionalData.RData')
    outputfilename = 'b_splines_%s_%s.RData' % (elementary_action, motion_primitive)
    outPutDir = get_output_folder()
    outputfile = outPutDir + os.sep + outputfilename
    saveRDS = robjects.r('saveRDS')
    saveRDS(pca, outputfile)
    
if __name__== '__main__':
    main()