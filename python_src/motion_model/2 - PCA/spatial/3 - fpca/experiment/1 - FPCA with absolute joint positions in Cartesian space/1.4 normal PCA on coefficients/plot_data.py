# -*- coding: utf-8 -*-
"""
Created on Do Jan 08 10:06:02 2015

@author: mamauer, hadu01
"""


import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
import os
import glob
import numpy as np
import matplotlib.pylab as plt
ROOT_DIR = os.sep.join([".."] * 7)


def get_input_data_folder():
    """
    Return folder path of feature data without trailing os.sep
    """
    data_dir_name = "src"   # Use the src folder, since the R Code is not cross path yet.
                            # TODO: Fix this.
    PCA_dir_name = "2 - PCA"
    type_parameter = "spatial"
    step = "3 - fpca"
    action = 'experiment'   # Inconsistent naming here...
    test_feature = "1 - FPCA with absolute joint positions in Cartesian space"
    subtest = "1.3 fpca with pca on scores"
    input_dir = os.sep.join([ROOT_DIR,
                             data_dir_name,
                             PCA_dir_name,
                             type_parameter,
                             step,
                             action,
                             test_feature,
                             subtest])
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
    
def plot_rdata_in_folder(folder, dim=0):
    files = glob.glob(folder[4:] + os.sep + '*.RData')
    
    data = {}
    
    readRDS = robjects.r['readRDS']
    for f in files:
        name = f.split(os.sep)[-1]
        data[name] = readRDS(f)
        data[name] = np.array(data[name])
    
    for key in data:
        plt.plot(data[key][:,dim], label=key)
    plt.legend()
    plt.show()
        
    
def main():
    path = clean_path(get_input_data_folder())
    plot_rdata_in_folder(path)


if __name__=='__main__':
    main()