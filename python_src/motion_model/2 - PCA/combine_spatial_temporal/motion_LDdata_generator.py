# -*- coding: utf-8 -*-
"""
Created on Mon Feb 02 15:17:50 2015

@author: hadu01
"""
import numpy as np
import json
import os
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
import shutil
ROOT_DIR = os.sep.join([".."] * 3)


class MotionLDdataGenerator(object):
    '''
    Load sptaial and temporal parameters from fpca, and generator concatenated
    motion vector for statistical modeling
    '''
    def __init__(self, spatial_data, temporal_data):
        self._load_spatial_data(spatial_data)
        self._load_temporal_data(temporal_data)
        self._combine_spatial_temporal_parameters()
        self._save_parameters()

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
        self._n_basis = int(tmp['n_basis'])
        self._mean_motion = tmp['mean_motion']
        self._n_dim_spatial = int(tmp['n_dim_spatial'])
        self._file_order = tmp['file_order']

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
        mean_fd = self._temporal_pca[self._temporal_pca.names.index('meanfd')]
        self._mean_time_vector = np.array(mean_fd[mean_fd.names.index('coefs')])        
        harms = self._temporal_pca[self._temporal_pca.names.index('harmonics')]        
        self._temporal_eigenvectors = np.array(harms[harms.names.index('coefs')])
        
    def _weight_temporal_parameters(self):
        '''
        Weight low dimensional temporal parameters
        '''
        weight_matrix = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        self._temporal_parameters = np.dot(self._temporal_parameters,
                                           weight_matrix)        
                                          
    def _combine_spatial_temporal_parameters(self):
        '''
        Concatenate temporal and spatial paramters of same motion sample as a 
        long vector
        '''
        assert self._spatial_parameters.shape[0] == \
               self._temporal_parameters.shape[0], ('Number of samples are the same for spatial parametersand temporal parameters')
#        self._weight_temporal_parameters()
        self._motion_parameters = np.concatenate((self._spatial_parameters,
                                                  self._temporal_parameters,),
                                                 axis=1)
    
    def _save_parameters(self):
        '''
        Save the parameters as json file
        '''
        filename = self._motion_primitive_name + \
                 '_low_dimensional_motion_data.json'
        data = {
            'motion_type': self._motion_primitive_name,
            'motion_data': self._motion_parameters.tolist(),
            'eigen_vectors_spatial': self._spatial_eigenvectors,
            'eigen_vectors_temporal': self._temporal_eigenvectors.tolist(),
            'mean_spatial_vector': self._mean_motion,
            'mean_temporal_vector': self._mean_time_vector.tolist(),
            'translation_maxima': self._scale_vec,
            'n_basis_spatial': self._n_basis,
            'n_canonical_frames': self._n_frames,
            'n_basis_temporal': 8,
            'file_order': self._file_order
        }
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

if __name__ == '__main__':
    input_dir_spatial = get_input_folder_spatial()
    if len(input_dir_spatial) > 116:  # avoid too long path in windows
        input_dir_spatial = clean_path(input_dir_spatial)
    elementary_action = 'walk'
    motion_primitive = 'rightStance'
    filename_spatial = input_dir_spatial + os.sep + '%s_%s_low_dimensional_data.json' % \
        (elementary_action, motion_primitive)
    input_dir_temporal = get_input_folder_temporal()
    if len(input_dir_temporal) > 116:  # avoid too long path in windows
        input_dir_temporal = clean_path(input_dir_temporal)
    filename_temporal = input_dir_temporal + os.sep + 'b_splines_%s_%s.RData' % \
        (elementary_action, motion_primitive)
    print filename_temporal
    try:
        shutil.copyfile(filename_temporal, 'functionalData.RData')
    except:
        raise IOError('no existing file or file path is wrong') 
    LDdata_generator = MotionLDdataGenerator(filename_spatial, 'functionalData.RData')
    os.remove('functionalData.RData')