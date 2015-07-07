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
#import rpy2.robjects.numpy2ri as numpy2ri
#import rpy2.robjects as robjects
import shutil
ROOT_DIR = os.sep.join([".."] * 3)

class StatisticalModelTrainer(object):

    def __init__(self, data, RDataFile):
        self._load_spatial_data(data)
        self._load_temporal_data(RDataFile)
        self._combine_spatial_temporal_parameters()
        self._train_gmm()
        self._create_gmm()
        self._save_model()

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
        weight_matrix = np.array([[0.0, 0, 0], [0, 0.0, 0], [0, 0, 0.0]])
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

    def _train_gmm(self, n_K=40, DEBUG=0):
        '''
        Find the best number of Gaussian using BIC score
        '''
        print 'dimension of training data: '
        print self._motion_parameters.shape
        obs = np.random.permutation(self._motion_parameters)
#        obs = np.random.permutation(self._spatial_parameters)
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
        self.gmm.fit(self._motion_parameters)
        scores = self.gmm.score(self._motion_parameters)
#        print scores
        averageScore = np.mean(scores)
        print 'average score is:' + str(averageScore)

    def _save_model(self):
        '''
        Save model as a json file

        Parameters
        ----------
        *filename: string
        \tGive the file name to json file
        '''
        filename = self._motion_primitive_name + '_quaternion_mm.json'
        weights = self.gmm.weights_.tolist()
        means = self.gmm.means_.tolist()
        covars = self.gmm.covars_.tolist()
        mean_fd = self._temporal_pca[self._temporal_pca.names.index('meanfd')]
        self._mean_time_vector = np.array(mean_fd[mean_fd.names.index('coefs')])
        self._mean_time_vector = np.ravel(self._mean_time_vector)
        harms = self._temporal_pca[self._temporal_pca.names.index('harmonics')]
        self._eigen_vectors_time = np.array(harms[harms.names.index('coefs')])
        data = {'mpe_name': self._motion_primitive_name,
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
                'n_basis_time': 40}
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
    try:
        shutil.copyfile(filename_temporal, 'functionalData.RData')
    except:
        raise IOError('no existing file or file path is wrong') 
        
#    testFile = 'walk_leftStance_low_dimensional_data.json'
#    rDataFile = 'b_splines_walk_leftStance.RData'
#    temporalDataFile = 'scores_walk_leftStance.npy'
    model = StatisticalModelTrainer(filename_spatial, 'functionalData.RData')
    os.remove('functionalData.RData')
