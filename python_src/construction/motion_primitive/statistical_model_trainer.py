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
import sys
ROOTDIR = os.sep.join(['..']*2)
sys.path.append(ROOTDIR)
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
import shutil
from utilities.bvh import BVHReader, BVHWriter
ROOT_DIR = os.sep.join([".."] * 3)

class StatisticalModelTrainer(object):

    def __init__(self, fdata, save_path=None):


        self._load_spatial_data(fdata)
        self._combine_spatial_temporal_parameters()
    
    def gen_motion_primitive_model(self):
        self._train_gmm()
        self._create_gmm()
        self._save_model(save_path=save_path)

    def _load__data(self, fdata):
        '''
        Load dimensional representation for motion segements from a json file

        Parameters
        ----------
        * data: json file
        \tThe data is stored in a dictionary
        '''
        
        self._motion_primitive_name = fdata['motion_type']
        self._spatial_parameters = fdata['spatial_parameters']
        self._spatial_eigenvectors = fdata['spatial_eigenvectors']
        self._n_frames = int(fdata['n_frames'])
        self._scale_vec = fdata['scale_vector']
        self._n_basis = fdata['n_basis']
        self._mean_motion = fdata['mean_motion']
        self._n_dim_spatial = int(fdata['n_dim_spatial'])
        self._temporal_pca = fdata['temporal_pcaobj']
        self._temporal_parameters = np.asarray(self._temporal_pca[self._temporal_pca.names.index('scores')])        


    def _weight_temporal_parameters(self):
        '''
        Weight low dimensional temporal parameters
        '''
        weight_matrix = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        self._temporal_parameters = np.dot(self._temporal_parameters,
                                           weight_matrix)

    def _combine_spatial_temporal_parameters(self):
        '''
        Concatenate temporal and spatial paramters of same motion sample as a 
        long vector
        '''
        assert self._spatial_parameters.shape[0] == \
               self._temporal_parameters.shape[0], ('Number of samples are not the same for spatial parameters and temporal parameters')
        self._weight_temporal_parameters()
        self._motion_parameters = np.concatenate((self._spatial_parameters,
                                                  self._temporal_parameters,),
                                                 axis=1)

    def _train_gmm(self, n_K=10, DEBUG=0):
        '''
        Find the best number of Gaussian using BIC score
        '''

        obs = np.random.permutation(self._motion_parameters)
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
#        self.gmm.fit(self._spatial_parameters)
#        scores = self.gmm.score(self._spatial_parameters)
        if self.use_temporal_parameter:
            self.gmm.fit(self._motion_parameters)
            scores = self.gmm.score(self._motion_parameters)
        else:
            self.gmm.fit(self._spatial_parameters)
            scores = self.gmm.score(self._spatial_parameters)
#        print scores
        averageScore = np.mean(scores)
        print 'average score is:' + str(averageScore)
    
    def _sample_spatial_parameters(self, n, save_path=None):
        '''Generate ranmdon sample from mrophable model based on spatial p
           parameters
        '''
        self.new_ld_vectors = self.gmm.sample(n)
        for i in xrange(n):
            filename = 'generated_motions' + os.sep + str(i) + '.bvh'
            self._backprojection(self.new_ld_vectors[i], filename = filename)
    
    def _sample_fd_spatial_parameters(self, n, save_path=None):
        self.new_fd_ld_vectors = self.gmm.sample(n)
        
    
    def _backprojection(self, ld_vec, filename=None):
        """Back project a low dimensional spatial parameter to motion
        """
        eigenVectors = np.array(self._spatial_eigenvectors)
        backprojected_vector = np.dot(np.transpose(eigenVectors), ld_vec.T)
        backprojected_vector = np.ravel(backprojected_vector)
        backprojected_vector += self._mean_motion
        # reshape motion vector as a 2d array n_frames * n_dim
        assert len(backprojected_vector) == self._n_dim_spatial * self._n_frames, ('the length of back projected motion vector is not correct!')
        if filename is None:
            filename = 'sample.bvh'
        else:
            filename = filename
        frames = np.reshape(backprojected_vector, (self._n_frames, 
                                                   self._n_dim_spatial))
        # rescale root position for each frame
        for i in xrange(self._n_frames):
            frames[i, 0] = frames[i, 0] * self._scale_vec[0]
            frames[i, 1] = frames[i, 1] * self._scale_vec[1]
            frames[i, 2] = frames[i, 2] * self._scale_vec[2]
        skeleton = os.sep.join(('lib', 'skeleton.bvh'))
        reader = BVHReader(skeleton)                                                   
        BVHWriter(filename, reader, frames, frame_time=0.013889,
                  is_quaternion=True)                                                      
                                                           
                                                   
    def _save_model(self, save_path=None):
        '''
        Save model as a json file

        Parameters
        ----------
        *filename: string
        \tGive the file name to json file
        '''
        if save_path is None:
            filename = self._motion_primitive_name + '_quaternion_mm.json'
        else:
            filename = save_path + os.sep + self._motion_primitive_name + '_quaternion_mm.json'
        weights = self.gmm.weights_.tolist()
        means = self.gmm.means_.tolist()
        covars = self.gmm.covars_.tolist()
        mean_fd = self._temporal_pca[self._temporal_pca.names.index('meanfd')]
        self._mean_time_vector = np.array(mean_fd[mean_fd.names.index('coefs')])
        self._mean_time_vector = np.ravel(self._mean_time_vector)
        n_basis_time = len(self._mean_time_vector)
        harms = self._temporal_pca[self._temporal_pca.names.index('harmonics')]
        self._eigen_vectors_time = np.array(harms[harms.names.index('coefs')])
        data = {'name': self._motion_primitive_name,
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
                'n_dim_spatial': self._n_dim_spatial,
                'n_basis_time': n_basis_time}
        with open(filename, 'wb') as outfile:
            json.dump(data, outfile)
        outfile.close()

def main():
    pass    
    
if __name__ == '__main__':
#    test_standPCA_on_quaternion_spatial()
    main()
