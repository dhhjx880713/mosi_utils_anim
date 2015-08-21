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
from ...animation_data.bvh import BVHReader, BVHWriter


class StatisticalModelTrainer(object):

    def __init__(self, fdata, save_path=None):

        self.load_data(fdata)
        self.comb_params()
        self.save_path = save_path
        self.n_gaussians = 1
        self.training_data = {}
        self.gmm = None

    def gen_motion_primitive_model(self):
        self._train_gmm()
        self._create_gmm()
        self._save_model(self.save_path)

    def load_data(self, fdata):
        '''
        Load dimensional representation for motion segements from a json file

        Parameters
        ----------
        * data: json file
        \tThe data is stored in a dictionary
        '''

        self.training_data['motion_primitive_name'] = fdata['motion_type']
        self.training_data['spatial_parameters'] = fdata['spatial_parameters']
        self.training_data['spatial_eigenvectors'] = fdata[
            'spatial_eigenvectors']
        self.training_data['n_frames'] = int(fdata['n_frames'])
        self.training_data['scale_vector'] = fdata['scale_vector']
        self.training_data['n_basis'] = fdata['n_basis']
        self.training_data['mean_motion'] = fdata['mean_motion']
        self.training_data['n_dim_spatial'] = int(fdata['n_dim_spatial'])
        self.training_data['temporal_pca'] = fdata['temporal_pcaobj']
        self.training_data['temporal_parameters'] = np.asarray(
            self.training_data['temporal_pca'][
                self.training_data['temporal_pca'].names.index('scores')])

    def _weight_temporal_parameters(self):
        '''
        Weight low dimensional temporal parameters
        '''
        weight_matrix = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        self.training_data['temporal_parameters'] = np.dot(
            self.training_data['temporal_parameters'],
            weight_matrix)

    def comb_params(self):
        '''
        Concatenate temporal and spatial paramters of same motion sample as a
        long vector
        '''
        assert self.training_data['spatial_parameters'].shape[0] == \
               self.training_data['temporal_parameters'].shape[0],\
            ('Number of samples are not the same for spatial parameters and temporal parameters')
        self._weight_temporal_parameters()
        self._motion_parameters = np.concatenate(
            (self.training_data['spatial_parameters'],
             self.training_data['temporal_parameters']),
            axis=1)

    def _train_gmm(self, n_gaussians=10, debug=0):
        '''
        Find the best number of Gaussian using BIC score
        '''

        obs = np.random.permutation(self._motion_parameters)
        lowest_bic = np.infty
        bic = []
        bic_scores = []
        for i in range(1, n_gaussians):
            gmm = mixture.GMM(n_components=i, covariance_type='full')
            gmm.fit(obs)
            bic.append(gmm.bic(obs))
            bic_scores.append(bic[-1])
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
        index = min(xrange(n_gaussians - 1), key=bic.__getitem__)
        print 'number of Gaussian: ' + str(index + 1)
        self.n_gaussians = index + 1
        if debug:
            plt.figure()
            plt.plot(bic_scores)
            plt.show()

    def _create_gmm(self):
        '''
        Using GMM to model the data with optimized number of Gaussian
        '''
        self.gmm = mixture.GMM(n_components=self.n_gaussians,
                               covariance_type='full')
#        self.gmm.fit(self._spatial_parameters)
#        scores = self.gmm.score(self._spatial_parameters)

        self.gmm.fit(self._motion_parameters)

    def _sample_spatial_parameters(self, n_samples):
        '''Generate ranmdon sample from mrophable model based on spatial p
           parameters
        '''
        new_ld_vectors = self.gmm.sample(n_samples)
        for i in xrange(n_samples):
            filename = 'generated_motions' + os.sep + str(i) + '.bvh'
            self._backprojection(new_ld_vectors[i], filename=filename)

    def _backprojection(self, ld_vec, filename=None):
        """Back project a low dimensional spatial parameter to motion
        """
        eigen_vectors = np.array(self.training_data['spatial_eigenvectors'])
        backprojected_vector = np.dot(np.transpose(eigen_vectors), ld_vec.T)
        backprojected_vector = np.ravel(backprojected_vector)
        backprojected_vector += self.training_data['mean_motion']
        # reshape motion vector as a 2d array n_frames * n_dim
        assert len(backprojected_vector) == self.training_data['n_dim_spatial'] * \
            self.training_data['n_frames'], \
            ('the length of back projected motion vector is not correct!')
        if filename is None:
            filename = 'sample.bvh'
        else:
            filename = filename
        frames = np.reshape(
            backprojected_vector,
            (self.training_data['n_frames'],
             self.training_data['n_dim_spatial']))
        # rescale root position for each frame
        for i in xrange(self.training_data['n_frames']):
            frames[i, 0] = frames[i, 0] * self.training_data['scale_vector'][0]
            frames[i, 1] = frames[i, 1] * self.training_data['scale_vector'][1]
            frames[i, 2] = frames[i, 2] * self.training_data['scale_vector'][2]
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
            filename = self.training_data[
                'motion_primitive_name'] + '_quaternion_mm.json'
        else:
            filename = save_path + os.sep + \
                self.training_data['motion_primitive_name'] + '_quaternion_mm.json'
        weights = self.gmm.weights_.tolist()
        means = self.gmm.means_.tolist()
        covars = self.gmm.covars_.tolist()
        mean_fd = self.training_data['temporal_pca'][
            self.training_data['temporal_pca'].names.index('meanfd')]
        mean_time_vector = np.array(
            mean_fd[mean_fd.names.index('coefs')])
        mean_time_vector = np.ravel(mean_time_vector)
        n_basis_time = len(self.training_data['temporal_pca'])
        harms = self.training_data['temporal_pca'][
            self.training_data['temporal_pca'].names.index('harmonics')]
        eigen_vectors_time = np.array(harms[harms.names.index('coefs')])
        data = {
            'name': self.training_data['motion_primitive_name'],
            'gmm_weights': weights,
            'gmm_means': means,
            'gmm_covars': covars,
            'eigen_vectors_spatial': self.training_data['spatial_eigenvectors'].tolist(),
            'mean_spatial_vector': self.training_data['mean_motion'].tolist(),
            'n_canonical_frames': self.training_data['n_frames'],
            'translation_maxima': self.training_data['scale_vector'],
            'n_basis_spatial': self.training_data['n_basis'],
            'eigen_vectors_time': eigen_vectors_time.tolist(),
            'mean_time_vector': mean_time_vector.tolist(),
            'n_dim_spatial': self.training_data['n_dim_spatial'],
            'n_basis_time': n_basis_time}
        with open(filename, 'wb') as outfile:
            json.dump(data, outfile)
        outfile.close()
