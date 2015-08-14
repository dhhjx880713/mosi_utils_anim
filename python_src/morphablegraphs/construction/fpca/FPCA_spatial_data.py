# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 10:48:53 2015

@author: hadu01
"""

from PCA_fd import PCA_fd
import numpy as np


class FPCASpatialData(object):

    def __init__(self, motion_data, n_basis, fraction):
        """
        * motion_data: dictionary
        \tContains quaternion frames of each file
        """
        self.spatial_data = motion_data
        self.n_basis = n_basis
        self.fraction = fraction

    def fpca_on_spatial_data(self):
        self.convert_data_for_fpca()
        self.fpcaobj = PCA_fd(self.reshaped_data, self.n_basis, self.fraction)

    def convert_data_for_fpca(self):
        # reorder data based on filename
        self.fileorder = sorted(self.spatial_data.keys())
        reordered_data = []
        for filename in self.fileorder:
            reordered_data.append(self.spatial_data[filename])
        reordered_data = np.array(reordered_data)
        # print reordered_data.shape
        n_samples, n_frames, n_dims = reordered_data.shape
        # in order to use Functional data representation from Fda(R), the
        # input data should be a matrix of shape (n_frames * n_samples *
        # n_dims)
        self.reshaped_data = np.zeros((n_frames, n_samples, n_dims))
        for i in xrange(n_frames):
            for j in xrange(n_samples):
                self.reshaped_data[i, j] = reordered_data[j,i]


