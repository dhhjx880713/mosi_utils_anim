# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 10:48:53 2015

@author: hadu01
"""

from .pca_functional_data import PCAFunctionalData
from .functional_data import FunctionalData
import numpy as np


class FPCASpatialData(object):

    def __init__(self, n_basis, n_components=None, fraction=0.95):
        """
        * motion_data: dictionary
        \tContains quaternion frames of each file
        """
        self.n_basis = n_basis
        self.fraction = fraction
        self.reshaped_data = None
        self.fpcaobj = None
        self.fileorder = None
        self.n_components = n_components

    def fit_motion_dictionary(self, motion_dic):
        self.fileorder = list(motion_dic.keys())
        self.fit(np.asarray(list(motion_dic.values())))

    def fit(self, motion_data):
        """
        Reduce the dimension of motion data using Functional Principal Component Analysis
        :param motion_data (numpy.array<3d>): can be either spatial data or temporal data, n_samples * n_frames * n_dims
        :return lowVs (numpy.array<2d>): low dimensional representation of motion data
        """
        assert len(motion_data.shape) == 3
        self.fpcaobj = PCAFunctionalData(motion_data,
                                         n_basis=self.n_basis,
                                         fraction=self.fraction,
                                         n_pc=self.n_components)

