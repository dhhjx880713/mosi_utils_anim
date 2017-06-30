# encoding: UTF-8
import numpy as np
import scipy.interpolate as si
from ..utils import get_cubic_b_spline_knots


class FunctionalData(object):

    def __init__(self):
        self.knots = None

    def get_knots(self, n_basis, n_frames):
        self.knots = get_cubic_b_spline_knots(n_basis, n_frames)

    def convert_motion_to_functional_data(self, motion_data, n_basis=7, degree=3):
        """
        Represent motion data as functional data, motion data should be narray<2d> n_frames * n_dims,
        the functional data has the shape n_basis * n_dims
        """
        motion_data = np.asarray(motion_data)
        n_frames, n_dims = motion_data.shape
        if self.knots is None:
            self.get_knots(n_basis, n_frames)
        x = range(n_frames)
        coeffs =[si.splrep(x, motion_data[:, i], k=degree,
                            t=self.knots[degree+1: -(degree+1)])[1][:-4] for i in range(n_dims)]
        return np.asarray(coeffs).T

    def convert_motions_to_functional_data(self, motion_mat, n_basis, degree=3):
        """
        Represent motion data as functional data, motion data should be narray<3d> n_samples * n_frames * n_dims,
        the functional data has the shape n_samples * n_basis * n_dims
        """
        motion_mat = np.asarray(motion_mat)
        print('shape of motion_mat', motion_mat.shape)
        n_samples, n_frames, n_dims = motion_mat.shape
        functional_mat = np.zeros((n_samples, n_basis, n_dims))
        self.get_knots(n_basis, n_frames)
        for i in range(n_samples):
            functional_mat[i] = self.convert_motion_to_functional_data(motion_mat[i], n_basis, degree)
        return functional_mat
