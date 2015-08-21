# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 14:11:11 2015

@author: mamauer, hadu01 ,erhe01
"""

import numpy as np
import json
from sklearn import mixture
import scipy.interpolate as si
from . import B_SPLINE_DEGREE
from motion_primitive_sample import MotionPrimitiveSample


class MotionPrimitive(object):
    """ Represent a motion primitive which can be sampled
    Parameters
    ----------
    * filename: string
    \tThe filename with the saved data in json format.
    Attributes
    ----------
    * s_pca: dictionary
    \tThe result of the spacial PCA. It is a dictionary having the
    (eigen_vectors, mean_vectors, n_basis,n_dim, maxima, n_components, knots) as values
    * t_pca: dictionary
    \tThe result of the temporal PCA. It is a dictionary having the
    (eigen_vectors, mean_vectors, n_basis,n_dim, knots) as values
    * gaussian_mixture_model: sklearn.mixture.GMM
    \tstatistical model on the low dimensional representation of motion samples
    *name: string
    \tIdentifier of the motion primitive
    *n_canonical_frames: int
    \tNumber of frames in the canonical timeline of the spatial data
    *translation_maxima: numpy.ndarray
    \tScaling factor to reconstruct the unnormalized translation parameters of a motion after inverse pca
    """
    def __init__(self, filename):
        self.filename = filename
        self.name = ""
        self.gaussian_mixture_model = None
        self.s_pca = dict()
        self.t_pca = dict()
        self.n_canonical_frames = 0
        self.translation_maxima = np.array([1.0, 1.0, 1.0])
        self.has_time_parameters = True
        if self.filename is not None:
            self._load(self.filename)

    def _load(self, filename=None):
        """ Load a motion primitive from a file

        Parameters
        ----------
        * filename: string, optinal
        \tThe path to the saved json file. If None (default) the filename
        of the object will be taken.
        """

        with open(filename, 'rb') as infile:
            tmp = json.load(infile)
            infile.close()
            self._initialize_from_json(tmp)

    def _initialize_from_json(self, data):
        """ Load morphable model parameters from a dictionary and initialize
            the fda library and the Gaussian Mixture model.

        Parameters
        ----------
        * data: dictionary
        \tThe dictionary must contain all parameters for the motion primitive

        """
        #load name data and canonical frames of the motion
        if 'name' in data.keys():
            self.name = data['name']
        self.n_canonical_frames = data['n_canonical_frames']
        self.canonical_time_range = np.arange(0, self.n_canonical_frames)
        # initialize parameters for the motion sampling and back projection
        self._init_gmm_from_json(data)
        self._init_spatial_parameters_from_json(data)
        if 'eigen_vectors_time' in data.keys():
            self._init_time_parameters_from_json(data)
        else:
            self.has_time_parameters = False

    def _init_gmm_from_json(self, data):
        """ Initialize the Gaussian Mixture model.

        Parameters
        ----------
        * data: dictionary
        \tThe dictionary must contain all parameters for the Gaussian Mixture Model.

        """
        n_components = len(np.array(data['gmm_weights']))
        self.gaussian_mixture_model = mixture.GMM(n_components, covariance_type='full')
        self.gaussian_mixture_model.weights_ = np.array(data['gmm_weights'])
        self.gaussian_mixture_model.means_ = np.array(data['gmm_means'])
        self.gaussian_mixture_model.converged_ = True
        self.gaussian_mixture_model.covars_ = np.array(data['gmm_covars'])

    def _init_spatial_parameters_from_json(self, data):
        """  Set the parameters for the _inverse_spatial_pca function.

        Parameters
        ----------
        * data: dictionary
        \tThe dictionary must contain all parameters for the spatial pca.

        """
        self.translation_maxima = np.array(data['translation_maxima'])
        self.s_pca = dict()
        self.s_pca["eigen_vectors"] = np.array(data['eigen_vectors_spatial'])
        self.s_pca["mean_vector"] = np.array(data['mean_spatial_vector'])
        self.s_pca["n_basis"] = int(data['n_basis_spatial'])
        self.s_pca["n_dim"] = int(data['n_dim_spatial'])
        self.s_pca["n_components"] = len(self.s_pca["eigen_vectors"])
        self.s_pca["knots"] = np.asarray(data['b_spline_knots_spatial'])

    def _init_time_parameters_from_json(self, data):
        """  Set the parameters for the _inverse_temporal_pca function.

        Parameters
        ----------
        * data: dictionary
        \tThe dictionary must contain all parameters for the spatial pca.
        """
        self.t_pca = dict()
        self.t_pca["eigen_vectors"] = np.array(data['eigen_vectors_time'])
        self.t_pca["mean_vector"] = np.array(data['mean_time_vector'])
        self.t_pca["n_basis"] = int(data['n_basis_time'])
        self.t_pca["n_dim"] = 1
        self.t_pca["n_components"] = len(self.t_pca["eigen_vectors"].T)
        self.t_pca["knots"] = np.asarray(data['b_spline_knots_time'])
        self.t_pca["eigen_coefs"] = zip(* self.t_pca["eigen_vectors"])

    def sample_low_dimensional_vector(self):
        """ Sample the motion primitive and return a low dimensional vector
        Returns
        -------
        * s:  numpy.ndarray
        """
        assert self.gaussian_mixture_model is not None, "Motion primitive not initialized."
        return np.ravel(self.gaussian_mixture_model.sample())

    def sample(self):
        """ Sample the motion primitive and return a motion sample
        Returns
        -------
        * motion: MotionSample
        \tThe sampled motion as object of type MotionPrimitiveSample
        """
        return self.back_project(self.sample_low_dimensional_vector())

    def back_project(self, low_dimensional_vector, use_time_parameters=True):
        """ Return a motion sample based on a low dimensional motion vector.

        Parameters
        ----------
        *low_dimensional_vector: numpy.ndarray
        \tThe low dimensional motion representation sampled from a GMM or GP
        *use_time_parameters: boolean
        \tIf True use time function from _inverse_temporal_pca else canonical time line
        Returns
        -------
        * motion: MotionSample
        \tThe sampled motion as object of type MotionSample
        """
        spatial_coefs = self._inverse_spatial_pca(low_dimensional_vector[:self.s_pca["n_components"]])
        if self.has_time_parameters and use_time_parameters:
            time_function = self._inverse_temporal_pca(low_dimensional_vector[self.s_pca["n_components"]:])
        else:
            time_function = np.arange(0, self.n_canonical_frames)
        return MotionPrimitiveSample(low_dimensional_vector, spatial_coefs, time_function, self.s_pca["knots"])


    def _inverse_spatial_pca(self, alpha):
        """ Backtransform a lowdimensional vector alpha to a coefficients of
        a functional motion representation.

        Parameters
        ----------
        * alpha: numpy.ndarray
        \tThe lowdimensional vector

        Returns
        -------
        * motion: numpy.ndarray
        \t Reconstructed coefficients of the functional motion representation.
        """
        #reconstruct coefs of the functionial representation
        coefs = np.dot(np.transpose(self.s_pca["eigen_vectors"]), alpha.T)
        coefs += self.s_pca["mean_vector"]
        coefs = coefs.reshape((self.s_pca["n_basis"], self.s_pca["n_dim"]))
        #undo the scaling on the translation
        coefs[:, 0] *= self.translation_maxima[0]
        coefs[:, 1] *= self.translation_maxima[1]
        coefs[:, 2] *= self.translation_maxima[2]
        return coefs

    def _mean_temporal(self):
        """Evaluates the mean time b-spline for the canonical time range.
        Returns
        -------
        * mean_t: np.ndarray
            Discretized mean time function.
        """
        mean_tck = (self.t_pca["knots"], self.t_pca["mean_vector"], 3)
        return si.splev(self.canonical_time_range, mean_tck)

    def _inverse_temporal_pca(self, gamma):
        """ Backtransform a lowdimensional vector gamma to the timewarping
        function t(t') and inverse it to t'(t).

        Parameters
        ----------
        * gamma: numpy.ndarray
        \tThe lowdimensional vector

        Returns
        -------
        * time_function: numpy.ndarray
        \tThe indices of the timewarping function t'(t)
        """
        canonical_time_function = self._back_transform_gamma_to_canonical_time_function(gamma)
        sample_time_function = self._invert_canonical_to_sample_time_function(canonical_time_function)
        return sample_time_function

    def _back_transform_gamma_to_canonical_time_function(self, gamma):
        """backtransform gamma to a discrete timefunction reconstruct t by evaluating the harmonics and the mean
        """
        mean_t = self._mean_temporal()
        n_latent_dim = len(self.t_pca["eigen_coefs"])
        t_eigen_spline = [(self.t_pca["knots"], self.t_pca["eigen_coefs"][i], B_SPLINE_DEGREE) for i in xrange(n_latent_dim)]
        t_eigen_discrete = np.array([si.splev(self.canonical_time_range, spline_definition) for spline_definition in t_eigen_spline]).T
        canonical_time_function = [0]
        for i in xrange(self.n_canonical_frames):
            canonical_time_function.append(canonical_time_function[-1] + np.exp(mean_t[i] + np.dot(t_eigen_discrete[i], gamma)))
        # undo step from timeVarinaces.transform_timefunction during alignment
        canonical_time_function = np.array(canonical_time_function[1:])
        canonical_time_function -= 1.0
        return canonical_time_function

    def _invert_canonical_to_sample_time_function(self, canonical_time_function):
        """ calculate inverse spline and then sample that inverse spline
            # i.e. calculate t'(t) from t(t')
        """
        # 1 get a valid inverse spline
        x_sample = np.arange(self.n_canonical_frames)
        sample_time_spline = si.splrep(canonical_time_function, x_sample, w=None, k=B_SPLINE_DEGREE)
        # 2 sample discrete data from inverse spline
        # canonical_time_function gets inverted to map from sample to canonical time
        frames = np.linspace(1, stop=canonical_time_function[-2], num=np.round(canonical_time_function[-2]))
        sample_time_function = si.splev(frames, sample_time_spline)
        sample_time_function = np.insert(sample_time_function, 0, 0)
        sample_time_function = np.insert(sample_time_function, len(sample_time_function), self.n_canonical_frames-1)
        return sample_time_function

