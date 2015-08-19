# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 14:11:11 2015

@author: mamauer, hadu01 ,erhe01
"""

import numpy as np
import json
from sklearn import mixture # statistical model
import rpy2.robjects as robjects
from motion_primitive_sample import MotionPrimitiveSample
import scipy.interpolate as si # B-spline definition and evaluation

class MotionPrimitive(object): #StatisticalModel
    """ Represent a motion primitive which can be sampled

    Parameters
    ----------
    * filename: string
    \tThe filename with the saved data in json format.


    Attributes
    ----------
    * s_pca: dictionary
    \tThe result of the spacial PCA. It is a dictionary having the
    (eigen_vectors, mean_vectors, n_basis,n_dim, maxima, n_components) as values

    * t_pca: dictionary
    \tThe result of the temporal PCA. It is a dictionary having the
    (eigen_vectors, mean_vectors, n_basis,n_dim) as values

    * gmm: sklearn.mixture.GMM
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
        self.s_pca = {}
        self.t_pca = {}
        # information about the motion necessary for the reconstruction
        self.n_canonical_frames = 0
        self.translation_maxima = np.array([1.0,1.0,1.0])
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

    def _initialize_from_json(self,data):
        """ Load morphable model parameters from a dictionary and initialize
            the fda library and the Gaussian Mixture model.

        Parameters
        ----------
        * data: dictionary
        \tThe dictionary must contain all parameters for the motion primitive

        """

        robjects.r('library("fda")')  #initialize fda for later operations
        
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
        self.s_pca = {}
        self.s_pca["eigen_vectors"] = np.array(data['eigen_vectors_spatial'])
        self.s_pca["mean_vector"] = np.array(data['mean_spatial_vector'])
        self.s_pca["n_basis"]= int(data['n_basis_spatial'])
        self.s_pca["n_dim"] = int(data['n_dim_spatial'])
        self.s_pca["n_components"]= len(self.s_pca["eigen_vectors"])

        rcode ="""
            n_basis = %d
            n_frames = %d
            basisobj = create.bspline.basis(c(0, n_frames - 1), nbasis = n_basis)
        """% ( self.s_pca["n_basis"],self.n_canonical_frames)
        robjects.r(rcode)
        self.s_pca["basis_function"] = robjects.globalenv['basisobj']
        self.s_pca["knots"] = np.asarray(robjects.r['knots'](self.s_pca["basis_function"],False))

    def _init_time_parameters_from_json(self, data):
        """  Set the parameters for the _inverse_temporal_pca function.

        Parameters
        ----------
        * data: dictionary
        \tThe dictionary must contain all parameters for the spatial pca.

        """
        self.t_pca = {}
        self.t_pca["eigen_vectors"] = np.array(data['eigen_vectors_time'])
        self.t_pca["mean_vector"]= np.array(data['mean_time_vector'])
        self.t_pca["n_basis"]= int(data['n_basis_time'])
        self.t_pca["n_dim"] = 1
        self.t_pca["n_components"]= len(self.t_pca["eigen_vectors"].T)

        rcode ="""
            n_basis = %d
            n_frames = %d
            basisobj = create.bspline.basis(c(0, n_frames - 1), nbasis = n_basis)
        """% ( self.t_pca["n_basis"],self.n_canonical_frames)
        robjects.r(rcode)
        self.t_pca["basis_function"] = robjects.globalenv['basisobj']
        self.t_pca["knots"] = np.asarray(robjects.r['knots'](self.t_pca["basis_function"],False))
        self.t_pca["eigen_coefs"] =zip(* self.t_pca["eigen_vectors"])

    def sample(self, return_lowdimvector=False):#todo make it two functions
        """ Sample the motion primitive and return a motion sample

        Parameters
        ----------
        *return_lowdimvector: boolean
        \tIf True, return the s vector, else return a MotionSample object
        Returns
        -------
        * motion: MotionSample or numpy.ndarray
        \tThe sampled motion as object of type MotionSample or numpy.ndarray \
        (Depending on parameter)
        """
        assert self.gaussian_mixture_model is not None, "Motion primitive not initialized."
        low_dimensional_vector = np.ravel(self.gaussian_mixture_model.sample())
        if return_lowdimvector:
            return low_dimensional_vector
        else:
            return self.back_project(low_dimensional_vector)

    def back_project(self,low_dimensional_vector,use_time_parameters=True):
        """ Return a motion sample based on a low dimensional motion vector.

        Parameters
        ----------
        *low_dimensional_vector: numpy.ndarray
        \tThe low dimensional motion representation sampled from a GMM or GP
        *return_lowdimvector: boolean
        \tIf True, return the s vector, else return a MotionSample object
        Returns
        -------
        * motion: MotionSample or numpy.ndarray
        \tThe sampled motion as object of type MotionSample
        """
        spatial_coefs = self._inverse_spatial_pca(low_dimensional_vector[:self.s_pca["n_components"]])
        if self.has_time_parameters and use_time_parameters:
            time_coefs = low_dimensional_vector[self.s_pca["n_components"]:]
            time_coefs = [i*10 for i in time_coefs]
            time_function = self._inverse_temporal_pca(time_coefs)
        else:
            time_function = np.arange(0,self.n_canonical_frames)
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

        coefs = coefs.reshape((self.s_pca["n_basis"],self.s_pca["n_dim"]))
        #undo the scaling on the translation
        coefs[:,0] *= self.translation_maxima[0]
        coefs[:,1] *= self.translation_maxima[1]
        coefs[:,2] *= self.translation_maxima[2]
        return coefs

    def _mean_temporal(self):
        """Evaluates the mean time b-spline for the canonical time range.
        Returns
        -------
        * mean_t: np.ndarray
            Discretized mean time function.
        """
        mean_tck = (self.t_pca["knots"], self.t_pca["mean_vector"], 3)
        return si.splev(self.canonical_time_range,mean_tck)

    def _get_monotonic_indices(self, indices, epsilon=0.01, delta=0):
        """Return an ajusted set of Frameindices which is strictly monotonic

        Parameters
        ----------
        indices : list
        The Frameindices

        Returns
        -------
        A numpy-Float Array with indices similar to the provided list,
        but enforcing strict monotony
        """
        shifted_indices = np.array(indices, dtype=np.float)
        if shifted_indices[0] == shifted_indices[-1]:
            raise ValueError("First and Last element are equal")

        for i in xrange(1, len(shifted_indices) - 1):
            if shifted_indices[i] > shifted_indices[i - 1] + delta:
                continue

            while np.allclose(shifted_indices[i], shifted_indices[i - 1]) or \
                    shifted_indices[i] <= shifted_indices[i - 1] + delta:
                shifted_indices[i] = shifted_indices[i] + epsilon

        for i in xrange(len(indices) - 2, 0, -1):
            if shifted_indices[i] + delta < shifted_indices[i + 1]:
                break

            while np.allclose(shifted_indices[i], shifted_indices[i + 1]) or \
                    shifted_indices[i] + delta >= shifted_indices[i + 1]:
                shifted_indices[i] = shifted_indices[i] - epsilon

        return shifted_indices

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
        # step 1:backtransform gamma to a discrete timefunction 
        #1.1: reconstruct t by evaluating the harmonics and the mean

        mean_t = self._mean_temporal()
        print "mean temporal: "
        print mean_t
        n_latent_dim = len(self.t_pca["eigen_coefs"])
        print "n_latent_dim: "
        print n_latent_dim
        print self.t_pca["knots"]
        print self.t_pca["eigen_coefs"]
        eigen_tck = [(self.t_pca["knots"],self.t_pca["eigen_coefs"][i],3) for i in xrange(n_latent_dim)]
        eigen_t =np.array([ si.splev(self.canonical_time_range,tck) for tck in eigen_tck]).T

        t=[0,]
        for i in xrange(self.n_canonical_frames):
            t.append(t[-1] + np.exp(mean_t[i] + np.dot(eigen_t[i], gamma)))
        print "#################################################################"
        #1.2: undo step from timeVarinaces.transform_timefunction during alignment
        t = np.array(t[1:])
        t -= 1
        zeroindices = t < 0
        t[zeroindices] = 0
        t = self._get_monotonic_indices(t)
        # step 2: calculate inverse spline and then sample that inverse spline
        # using step size 1
        # i.e. calculate t'(t) from t(t')

        #2.1 get a valid inverse spline
        x_sample = np.arange(self.n_canonical_frames)
        print x_sample
        inverse_spline = si.splrep(t, x_sample,w=None, k=3)

        
        #2.2 sample discrete data from inverse spline
        # Note: t gets inverted. Before, t mapped from canonical to sample time, now
        # from sample to canonical time
        frames = np.linspace(1, t[-2], np.round(t[-2]))# note this is where the number of frames are changed
        t = si.splev(frames,inverse_spline)

        t = np.insert(t, 0, 0)
        t = np.insert(t, len(t), self.n_canonical_frames-1)
        return t

