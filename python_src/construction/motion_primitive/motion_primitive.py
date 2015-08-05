# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 14:11:11 2015

@author: mamauer, hadu01 ,erhe01
"""

import numpy as np
import json
from sklearn import mixture
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
import time
from motion_sample import MotionSample
from scipy.interpolate import UnivariateSpline
from lib.bvh import BVHReader, BVHWriter
import os
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
np.random.seed(int(time.time()))


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
        self.name = ""  # useful for identifying the data source
        self.gmm = None  # gaussian mixture model
        self.s_pca = {}  # pca result on spatial data
        self.t_pca = {}  # pca result on time data

        # information about the motion necessary for the reconstruction
        self.n_canonical_frames = 0
        self.translation_maxima = np.array([1.0, 1.0, 1.0])
        self.use_time_parameters = True

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

        # initialize fda for later operations
        robjects.r('library("fda")')

        # load additional data
        if 'name' in data.keys():
            self.name = data['name']
        self.n_canonical_frames = data['n_canonical_frames']
        self.translation_maxima = np.array(data['translation_maxima'])

        # initialize gmm
        n_components = len(np.array(data['gmm_weights']))
        self.gmm = mixture.GMM(n_components, covariance_type='full')
        self.gmm.weights_ = np.array(data['gmm_weights'])
        self.gmm.means_ = np.array(data['gmm_means'])
        self.gmm.converged_ = True
        self.gmm.covars_ = np.array(data['gmm_covars'])

        # load spatial parameters
        self.s_pca = {}
        self.s_pca["eigen_vectors"] = np.array(data['eigen_vectors_spatial'])
        self.s_pca["mean_vector"] = np.array(data['mean_spatial_vector'])
        self.s_pca["n_basis"] = int(data['n_basis_spatial'])
        self.s_pca["n_dim"] = int(data['n_dim_spatial'])
        self.s_pca["n_components"] = len(self.s_pca["eigen_vectors"])
        rcode = """
            n_basis = %d
            n_frames = %d
            basisobj = create.bspline.basis(c(0, n_frames - 1), nbasis = n_basis)
        """ % ( self.s_pca["n_basis"], self.n_canonical_frames)
        robjects.r(rcode)
        self.s_pca["basis_function"] = robjects.globalenv['basisobj']

        # load time parameters
        if 'eigen_vectors_time' not in data.keys():
            self.use_time_parameters = False
        else:
            self.t_pca = {}
            self.t_pca["eigen_vectors"] = np.array(data['eigen_vectors_time'])
            self.t_pca["mean_vector"] = np.array(data['mean_time_vector'])
            self.t_pca["n_basis"] = int(data['n_basis_time'])
            self.t_pca["n_dim"] = 1

            rcode = """
                n_basis = %d
                n_frames = %d
                basisobj = create.bspline.basis(c(0, n_frames - 1), nbasis = n_basis)
            """ % ( self.t_pca["n_basis"], self.n_canonical_frames)
            robjects.r(rcode)
            self.t_pca["basis_function"] = robjects.globalenv['basisobj']

    def sample(self, return_lowdimvector=False):
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
        assert self.gmm != None, "Motion primitive not initialized."
        low_dimensional_vector = np.ravel(self.gmm.sample())
#        n = len(low_dimensional_vector)
#        spatial_len = n -3
#        print 'len of low dimensional vector' + str(len(low_dimensional_vector))
#        low_dimensional_vector_spatial = low_dimensional_vector
#        return low_dimensional_vector_spatial

        print low_dimensional_vector

        if return_lowdimvector:
            return low_dimensional_vector

        else:
            return self.back_project(low_dimensional_vector)

    def back_project_spatial(self, low_dimensional_vector_spatial):
        coefs = np.dot(
            np.transpose(self.s_pca["eigen_vectors"]), low_dimensional_vector_spatial)
        coefs += self.s_pca["mean_vector"]
        coefs = coefs.reshape((self.s_pca["n_basis"], 1, self.s_pca["n_dim"]))
        robjects.conversion.py2ri = numpy2ri.numpy2ri
        r_data = robjects.Matrix(np.asarray(coefs))
        n_frames = 132
        rcode = '''
            library(fda)
            data = %s
            n_frames = %d
            n_basis = dim(data)[1]
            n_samples = dim(data)[2]
            n_dim = dim(data)[3]
            basisobj = create.bspline.basis(c(0, n_frames - 1), nbasis = n_basis)
            samples_mat = array(0, c(n_samples, n_frames, n_dim))
            for (i in 1:n_samples){
                for (j in 1:n_dim){
                    fd = fd(data[,i,j], basisobj)
                    samples = eval.fd(seq(0, n_frames -1, len = n_frames), fd)
                    samples_mat[i,,j] = samples
                }
            }
        ''' % (r_data.r_repr(), n_frames)
        robjects.r(rcode)
        reconstructed_data = np.asarray(robjects.globalenv['samples_mat'])
        print reconstructed_data.shape
        return reconstructed_data

    def back_project(self, low_dimensional_vector):
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

        spatial_coefs = self._inverse_spatial_pca(
            low_dimensional_vector[:self.s_pca["n_components"]])
        if self.use_time_parameters:
            tmp = low_dimensional_vector[self.s_pca["n_components"]:]
            tmp = [i * 10 for i in tmp]
            time_fd = self._inverse_temporal_pca(tmp)
        else:
            time_fd = None
        return MotionSample(spatial_coefs, self.n_canonical_frames, time_fd)

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
        # reconstruct coefs of the functionial representation
        coefs = np.dot(np.transpose(self.s_pca["eigen_vectors"]), alpha.T)
        coefs += self.s_pca["mean_vector"]
        coefs = coefs.reshape((self.s_pca["n_basis"], 1, self.s_pca["n_dim"]))
        # undo the scaling on the translation
        coefs[:, 0, 0] *= self.translation_maxima[0]
        coefs[:, 0, 1] *= self.translation_maxima[1]
        coefs[:, 0, 2] *= self.translation_maxima[2]
        return coefs

    def _inverse_temporal_pca(self, gamma):
        """ Backtransform a lowdimensional vector gamma to the timewarping
        function t'(t).

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
        # 1.1  reconstruct harmonics and meanfd from coefs
#        robjects.r('library("fda")')
        fd = robjects.r['fd']
        basis = self.t_pca["basis_function"]
#        robjects.conversion.py2ri = numpy2ri.numpy2ri
        eigen_coefs = numpy2ri.numpy2ri(self.t_pca["eigen_vectors"])
#        eigen_coefs = robjects.Matrix(self.t_pca["eigen_vectors"])
        eigenfd = fd(eigen_coefs, basis)
        mean_coefs = numpy2ri.numpy2ri(self.t_pca["mean_vector"])
        meanfd = fd(mean_coefs, basis)
        numframes = self.n_canonical_frames

        # 1.2: reconstruct discrete vector t(t') from gamma by
        # evaluating the harmonics and the mean
        fdeval = robjects.r['eval.fd']
#        t = []
#        t.append(0)
#        for i in xrange(numframes):
#            mean_i = fdeval(i, meanfd)
#            mean_i = np.ravel(np.asarray(mean_i))[-1]
#            eigen_i = np.asarray(fdeval(i, eigenfd))[0]    # its a nested array
#            t.append(t[-1] + np.exp(mean_i + np.dot(eigen_i, gamma)))

        time_frame = np.arange(0, numframes).tolist()
        mean = np.array(fdeval(time_frame, meanfd))
        eigen = np.array(fdeval(time_frame, eigenfd))
        t = [0, ]
        for i in xrange(numframes):
            t.append(t[-1] + np.exp(mean[i] + np.dot(eigen[i], gamma)))

        # 1.3: undo step from timeVarinaces.transform_timefunction during
        # alignment
        t = np.ravel(t[1:])
        t -= 1
        zeroindices = t < 0
        t[zeroindices] = 0
        largest_frame_index = t[-1]
        a = 0.25
        b = 2
        offset = b / a * \
            np.log(1 + a * abs(largest_frame_index - self.n_canonical_frames))
        if largest_frame_index > self.n_canonical_frames:
            new_largest_frame_index = offset + self.n_canonical_frames
        else:
            new_largest_frame_index = self.n_canonical_frames - offset
        scale_factor = new_largest_frame_index / largest_frame_index
        t = [i * scale_factor for i in t]

        # step 2: calculate inverse spline by creating a spline, upsampling it and
        # use the samples to get an inverse spline then sample that inverse spline
        # using step size 1
        # i.e. calculate t'(t) from t(t')

        # 2.1 do the upsampling
        # 2.1.1 create spline from discrete time function
        T = len(t) - 1
        x = np.linspace(0, T, T + 1)
        spline = UnivariateSpline(x, t, s=0, k=2)
        # 2.2.1 sample from spline
        x_sample = np.linspace(0, T, 200)
        w_sample = spline(x_sample)

        # 2.3 try to get a valid inverse spline from upsampled data
        s = 10
        frames = np.linspace(1, t[-1], np.round(t[-1]) - 1)
        while True:
            inverse_spline = UnivariateSpline(w_sample, x_sample, s=s, k=2)
            if not np.isnan(inverse_spline(frames)).any():
                break
            s = s + 1

        # 2.4 sample discrete data from inverse spline
        frames = np.linspace(1, t[-2], np.round(t[-2]))
        # possible bug when frames not goes out of bound
        t = inverse_spline(frames)
        t = np.insert(t, 0, 0)
        t = np.insert(t, len(t), numframes - 1)
        t_max = max(t)
        # temporary solution: smooth temporal parameters, then linearly extend
        # it to (0, t_max)
        sigma = 10
        t = gaussian_filter1d(t, sigma)
        a = min(t)
        b = max(t)
        t = [(i - a) / (b - a) * t_max for i in t]
        t = np.asarray(t)
#        t1 = np.linspace(0,1, len(t))
#        t2 = np.linspace(0, 1, 10*len(t))
#        t_new = np.interp(t2, t1, t)
#        t = gaussian_filter1d(t_new, sigma)
        # handle bug: sometimes samples of the Univariate spline  go out of bounds
        # or are not monotonously increasing
#        over_bound_indices = [i for i in xrange(len(t)) if t[i] > numframes-1]
#        if len(over_bound_indices)>0:
#            last_index = min(over_bound_indices)
#            t =t[:last_index]
#            #t = np.insert(t,last_index,numframes-1)
#            t[last_index-1] = numframes-1
#        else:
#            decreasing_indices = [i-1 for i in xrange(len(t))  if i >0 and t[i]<t[i-1] ]
#            if len(decreasing_indices)>0:
#                last_index = min(decreasing_indices)
#                t=t[:last_index+1]#remove those samples
#            else:
#                last_index = -1
#            t[last_index] = numframes-1

        return t


def plot_data(data, joint_order):
    """Plot data to check the smoothness of data
    plot the quaternion of left shoulder, the index of should is 6
    3 + 5*4 = 23, [24: 28]
    """
    print data.shape
    n_samples, n_frames, n_dims = data.shape
#    n_frames, n_samples, n_dims = data.shape
    fig, axarr = plt.subplots(2, 2)
    for i in xrange(n_samples):
        axarr[0, 0].plot(data[i, :, 3 + 4 * (joint_order - 1) + 0])
        axarr[0, 1].plot(data[i, :, 3 + 4 * (joint_order - 1) + 1])
        axarr[1, 0].plot(data[i, :, 3 + 4 * (joint_order - 1) + 2])
        axarr[1, 1].plot(data[i, :, 3 + 4 * (joint_order - 1) + 3])
    plt.suptitle('LeftShoulder')
    plt.show()


def main():
    """ Function to demonstrate this module """
    mm_file = 'place_second_quaternion_mm.json'

    m = MotionPrimitive(mm_file)
#    sample = m.sample()
#    motion_data = m.back_project_spatial(sample)
#    plot_data(motion_data, 6)
#    skeleton = os.sep.join(('lib', 'skeleton.bvh'))
#    reader = BVHReader(skeleton)
#    BVHWriter(out_file, reader, motion_data[0,:,:], frame_time=0.013889,
#              is_quaternion=True)

    skeleton = os.sep.join(('lib', 'skeleton.bvh'))
    reader = BVHReader(skeleton)
    for i in xrange(1):
        sample = m.sample()
        out_file = 'samples' + os.sep + str(i) + '.bvh'
#        print 'sample: '
#        print sample
#        motion_data = m.back_project_spatial(sample)
#        BVHWriter(out_file, reader, motion_data[0,:,:], frame_time=0.013889,
#                      is_quaternion=True)
    sample.save_motion_vector(out_file)

if __name__ == '__main__':
    main()
