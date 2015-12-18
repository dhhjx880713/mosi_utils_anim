# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 13:51:36 2015

@author: mamauer,erhe01
"""
import numpy as np
from ..animation_data.bvh import BVHReader, BVHWriter
import scipy.interpolate as si
from . import B_SPLINE_DEGREE


class MotionSpline(object):
    """ Represent a Sample from a MotionPrimitive
    * get_motion_vector(): Returns a vector of frames, representing a \
    discrete version of this sample

    Parameters
    ----------
    *low_dimensional_parameters: np.ndarray
    \tparamters used to backproject the sample
    * canonical_motion_coefs: numpy.ndarray
    \tA tuple with a Numpy array holding the coefficients of the multidimensional spline
    * time_function: numpy.ndarray
    \tThe indices of the timewarping function t'(t)
    * knots: numpy.ndarray
    \tThe knots for the coefficients of the multidimensional spline definition
    Attributes
    ----------
    *low_dimensional_parameters: np.ndarray
    \tparamters used to backproject the sample
    * canonical_motion_splines: list
    \tA list of spline definitions for each pose parameter to represent the multidimensional spline.
    * time_function: no.ndarray
    \tThe timefunction t'(t) that warps the motion from the canonical timeline \
    into a new timeline. The first value of the tuple is the spline, the second\
    value is the new number of frames n'

    """
    def __init__(self, low_dimensional_parameters, canonical_motion_coefs, time_function, knots, semantic_annotation=None):
        self.low_dimensional_parameters = low_dimensional_parameters
        self.time_function = time_function
        self.buffered_frames = None
        canonical_motion_coefs = canonical_motion_coefs.T
        self.n_pose_parameters = len(canonical_motion_coefs)
        #create a b-spline for each pose parameter from the cooeffients
        self.canonical_motion_splines = [(knots, canonical_motion_coefs[i], B_SPLINE_DEGREE) for i in xrange(self.n_pose_parameters)]
        self.semantic_annotation = semantic_annotation

    def get_motion_vector(self):
        """ Return a 2d - vector representing the motion in the new timeline

        Returns
        -------
        * frames: numpy.ndarray
        \tThe new frames as 2d numpy.ndarray with shape (number of frames, \
        number of channels)
        """
        temp_frames = [si.splev(self.time_function, spline_def) for spline_def in self.canonical_motion_splines]
        self.buffered_frames = np.asarray(temp_frames).T
        return self.buffered_frames

    def get_buffered_motion_vector(self):
        """ Returns a buffered version  of the motion vector from the last call to get_motion_vector

        Returns
        -------
        * frames: numpy.ndarray
        \tThe new frames as 2d numpy.ndarray with shape (number of frames, \
        number of channels)
        """
        if self.buffered_frames is not None:
            return self.buffered_frames
        else:
            return self.get_motion_vector()
