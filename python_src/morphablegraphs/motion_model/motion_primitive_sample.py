# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 13:51:36 2015

@author: mamauer,erhe01
"""
import numpy as np
from ..animation_data.bvh import BVHReader, BVHWriter
import scipy.interpolate as si

B_SPLINE_DEGREE = 3

class MotionPrimitiveSample(object):
    """ Represent a Sample from a Morphable Model or from a Transition Model

    It provides the following functionality:

    * get_motion_vector(): Returns a vector of frames, representing a \
    discrete version of this sample

    * save_motion(filename): Saves the motion to a bvh file.

    * add_motion(other): Adds the other motion to the end of this motion, \
    i.e. the splines are concatenated and smoothed at the transition

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
    def __init__(self, low_dimensional_parameters, canonical_motion_coefs, time_function, knots):
        self.low_dimensional_parameters = low_dimensional_parameters
        self.time_function = time_function
        self.buffered_frames = None
        canonical_motion_coefs = canonical_motion_coefs.T
        self.n_pose_parameters = len(canonical_motion_coefs)
        #create a b-spline for each pose parameter from the cooeffients
        self.canonical_motion_splines = [(knots, canonical_motion_coefs[i], B_SPLINE_DEGREE) for i in xrange(self.n_pose_parameters)]
        

    def get_motion_vector(self, usebuffer=True):
        """ Return a 2d - vector representing the motion in the new timeline

        Returns
        -------
        * frames: numpy.ndarray
        \tThe new frames as 2d numpy.ndarray with shape (number of frames, \
        number of channels)

        * usebuffer: boolean
        \tWether to return the buffered frame if available
        """
        if usebuffer and self.buffered_frames is not None:
            return self.buffered_frames
        temp_frames = [si.splev(self.time_function,spline_def) for spline_def in self.canonical_motion_splines]
        self.buffered_frames = np.asarray(temp_frames).T
        return self.buffered_frames

    def save_motion_vector(self, filename):
        """ Save the motion vector as bvh file
        (Calls get_motion_vector())

        Parameters
        ----------
        * filename: string
        \tThe path to the target file
        """
        # Save opperation costs much, therefor we can recalculate the vector
        frames = self.get_motion_vector(usebuffer=False)
#        skeleton = os.sep.join(('lib', 'skeleton.bvh'))
        skeleton = ('skeleton.bvh')     
        reader = BVHReader(skeleton)
        BVHWriter(filename, reader, frames, frame_time=0.013889,
                  is_quaternion=True)

    def add_motion(self, other):
        """ Concatenate this motion with another. The other motion will be
        added to the end of this motion

        Parameters
        ----------
        * other: MotionSample
        \tThe other motion as MotionSample object.
        """
        raise NotImplementedError()




def main():
    """ Function to demonstrate this module """
    import json
    testfile = 'MotionSample_test.json'
    with open(testfile) as f:
        data = json.load(f)

    canonical_motion = np.array(data['canonical_motion'])
    canonical_framenumber = data['canonical_frames']
    time_function = np.array(data['time_function'])

    targetfile = 'test.bvh'
    sample = MotionPrimitiveSample(canonical_motion, canonical_framenumber,
                          time_function)

    sample.save_motion_vector(targetfile)


if __name__ == '__main__':
    main()
