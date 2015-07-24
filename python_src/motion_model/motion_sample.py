# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 13:51:36 2015

@author: mamauer,erhe01
"""
import numpy as np
from animation_data.bvh import BVHReader, BVHWriter
import scipy.interpolate as si

class MotionSample(object):
    """ Represent a Sample from a Morphable Model or from a Transition Model

    It provides the following functionality:

    * get_motion_vector(): Returns a vector of frames, representing a \
    discrete version of this sample

    * save_motion(filename): Saves the motion to a bvh file.

    * add_motion(other): Adds the other motion to the end of this motion, \
    i.e. the splines are concatenated and smoothed at the transition

    Parameters
    ----------
    * canonical_motion: numpy.ndarray
    \tA tuple with a Numpy array holding the coeficients of the fd object

    * canonical_framenumber: int
    \tThe number of frames in the canonical timeline

    * time_function: numpy.ndarray
    \tThe indices of the timewarping function t'(t)

    Attributes
    ----------
    * canonical_motion: rpy2.robjects.vectors.ListVector
    \tThe functional data (fd) object from the R-Library "fda" \
    representing this motion in the canonical timeline

    * time_function: tuple with (scipy.UnivariateSpline, int)
    \tThe timefunction t'(t) that warps the motion from the canonical timeline \
    into a new timeline. The first value of the tuple is the spline, the second\
    value is the new number of frames n'

    * canonical_frames: int
    \tThe number of frames in the canonical timeline

    * new_framenumber: int
    \tThe number of frames in the new timeline

    * t: numpy.ndarray
    \tThe times where to evaluate the motion in the new timeline
    """
    def __init__(self, canonical_motion, canonical_frames, time_function, knots):

        self.time_function = time_function
        self.canonical_frames = canonical_frames
        self.frames = None
        n_dim = len(canonical_motion[0][0])
        canonical_motion_coefs = canonical_motion.T
        self.canonical_motion_splines = [(knots,canonical_motion_coefs[i],3) for i in xrange(n_dim)]
        

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
        if usebuffer and self.frames is not None:
            return self.frames
            
        temp_frames = [ si.splev(self.time_function,spline_def) for spline_def in self.canonical_motion_splines]
        self.frames  = np.asarray(temp_frames).T

        # Change the result from a 3D array into a 2D array. example: change 47x1x79 to 47x79
        self.frames = np.reshape(self.frames, (self.frames.shape[0],
                                               self.frames.shape[-1]))

        return self.frames

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
    sample = MotionSample(canonical_motion, canonical_framenumber,
                          time_function)

    sample.save_motion_vector(targetfile)


if __name__ == '__main__':
    main()
