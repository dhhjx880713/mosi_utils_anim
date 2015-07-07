# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 13:51:36 2015

@author: mamauer,erhe01
"""
import numpy as np
#import rpy2.robjects as robjects
#from rpy2.robjects import numpy2ri
import os
from bvh import BVHReader, BVHWriter
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
        # initialize fda library
#        robjects.r('library("fda")')
#
#        # define basis object
#        n_basis = canonical_motion.shape[0]
#        rcode = """
#            n_basis = %d
#            n_frames = %d
#            basisobj = create.bspline.basis(c(0, n_frames - 1),
#                                            nbasis = n_basis)
#        """ % (n_basis, canonical_frames)
#        robjects.r(rcode)
#        self.basis = robjects.globalenv['basisobj']
#
#        # create fd object
#        fd = robjects.r['fd']
#        coefs = numpy2ri.numpy2ri(canonical_motion)
#        self.canonical_motion = fd(coefs, self.basis)

        # save time function
        self.time_function = time_function.tolist()

        # save number of frames in canonical timeline
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

#        eval_fd = robjects.r['eval.fd']
#
#        self.frames = []
#        for t_i in self.time_function:
#            t_i_r = numpy2ri.numpy2ri(np.float(t_i))
#            frame_r = eval_fd(t_i_r, self.canonical_motion)
#            frame = np.array(frame_r)
#            self.frames.append(frame)
#        self.frames_rpy2 = np.array(eval_fd(self.time_function, self.canonical_motion))
        self.frames  = np.array([ si.splev(self.time_function,spline_def) for spline_def in self.canonical_motion_splines]).T
#        print self.frames_rpy2 == self.frames
        # nested array
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
