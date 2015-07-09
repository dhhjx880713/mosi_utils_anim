# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 13:38:46 2015

@author: mamauer
"""
from motion_sample import MotionSample
from motion_primitive import MotionPrimitive
import os
from bvh import BVHReader, BVHWriter
from motion_editing import concatenate_frames
from kinematic import _convert_frames_to_bvh_frames

SKELETONFILE = os.sep.join(('lib', 'skeleton.bvh'))

class ConcatenatedMotion(object):
    """ A concatenated Motion object

    Parameters
    ----------
    * ms: (optional) MotionSample or list of MotionSamples
        The initial motion sample(s). If None (default) it creates an empty
        Motion

    Attributes
    ----------
    * ms: MotionSample or list of MotionSamples
        The list of current motion samples
    """
    def __init__(self, ms=None):
        if isinstance(ms, MotionSample):
            self.ms = [ms,]
        elif isinstance(ms, list):
            self.ms = ms
        else:
            raise ValueError("Type (%s) of ms is not ok" % type(ms))

    def __add__(self, other):
        """ Return a new ConcatenatedMotion with motion samples from
        both objects

        Parameters
        ----------
        * other: MotionSample or ConcatenatedMotion
            The other motion(list)
        """
        newms = self.ms[:]
        if isinstance(other, ConcatenatedMotion):
            newms += other.ms
        elif isinstance(other, MotionSample):
            newms.append(other)
        return ConcatenatedMotion(newms)

    def add(self, ms):
        """ Add a motion sample to the motion

        Parameters
        ----------
        * ms: MotionSample
            The other motion
        """
        self.ms.append(ms)

    def get_motion_vector(self):
        """ Return a 2d - vector representing the motion in the new timeline

        Returns
        -------
        * frames: numpy.ndarray
        \tThe new frames as 2d numpy.ndarray with shape (number of frames, \
        number of channels)
        """
        self.quat = False
        reader = BVHReader(SKELETONFILE)

        frames = self.ms[0].get_motion_vector()
        frames = _convert_frames_to_bvh_frames(frames)

        for i in xrange(1, len(self.ms)):
            tmp = self.ms[i].get_motion_vector()
            tmp = _convert_frames_to_bvh_frames(tmp)

            frames = concatenate_frames(reader, frames, tmp)
        return frames

#    def get_motion_vector(self):
#        self.quat = True
#        frames = self.ms[0].get_motion_vector()
#
#        for i in xrange(1, len(self.ms)):
#            tmp = self.ms[i].get_motion_vector()
#            diff = frames[-1, :3] - tmp[0, :3]
#
#            tmp[:, :3] += diff
#
#            frames = np.vstack((frames, tmp))
#        return frames

    def save_motion_vector(self, filename):
        """ Save the motion vector as bvh file
        (Calls get_motion_vector())

        Parameters
        ----------
        * filename: string
        \tThe path to the target file
        """
        frames = self.get_motion_vector()
        reader = BVHReader(SKELETONFILE)
        BVHWriter(filename, reader, frames, frame_time=0.013889,
                  is_quaternion=self.quat)


def main():
    input_path_prefix = '../input_data'
    mmfile_left = os.sep.join((input_path_prefix,
                               'walk_leftStance_quaternion_mm.json'))
    mmfile_right = os.sep.join((input_path_prefix,
                                'walk_rightStance_quaternion_mm.json'))

    mmleft = MotionPrimitive(mmfile_left)
    mmright = MotionPrimitive(mmfile_right)

    ms = mmleft.sample()
    ms2 = mmright.sample()
    cm1 = ConcatenatedMotion(ms)
    cm2 = cm1 + ms2

    cm2.save_motion_vector("test.bvh")

if __name__ == '__main__':
    main()
