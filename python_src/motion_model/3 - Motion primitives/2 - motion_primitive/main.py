# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 08:39:01 2015

@author: mamauer

This module clues the MotionPrimitive and the MotionSample module to
test both modules.
"""
from motion_primitive import MotionPrimitive
from motion_sample import MotionSample


def main():
    testfile = "walk_leftStance_quaternion_mm.json"
    savefile = "test.bvh"
    mp = MotionPrimitive(testfile)

    sample = mp.sample()
    sample.save_motion_vector(savefile)

if __name__=='__main__':
    main()