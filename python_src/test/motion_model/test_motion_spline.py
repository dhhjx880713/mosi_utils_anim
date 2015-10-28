# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 11:42:34 2015

@author: mamauer

All Unittests for the MotionSample class
"""
import os
import sys
ROOTDIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3]) + os.sep
sys.path.append(ROOTDIR)
TESTLIBPATH = ROOTDIR + os.sep + 'test/'
sys.path.append(TESTLIBPATH)
sys.path.append(ROOTDIR + os.sep + 'construction')
import rpy2.robjects as robjects
import numpy as np
import pytest
import json
TEST_DATA_PATH = ROOTDIR + os.sep + r'../test_data/motion_model'
# from libtest import params, pytest_generate_tests
from morphablegraphs.motion_model.motion_spline import MotionSpline

@pytest.fixture(scope="module")
def motionSpline():
    testfile = TEST_DATA_PATH + os.sep + 'Motion_sample_test.json'
    with open(testfile) as f:
        data = json.load(f)

    robjects.r('library("fda")')
    canonical_motion = np.array(data['canonical_motion'])
    time_function = np.array(data['time_function'])
    knots = np.array(data['knots'])
    return MotionSpline(None, canonical_motion, time_function, knots)


class Test__init__(object):
    """ Test if the MotionSample class is initialized correctly """

    def test_canonical_motion_init(self, motionSpline):
        """ Test if the canonical_motion object has the correct shape"""
        assert len(motionSpline.canonical_motion_splines) == motionSpline.n_pose_parameters



class Test_get_motion_vector(object):
    """ Test if the MotionSample.get_motion_vector returns a framevector """

    def test_length_equal_to_time_vector(self, motionSpline):
        """
        Test if the number of frames is equal to the number of time indices
        """
        frames = motionSpline.get_motion_vector()
        assert frames.shape[0] == len(motionSpline.time_function)

    def test_buffer(self, motionSpline):
        """ Test if the motion can be buffered and recalculated """
        frames = motionSpline.get_motion_vector()

        motionSpline.time_function[1] -= 0.1

        bufferedframes = motionSpline.get_buffered_motion_vector()
        changedframes = motionSpline.get_motion_vector()

        assert np.alltrue(frames == bufferedframes)
        assert np.any(frames != changedframes)

