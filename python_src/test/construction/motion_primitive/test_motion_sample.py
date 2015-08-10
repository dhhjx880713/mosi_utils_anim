# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 11:42:34 2015

@author: mamauer

All Unittests for the MotionSample class
"""
import os
import sys
ROOTDIR = os.sep.join(['..'] * 3)
sys.path.append(ROOTDIR + os.sep)
TESTPATH = ROOTDIR + os.sep + r'construction/motion_primitive'
TESTLIBPATH = ROOTDIR + os.sep + 'test/'
sys.path.append(TESTPATH)
sys.path.append(TESTLIBPATH)
sys.path.append(ROOTDIR + os.sep + 'construction')
import rpy2.robjects as robjects
import numpy as np
import pytest
import json
TEST_DATA_PATH = ROOTDIR + os.sep + r'../test_data/constrction/motion_primitive'
# from libtest import params, pytest_generate_tests
from motion_sample import MotionSample

@pytest.fixture(scope="module")
def motionSample():
    testfile = TEST_DATA_PATH + os.sep + 'MotionSample_test.json'
    with open(testfile) as f:
        data = json.load(f)

    robjects.r('library("fda")')
    canonical_motion = np.array(data['canonical_motion'])
    canonical_framenumber = data['canonical_frames']
    time_function = np.array(data['time_function'])
    return MotionSample(canonical_motion, canonical_framenumber, time_function)


class Test__init__(object):
    """ Test if the MotionSample class is initialized correctly """

    def test_basis_init(self, motionSample):
        """ Test if the basis object is a fda basis """
        assert robjects.r['is.basis'](motionSample.basis)

    def test_canonical_motion_init(self, motionSample):
        """ Test if the canonical_motion object is a fda fd """
        assert robjects.r['is.fd'](motionSample.canonical_motion)


class Test_get_motion_vector(object):
    """ Test if the MotionSample.get_motion_vector returns a framevector """

    def test_length_equal_to_time_vector(self, motionSample):
        """
        Test if the number of frames is equal to the number of time indices
        """
        frames = motionSample.get_motion_vector()
        assert frames.shape[0] == len(motionSample.time_function)

    def test_buffer(self, motionSample):
        """ Test if the motion can be buffered and recalculated """
        frames = motionSample.get_motion_vector(usebuffer=False)

        motionSample.time_function[1] -= 0.1

        bufferedframes = motionSample.get_motion_vector(usebuffer=True)
        changedframes = motionSample.get_motion_vector(usebuffer=False)

        assert np.alltrue(frames == bufferedframes)
        assert np.any(frames != changedframes)


