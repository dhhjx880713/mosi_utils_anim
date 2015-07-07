# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 14:05:34 2015


@author: herrmann

All unit tests for the MotionPrimitive evaluation functions 
"""


import os
import sys
from libtest import params, pytest_generate_tests

TESTPATH = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]) + os.sep
sys.path.insert(1, TESTPATH)
sys.path.insert(1, TESTPATH + (os.sep + os.pardir))

from bvh import BVHReader
from evaluate_motion_primitive import calculate_cartesian_pose_bounding_box, \
                                        calculate_parameter_bounding_box, \
                                        calculate_avg_motion_velocity_from_bvh

class TestMotionPrimitiveEvaluation(object):
    """test class for MotionPrimitive evaluation functions """

    def setup_method(self, method):
        testfile =  'skeleton.bvh'
        self.bvh_reader = BVHReader(testfile)
        self.eps = 0.00001

    def _isequal(self, a, b):
         return abs(a) - abs(b) < self.eps 
         
    def test_calculate_cartesian_pose_bounding_box(self):
        """ Test if the cartesian pose bounding box is calculated correctly 
            for one motion
        """
        c_bb = calculate_cartesian_pose_bounding_box(self.bvh_reader)
        assert self._isequal(c_bb['X']['max'], 35.748958515963047)
        assert self._isequal(c_bb['X']['min'], 37.660904382904491)
        assert self._isequal(c_bb['Y']['max'], 149.55068984778396)
        assert self._isequal(c_bb['Y']['min'], 2.4597640598058774)
        assert self._isequal(c_bb['Z']['max'], 28.84273180935914)
        assert self._isequal(c_bb['Z']['min'], 83.01746453040171)

        
        

    def test_calculate_parameter_bounding_box(self):
        """ Test if the parameter bounding box is calculated correctly for one
            motion
        """
        pose_bb = calculate_parameter_bounding_box(self.bvh_reader)
        assert self._isequal(pose_bb['Hips']['Xposition']['max'], 2.02552772072)
        assert self._isequal(pose_bb['Hips']['Xposition']['min'], 1.66148452799)
        assert self._isequal(pose_bb['LeftHand']['Yrotation']['max'], 18.921507)
        assert self._isequal(pose_bb['LeftHand']['Yrotation']['min'], 5.537385)
        assert self._isequal(pose_bb['RightFoot']['Xrotation']['max'], 3.068707)
        assert self._isequal(pose_bb['RightFoot']['Xrotation']['min'], 8.188002)


      
        
    def test_calculate_avg_motion_velocity_from_bvh(self):
        """ Test if the average velocity is calculated correctly for one motion
        """
        avg_v = calculate_avg_motion_velocity_from_bvh(self.bvh_reader)
        assert self._isequal(avg_v['Hips']['Xposition']['var'], 0.0068377127280636911)
        assert self._isequal(avg_v['Hips']['Xposition']['avg'], 0.093349612057607917)
        assert self._isequal(avg_v['RightLeg']['Xrotation']['var'], 2.2826086953228587e-06)
        assert self._isequal(avg_v['RightFoot']['Xrotation']['avg'], 0.96339993478260533 )
        assert self._isequal(avg_v['RightFoot']['Xrotation']['max'], 5.4390439999999671 )
        assert self._isequal(avg_v['LeftHand']['Yrotation']['var'], 0.058402190667105204)
        assert self._isequal(avg_v['LeftHand']['Yrotation']['avg'], 0.31424626086956292)

   
