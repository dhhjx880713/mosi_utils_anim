# -*- coding: utf-8 -*-

"""
test_cartesian_frame.py
Unit test for cartesian_frame.py
======

Author: Erik

"""

import os
import sys
from libtest import params, pytest_generate_tests

TESTPATH = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]) + os.sep
sys.path.insert(1, TESTPATH)
sys.path.insert(1, TESTPATH + (os.sep + os.pardir))
from bvh import BVHReader
from cartesian_frame import CartesianFrame


class TestCartesianFrame(object):
    """CartesianFrame test class"""

    def setup_method(self, method):

        testfile = TESTPATH + "/walk_001_1_rightStance_86_128.bvh"
        self.bvh_reader = BVHReader(testfile)

    param_test_foward_kinematics = [{"node_name": "LeftHand",
                                     "frame_number": 0,
                                     "expected": [-23.6024, 78.1499, -2.39092]
                                     },
                                    {"node_name": "RightHand",
                                     "frame_number": 0,
                                     "expected": [25.9699, 80.7445, -2.84261]},
                                    {"node_name": "Hips",
                                     "frame_number": 0,
                                     "expected": [-1.1578, 90, 4.68537]},
                                    {"node_name": "Spine",
                                     "frame_number": 0,
                                     "expected": [-1.07501, 105.317, 4.72439]},
                                     {"node_name": "Hips",
                                     "frame_number": 0,
                                     "expected": [-1.15780000332, 90.0, 4.68537284741]}
                                    ]

    @params(param_test_foward_kinematics)
    def test_foward_kinematics(self, node_name, frame_number, expected):
        """Unit test for the joint positions stored in CartesianFrame"""

        frame = CartesianFrame(self.bvh_reader, frame_number)

        for i in xrange(len(expected)):
            assert round(frame[node_name][i], 3) == round(expected[i], 3)

    param_test_number_of_joints = [
        {"frame_number": 0, "expected": 19},
    ]

    @params(param_test_number_of_joints)
    def test_number_of_joints(self, frame_number, expected):
        """Unit test for the number of coordinates stored in CartesianFrame."""

        frame = CartesianFrame(self.bvh_reader, frame_number)

        assert len(frame) == expected
