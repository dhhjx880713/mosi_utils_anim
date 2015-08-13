# -*- coding: utf-8 -*-

"""

======

Author: Han

"""

import os
import sys
ROOT_DIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3]) + os.sep
sys.path.append(ROOT_DIR)
TEST_LIB_PATH = ROOT_DIR + 'test'
TEST_DATA_PATH = ROOT_DIR +  '../test_data/animation_data'
sys.path.append(TEST_LIB_PATH)
from animation_data.bvh import BVHReader
from animation_data.quaternion_frame import QuaternionFrame
from libtest import params, pytest_generate_tests

class TestQuaternionFrame(object):
    """QuaternionFrame test class"""

    def setup_class(self):

        testfile = TEST_DATA_PATH + os.sep + "walk_001_1_rightStance_86_128.bvh"
        self.bvh_reader = BVHReader(testfile)

    param_test_quaternion_representation = [{"node_name": "LeftHand",
                                     "frame_number": 0,
                                     "expected": (-0.657731, 0.74964, 0.0576654, 0.0458828)
                                     },
                                    {"node_name": "RightHand",
                                     "frame_number": 0,
                                     "expected": (0.51357, 0.841636, 0.13796, 0.0941383)},
                                    {"node_name": "Hips",
                                     "frame_number": 0,
                                     "expected": (0.00110963, 0.709016, 0.705188, 0.00233869)},
                                    {"node_name": "Spine",
                                     "frame_number": 0,
                                     "expected": (0.999661, 0.0170712, 0.018936, 0.00520762)}
                                    ]

    @params(param_test_quaternion_representation)
    def test_quaternion_representation(self, node_name, frame_number, expected):
        """Unit test for the joint orientations stored in QuaternionFrame"""
        frame = QuaternionFrame(self.bvh_reader, 
                                self.bvh_reader.frames[frame_number],
                                True)
        for i in xrange(len(expected)):
            assert round(frame[node_name][i], 3) == round(expected[i], 3)
            
    param_test_number_of_joints = [{"frame_number": 0,
                                     "expected": 19
                                     }
                                    ]
    @params(param_test_number_of_joints)
    def test_number_of_joints(self, frame_number, expected):
        """Unit test for the number of quaternions stored in QuaternionFrame."""
        frame = QuaternionFrame(self.bvh_reader, 
                                self.bvh_reader.frames[frame_number],
                                False)
        assert len(frame) == expected