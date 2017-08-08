"""
test_bvh_writer.py
Unit test for bvh.py
======

Author: Han

"""

import os
import sys
ROOTDIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3]) + os.sep
sys.path.append(ROOTDIR)
from morphablegraphs.animation_data.bvh import BVHReader, BVHWriter
from morphablegraphs.animation_data.motion_editing import get_cartesian_coordinates_from_euler_full_skeleton
from morphablegraphs.animation_data.skeleton import Skeleton
from libtest import params, pytest_generate_tests
ROOT_DIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3]) + os.sep
TEST_DATA_PATH = ROOT_DIR + '../test_data/animation_data'
TEST_RESULT_PATH = ROOT_DIR + '../test_output/animation_data'


class TestBVHWriter(object):
    
    def setup_class(self):
        testfile = TEST_DATA_PATH + os.sep + "walk_001_1_rightStance_86_128.bvh"
        self.bvh_reader = BVHReader(testfile)
        self.skeleton = Skeleton(self.bvh_reader)
        self.test_node_name = 'LeftHand'
        self.target_point = get_cartesian_coordinates_from_euler_full_skeleton(self.bvh_reader,
                                                                               self.skeleton,
                                                                               self.test_node_name,
                                                                               self.bvh_reader.frames[0])
        self.bvhwriter = BVHWriter(TEST_RESULT_PATH + os.sep + "walk_001_1_rightStance_86_128.bvh",
                                   self.skeleton, 
                                   self.bvh_reader.frames,
                                   self.bvh_reader.frame_time)
                                                                       
    param_quaternion_to_euler = [{'quat': [0.0011096303018118473, 
                                            0.70901567615554151, 
                                            0.70518796800437256, 
                                            0.0023386894102803268],
                                   'res': [-179.901166951,
                                           0.279680762937,
                                           -89.6900864729]}]  
    
    @params(param_quaternion_to_euler)
    def test_quaternion_to_euler(self, quat, res):
        euler_angles = self.bvhwriter._quaternion_to_euler(quat)
        for i in range(len(euler_angles)):
            assert round(euler_angles[i], 3) == round(res[i], 3)

    def test_write(self):
        bvhreader = BVHReader(TEST_RESULT_PATH + os.sep + "walk_001_1_rightStance_86_128.bvh")
        joint_position = get_cartesian_coordinates_from_euler_full_skeleton(self.bvh_reader,
                                                                            self.skeleton,
                                                                            self.test_node_name,
                                                                            bvhreader.frames[0])
        for i in range(len(joint_position)):
            assert round(self.target_point[i], 3) == round(joint_position[i], 3)    
            assert round(self.target_point[i], 3) == round(joint_position[i], 3)    