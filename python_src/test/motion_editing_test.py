# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:56:09 2015

@author: hadu01
"""

import os
import sys
import numpy as np
from math import sqrt
from libtest import params, pytest_generate_tests
sys.path.append("..")
from utilities.motion_editing import convert_quaternion_to_euler


param_convert_quaternion_to_euler = [
    {"quat": [[1.0,2.0,3.0,0.0,0.0,0.0,0.1], [1.0,2.0,4.0,0.0,0.0,0.0,0.2]],
     "res": [[1.0,2.0,3.0,0.0,0.0,180.0], [1.0,2.0,4.0,0.0,0.0,180.0]]},
    {"quat": [[0.0,2.0,3.0,0.0,0.1,0.2,0.1], [1.0,2.0,4.0,0.0,0.0,0.0,0.2]],
     "res": [[0.0,2.0,3.0,-135.0,19.471220634490695,-135.0],
             [1.0,2.0,4.0,0.0,0.0,180.0]]},
]

@params(param_convert_quaternion_to_euler)
def test_convert_quaternion_to_euler(quat, res):

    for r, e in zip(res, convert_quaternion_to_euler(quat)):
        for rr, ee in zip(r, e):
            a = rr**2
            b = ee**2
            assert np.isclose(a-b, 0.0)


#from motion_editing import *
#from bvh import BVHReader
#from helper_functions import get_motion_primitive_directory
#from motion_primitive import MotionPrimitive
#from morphable_graph import create_filtered_node_name_map

#param_get_cartesian_coordinates = [
#    {"test_file": "pick_002_1_first_540_610.bvh",
#     "frame_index": 0,
#     "joint": "RightHand",
#     "res": [35.81118176, 91.40765184, -5.18409637]},
#    {"test_file": "pick_008_3_first_634_759.bvh",
#     "frame_index": 2,
#     "joint": "LeftHand",
#     "res": [-31.32512642, 86.76008481, -12.4771128]}
#]
#
#@params(param_get_cartesian_coordinates)
#def test_get_cartesian_coordinates(test_file, frame_index, joint, res):
#    """Unit test for get_cartesian_coordinates
#    """
#    reader = BVHReader(test_file)
#    euler_frame = reader.keyframes[frame_index]
#    pos = get_cartesian_coordinates(reader, joint, euler_frame)
#    for i in xrange(len(res)):
#        assert round(pos[i], 3) == round(res[i], 3)
#
#
#param_shift_euler_frames_to_ground = [
#    {"elementary_action": "pick",
#     "primitive_type": "first",
#     "contact_joint": "LeftFoot"}
#]
#
#@params(param_shift_euler_frames_to_ground)
#def test_shift_euler_frames_to_ground(elementary_action,
#                                      primitive_type,
#                                      contact_joint):
#    """Unit test for shift_euler_frames_to_ground
#    """
#    input_dir = get_motion_primitive_directory(elementary_action)
#    mm_filename = '_'.join([elementary_action,
#                            primitive_type,
#                            'quaternion',
#                            'mm.json'])
#    bvh_reader = BVHReader('skeleton.bvh')
#    node_name_map = create_filtered_node_name_map(bvh_reader)
#    mm = MotionPrimitive(os.sep.join([input_dir, mm_filename]))
#    motion_sample = mm.sample()
#    motion_sample.get_motion_vector()
#    euler_frames = convert_quaternion_to_euler(motion_sample.frames.tolist())
#    new_euler_frames = shift_euler_frames_to_ground(euler_frames,
#                                                    contact_joint,
#                                                    bvh_reader,
#                                                    node_name_map)
#    for frame in new_euler_frames:
#        # get contact point position
#        position = get_cartesian_coordinates2(bvh_reader,
#                                              contact_joint,
#                                              frame,
#                                              node_name_map)
#        assert round(position[1], 3) == 0