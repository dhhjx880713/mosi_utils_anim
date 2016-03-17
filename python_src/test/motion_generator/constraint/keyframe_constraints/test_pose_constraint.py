# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:57:42 2015

@author: erhe01
"""
import sys
import os
ROOTDIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-5]) + os.sep
sys.path.append(ROOTDIR)
import numpy as np
from morphablegraphs.animation_data.bvh import BVHReader
from morphablegraphs.animation_data.skeleton import Skeleton
from morphablegraphs.motion_generator.constraint.keyframe_constraints.pose_constraint import PoseConstraint
from morphablegraphs.motion_generator.constraint.motion_primitive_constraints_builder import MotionPrimitiveConstraintsBuilder
from morphablegraphs.animation_data.motion_editing import convert_euler_frames_to_quaternion_frames

def test_pose_constraint():
    file_path = ROOTDIR+os.sep.join(["..", "test_data", "constrction", "preprocessing", "motion_dtw","walk_001_4_sidestepLeft_139_263.bvh"])
    precision = 1.0
    bvh_reader = BVHReader(file_path)
    skeleton = Skeleton(bvh_reader)
    quat_frames = np.asarray(convert_euler_frames_to_quaternion_frames(bvh_reader, bvh_reader.frames))
    constraint_desc = MotionPrimitiveConstraintsBuilder.create_pose_constraint(skeleton, quat_frames[0])
    pose_constraint = PoseConstraint(skeleton, constraint_desc, precision, weight_factor=1.0)
    error = pose_constraint.evaluate_motion_sample(quat_frames)
    assert error == 0.0
