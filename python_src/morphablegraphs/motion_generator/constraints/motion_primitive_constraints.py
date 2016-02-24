# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:42:13 2015

@author: erhe01
"""
import numpy as np
from ...animation_data.motion_editing import align_quaternion_frames, transform_point
from .spatial_constraints.keyframe_constraints.global_transform_constraint import GlobalTransformConstraint
from .spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION
from .spatial_constraints import MGRDKeyframeConstraint
from mgrd import PoseConstraint as MGRDPoseConstraint
from mgrd import SemanticConstraint as MGRDSemanticConstraint


class MotionPrimitiveConstraints(object):
    """ Represents the input to the generate_motion_primitive_from_constraints
        method of the MotionPrimitiveGenerator class.
    Attributes
     -------
     * constraints : list of dicts
      Each dict contains joint, position,orientation and semanticAnnotation describing a constraint
    """
    def __init__(self):
        self.pose_constraint_set = False
        self.motion_primitive_name = None
        self.settings = None
        self.constraints = []
        self.goal_arc_length = 0           
        self.use_local_optimization = False
        self.step_goal = None
        self.step_start = None
        self.start_pose = None
        self.skeleton = None
        self.precision = {"pos": 1.0, "rot": 1.0, "smooth": 1.0}
        self.verbose = False
        self.min_error = 0.0
        self.best_parameters = None
        self.evaluations = 0
        self.keyframe_event_list = dict()
        self.aligning_transform = None  # used to bring constraints in the local coordinate system of a motion primitive
        self.is_local = False

    def print_status(self):
        print("starting from: ")
        print(self.step_start)
        print("the new goal for " + self.motion_primitive_name)
        print(self.step_goal)
        print("arc length is: " + str(self.goal_arc_length))
#        print  "starting from",last_pos,last_arc_length,"the new goal for", \
#                current_motion_primitive,"is",goal,"at arc length",arc_length

    def evaluate(self, motion_primitive, parameters, prev_frames, use_time_parameters=False):
        """
        Calculates the error of a list of constraints given a sample parameter value.
        
        Returns
        -------
        * sum_error : float
        \tThe sum of the errors for all constraints
    
        """

        motion_spline = motion_primitive.back_project(parameters, use_time_parameters)
        quat_frames = motion_spline.get_motion_vector()
        if not self.is_local:
            #find aligned frames once for all constraints
            quat_frames = align_quaternion_frames(quat_frames, prev_frames, self.start_pose)

        #evaluate constraints with the generated motion
        error_sum = 0
        for constraint in self.constraints:
            error_sum += constraint.weight_factor * constraint.evaluate_motion_sample(quat_frames)
        self.evaluations += 1
        return error_sum

    def get_residual_vector(self, motion_primitive, parameters, prev_frames, use_time_parameters=False):
        """
        Get the residual vector which contains the error values from the motion sample corresponding to each constraint.

        Returns
        -------
        * residual_vector : list
        \tThe list of the errors for all constraints

        """

        motion_spline = motion_primitive.back_project(parameters, use_time_parameters)
        quat_frames = motion_spline.get_motion_vector()
        if not self.is_local:
            #find aligned frames once for all constraints
            quat_frames = align_quaternion_frames(quat_frames, prev_frames, self.start_pose)

        #evaluate constraints with the generated motion
        residual_vector = []
        for constraint in self.constraints:
            vector = constraint.get_residual_vector(quat_frames)
            for value in vector:
                residual_vector.append(value*constraint.weight_factor)
        self.evaluations += 1
        return residual_vector

    def get_length_of_residual_vector(self):
        """ If a trajectory is found it also counts each canonical frame of the graph node as individual constraint
        Returns
        -------
        * residual_vector_length : int
        """
        n_constraints = 0
        for constraint in self.constraints:
            n_constraints += constraint.get_length_of_residual_vector()
        return n_constraints

    def convert_to_mgrd_constraints(self):
        mgrd_constraints = []
        for c in self.constraints:
            if c.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION:
                pose_constraint = MGRDPoseConstraint(c.joint_name, c.weight_factor, c.position, orientation=None)#[None, None, None, None]
                label = "LeftFootContact"# c.semantic_annotation["keyframeLabel"]# TODO add "end" annotation label to all motion primitives
                active = True
                annotations = {label: active}
                semantic_constraint = MGRDSemanticConstraint(annotations, time=None)
                keyframe_constraint = MGRDKeyframeConstraint(pose_constraint, semantic_constraint)
                mgrd_constraints.append(keyframe_constraint)
        return mgrd_constraints

    def transform_constraints_to_local_cos(self):
        print "transform to local cos"
        if self.is_local:
            return self
        else:
            inv_aligning_transform = np.linalg.inv(self.aligning_transform)
            mp_constraints = MotionPrimitiveConstraints()
            mp_constraints.start_pose = {"orientation": [0,0,0], "position": [0,0,0]}
            mp_constraints.is_local = True
            for c in self.constraints:
                if c.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION:
                    position = c.position + [1]
                    if position is not None:
                        orig_position = position
                        indices = [i for i in range(3) if position[i] is None]
                        for i in indices:
                            position[i] = 0
                        #position = [position[0], 0, position[2], 1]
                        print position
                        position = np.dot(inv_aligning_transform, position)[:3].tolist()
                        for i in indices:
                            position[i] = None
                        #position = transform_point(position,  self.aligning_transform["orientation"], self.aligning_transform["translation"])
                        print "transformed constraint",orig_position, position
                        keyframe_constraint_desc = {"joint": c.joint_name,
                                                    "position": position,
                                                    "n_canonical_frames": c.n_canonical_frames,
                                                    "canonical_keyframe":  c.canonical_keyframe,
                                                    "semanticAnnotation": c.semantic_annotation}
                        mp_constraints.constraints.append(GlobalTransformConstraint(self.skeleton, keyframe_constraint_desc, 1.0))
            return mp_constraints

    def convert_to_ik_constraints(self, frame_offset=0, time_function=None):
        ik_constraints = {}
        for c in self.constraints:
            if c.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION and "generated" not in c.semantic_annotation.keys():
                if time_function is not None:
                    keyframe = int(time_function[c.canonical_keyframe])
                else:
                    keyframe = c.canonical_keyframe
                if keyframe not in ik_constraints.keys():
                    ik_constraints[frame_offset+keyframe] = []
                ik_constraint = {"canonical_frame": frame_offset+keyframe, "position": c.position, "joint_name": c.joint_name}
                ik_constraints[frame_offset+keyframe].append(ik_constraint)
        return ik_constraints