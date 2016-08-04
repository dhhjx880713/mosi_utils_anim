# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:42:13 2015

@author: erhe01
"""
import numpy as np
from copy import copy
from ...animation_data.motion_editing import align_quaternion_frames, transform_point, quaternion_from_vector_to_vector, euler_to_quaternion, quaternion_multiply
from .spatial_constraints.keyframe_constraints.global_transform_constraint import GlobalTransformConstraint
from .spatial_constraints.keyframe_constraints.two_hand_constraint import TwoHandConstraintSet
from .spatial_constraints.keyframe_constraints.pose_constraint import PoseConstraint
from .spatial_constraints.keyframe_constraints.direction_2d_constraint import Direction2DConstraint
from .spatial_constraints.keyframe_constraints.look_at_constraint import LookAtConstraint
from .spatial_constraints.keyframe_constraints.global_transform_ca_constraint import GlobalTransformCAConstraint
from .spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION, SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION, SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE,SPATIAL_CONSTRAINT_TYPE_KEYFRAME_DIR_2D, SPATIAL_CONSTRAINT_TYPE_KEYFRAME_LOOK_AT, SPATIAL_CONSTRAINT_TYPE_CA_CONSTRAINT
from ik_constraints import JointIKConstraint, TwoJointIKConstraint
from ik_constraints_builder import IKConstraintsBuilder
from ...utilities.log import write_log
try:
    from mgrd import CartesianConstraint as MGRDCartesianConstraint
    from mgrd import PoseConstraint as MGRDPoseConstraint
    from mgrd import SemanticConstraint as MGRDSemanticConstraint
    from mgrd import SemanticPoseConstraint as MGRDSemanticPoseConstraint
except ImportError:
    pass


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
        self.action_name = None
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
        self.is_last_step = False
        self.time = 0.0

    def print_status(self):
        write_log("starting from:", self.step_start)
        write_log("the new goal for " + self.motion_primitive_name, "is", self.step_goal)
        write_log("arc length is: " + str(self.goal_arc_length))

    def evaluate(self, motion_primitive, parameters, prev_frames, use_time_parameters=False):
        """
        Calculates the error of a list of constraints given a sample parameter value.
        
        Returns
        -------
        * sum_error : float
        \tThe sum of the errors for all constraints
    
        """
        motion_spline = motion_primitive.back_project(parameters, use_time_parameters)
        if not self.is_local:
            #find aligned frames once for all constraints#
            motion_spline.coeffs = align_quaternion_frames(motion_spline.coeffs, prev_frames, self.start_pose)

        #evaluate constraints with the generated motion
        error_sum = 0
        for constraint in self.constraints:
            error_sum += constraint.weight_factor * constraint.evaluate_motion_spline(motion_spline)
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
        if not self.is_local:
            #find aligned frames once for all constraints
            motion_spline.coeffs = align_quaternion_frames(motion_spline.coeffs, prev_frames, self.start_pose)
        #evaluate constraints with the generated motion
        residual_vector = []
        for constraint in self.constraints:
            vector = constraint.get_residual_vector_spline(motion_spline)
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

    def convert_to_mgrd_constraints(self, use_semantic_annotation=False):
        """ map constraints to mgrd constraints and merges 2d direction constraints with position constraints
        Returns
        -------
        * semantic_pose_constraints: list
            A list of mgrd semantic pose constraints with semantic annotation
        * cartesian_constraints: list
            A list of mgrd cartesian constraints without semantic annotation
        """
        semantic_pose_constraints = []
        cartesian_constraints = []
        temp_constraint_list = dict()
        UNLABELED_KEY = "unlabeled"
        temp_constraint_list[UNLABELED_KEY] = []
        for c in self.constraints:
            if c.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_DIR_2D:
                original_orientation = self.skeleton.get_root_reference_orientation()
                ref_vector = [0,0,-1]
                # convert angle between c.direction_constraint and reference vector into a quaternion and add reference orientation
                delta_orientation = quaternion_from_vector_to_vector(c.direction_constraint, ref_vector)
                orientation = quaternion_multiply(original_orientation, delta_orientation)
                desc = {"type": "dir", "value":orientation, "joint": "Hips"}
                if "keyframeLabel" in c.semantic_annotation and use_semantic_annotation:
                    semantic_label = c.semantic_annotation["keyframeLabel"]
                    if semantic_label not in temp_constraint_list.keys():
                        temp_constraint_list[semantic_label] = []
                    temp_constraint_list[semantic_label].append(desc)
                else:
                    temp_constraint_list[UNLABELED_KEY].append(desc)

            elif c.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION:
                if c.position[1] is None:
                    y_coordinate = 80
                else:
                    y_coordinate = c.position[1]
                desc = {"type":"pos","value":[c.position[0], y_coordinate, c.position[2]], "joint": c.joint_name, "weight_factor":c.weight_factor}
                if "keyframeLabel" in c.semantic_annotation and use_semantic_annotation:
                    semantic_label = c.semantic_annotation["keyframeLabel"]
                    if semantic_label not in temp_constraint_list.keys():
                        temp_constraint_list[semantic_label] = []
                    temp_constraint_list[semantic_label].append(desc)
                else:
                    temp_constraint_list[UNLABELED_KEY].append(desc)
            elif c.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE:
                semantic_label = UNLABELED_KEY#c.semantic_annotation["keyframeLabel"]#
                if semantic_label not in temp_constraint_list.keys():
                    temp_constraint_list[semantic_label] = []
                joints = self.skeleton.node_name_frame_map.keys()
                points = c.pose_constraint
                for j, p in zip(joints,points):
                    desc = {"type":"pos","value":p, "joint": j, "weight_factor":c.weight_factor}
                    temp_constraint_list[semantic_label].append(desc)
            elif c.constraint_type == SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION:
                c_desc_list = []
                for joint_name, position in zip(c.joint_names, c.positions):
                    desc = {"type":"pos","value":[position[0], position[1], position[2]], "joint": joint_name, "weight_factor":c.weight_factor}
                    c_desc_list.append(desc)
                if "keyframeLabel" in c.semantic_annotation and use_semantic_annotation:
                    semantic_label = c.semantic_annotation["keyframeLabel"]
                    if semantic_label not in temp_constraint_list.keys():
                        temp_constraint_list[semantic_label] = []
                    temp_constraint_list[semantic_label] += c_desc_list
                else:
                    temp_constraint_list[UNLABELED_KEY] += c_desc_list
            elif c.constraint_type == SPATIAL_CONSTRAINT_TYPE_CA_CONSTRAINT:
                desc = {"type":"pos","value":[c.position[0], c.position[1], c.position[2]], "joint": c.joint_name, "weight_factor":c.weight_factor}
                temp_constraint_list[UNLABELED_KEY].append(desc)

        for key in temp_constraint_list.keys():
            if key == UNLABELED_KEY:
                for temp_c in temp_constraint_list[key]:
                    if temp_c["type"] == "pos":
                        cartesian_constraint = MGRDCartesianConstraint(temp_c["value"], temp_c["joint"], temp_c["weight_factor"])
                        cartesian_constraints.append(cartesian_constraint)
            elif key =="start":
                for temp_c in temp_constraint_list[key]:
                    if temp_c["type"] == "pos":
                        pose_constraint = MGRDPoseConstraint(temp_c["joint"], temp_c["weight_factor"], temp_c["value"], [1,0,0,0])
                        semantic_constraint = MGRDSemanticConstraint({key: True}, time=None)
                        semantic_pose_constraint = MGRDSemanticPoseConstraint(pose_constraint, semantic_constraint)
                        semantic_pose_constraint.weights = (1.0,0.0)
                        semantic_pose_constraints.append(semantic_pose_constraint)

            else:
                print "merge constraints",key
                orientation = [1,0,0,0]
                position = None
                joint_name = None
                weight_factor = 1.0
                has_orientation = False
                for temp_c in temp_constraint_list[key]:
                    if temp_c["type"] == "pos":
                        position = temp_c["value"]
                        joint_name = temp_c["joint"]
                    elif temp_c["type"] == "dir":
                        orientation = temp_c["value"]
                        has_orientation = True
                if position is not None:
                    pose_constraint = MGRDPoseConstraint(joint_name, weight_factor, position, orientation)
                    semantic_constraint = MGRDSemanticConstraint({key: True}, time=None)
                    semantic_pose_constraint = MGRDSemanticPoseConstraint(pose_constraint, semantic_constraint)
                    if has_orientation:
                        weights = (1.0,1.0)
                    else:
                        weights = (1.0,0.0)
                    semantic_pose_constraint.weights = weights
                    semantic_pose_constraints.append(semantic_pose_constraint)
        return semantic_pose_constraints, cartesian_constraints

    def transform_constraints_to_local_cos(self):
        if self.is_local or self.aligning_transform is None:
            return self
        write_log("transform to local coordinate system")
        inv_aligning_transform = np.linalg.inv(self.aligning_transform)
        mp_constraints = MotionPrimitiveConstraints()
        mp_constraints.start_pose = {"orientation": [0,0,0], "position": [0,0,0]}
        mp_constraints.skeleton = self.skeleton
        mp_constraints.is_local = True
        mp_constraints.use_local_optimization = self.use_local_optimization
        for c in self.constraints:
            if c.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION:
                if c.position is not None:
                    position = [c.position[0], c.position[1], c.position[2], 1]
                    indices = [i for i in range(3) if position[i] is None]
                    for i in indices:
                        position[i] = 0
                    position = np.dot(inv_aligning_transform, position)[:3].tolist()
                    for i in indices:
                        position[i] = None
                    keyframe_constraint_desc = {"joint": c.joint_name,
                                                "position": position,
                                                "n_canonical_frames": c.n_canonical_frames,
                                                "canonical_keyframe":  c.canonical_keyframe,
                                                "semanticAnnotation": c.semantic_annotation}
                    mp_constraints.constraints.append(GlobalTransformConstraint(self.skeleton, keyframe_constraint_desc, 1.0))

            elif c.constraint_type == SPATIAL_CONSTRAINT_TYPE_CA_CONSTRAINT:
                position = [c.position[0], c.position[1], c.position[2], 1]
                position = np.dot(inv_aligning_transform, position)[:3].tolist()
                keyframe_constraint_desc = {"joint": c.joint_name,
                                            "position": position,
                                            "n_canonical_frames": c.n_canonical_frames,
                                            "canonical_keyframe":  c.canonical_keyframe,
                                            "semanticAnnotation": c.semantic_annotation,
                                            "ca_constraint": True}
                mp_constraints.constraints.append(GlobalTransformCAConstraint(self.skeleton, keyframe_constraint_desc, 1.0))

            elif c.constraint_type == SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION:
                positions = []
                for p in c.positions:
                    position = np.dot(inv_aligning_transform, [p[0], p[1], p[2], 1])[:3]
                    positions.append(position)
                keyframe_constraint_desc = {"joint": c.joint_names,
                                                "positions": positions,
                                                "orientations": c.orientations,
                                                "n_canonical_frames": c.n_canonical_frames,
                                                "canonical_keyframe":  c.canonical_keyframe,
                                                "semanticAnnotation": c.semantic_annotation}
                mp_constraints.constraints.append(TwoHandConstraintSet(self.skeleton, keyframe_constraint_desc, c.precision, c.weight_factor))

            elif c.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE:
                pose_constraint = []
                for p in c.pose_constraint:
                    position = np.dot(inv_aligning_transform, [p[0], p[1], p[2], 1])[:3]
                    pose_constraint.append(position)
                pose_constraint_desc = {"keyframeLabel": "start","canonical_keyframe": c.canonical_keyframe, "frame_constraint": pose_constraint,
                                        "semanticAnnotation": c.semantic_annotation}
                pose_constraint = PoseConstraint(self.skeleton, pose_constraint_desc, c.precision, c.weight_factor)
                mp_constraints.constraints.append(pose_constraint)

            elif c.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_DIR_2D:
                dir = copy(c.direction_constraint)
                local_dir = np.dot(inv_aligning_transform, [dir[0], dir[1], dir[2], 0])[:3]
                dir_constraint_desc = {"canonical_keyframe":c.canonical_keyframe,
                                       "dir_vector": local_dir,
                                       "semanticAnnotation": c.semantic_annotation}
                dir_constraint = Direction2DConstraint(self.skeleton, dir_constraint_desc, c.precision, c.weight_factor)
                mp_constraints.constraints.append(dir_constraint)

            elif c.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_LOOK_AT:
                local_target_pos = copy(c.target_position)
                local_target_pos = np.dot(inv_aligning_transform, [local_target_pos[0], local_target_pos[1], local_target_pos[2], 1])[:3]
                lookat_constraint_desc = {"canonical_keyframe":c.canonical_keyframe,
                                          "dir_vector": local_target_pos,
                                          "semanticAnnotation": c.semantic_annotation}
                lookat_constraint = LookAtConstraint(self.skeleton, lookat_constraint_desc, c.precision, c.weight_factor)
                mp_constraints.constraints.append(lookat_constraint)
        return mp_constraints

    def convert_to_ik_constraints(self, motion_state_graph, frame_offset, time_function=None, constrain_orientation=True):
        builder = IKConstraintsBuilder(self.action_name, self.motion_primitive_name, motion_state_graph, self.skeleton)
        return builder.convert_to_ik_constraints(self.constraints, frame_offset, time_function, constrain_orientation)

    def get_ca_constraints(self):
        ca_constraints = list()
        for c in self.constraints:
            if c.constraint_type == SPATIAL_CONSTRAINT_TYPE_CA_CONSTRAINT and \
               c.joint_name in self.skeleton.free_joints_map.keys():
                free_joints = self.skeleton.free_joints_map[c.joint_name]
                ca_constraints.append(JointIKConstraint(c.joint_name, c.position, None, -1, free_joints, c.step_idx))
        return ca_constraints

