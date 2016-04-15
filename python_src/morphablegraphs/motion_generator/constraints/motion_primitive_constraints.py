# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:42:13 2015

@author: erhe01
"""
import numpy as np
from copy import copy
from ...animation_data.motion_editing import align_quaternion_frames, transform_point
from .spatial_constraints.keyframe_constraints.global_transform_constraint import GlobalTransformConstraint
from .spatial_constraints.keyframe_constraints.two_hand_constraint import TwoHandConstraintSet
from .spatial_constraints.keyframe_constraints.pose_constraint import PoseConstraint
from .spatial_constraints.keyframe_constraints.direction_2d_constraint import Direction2DConstraint
from .spatial_constraints.keyframe_constraints.look_at_constraint import LookAtConstraint
from .spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION, SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION, SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE,SPATIAL_CONSTRAINT_TYPE_KEYFRAME_DIR_2D, SPATIAL_CONSTRAINT_TYPE_KEYFRAME_LOOK_AT
from .spatial_constraints import MGRDKeyframeConstraint
from ik_constraints import JointIKConstraint, TwoJointIKConstraint
from ...utilities.log import write_log
try:
    from mgrd import PoseConstraint as MGRDPoseConstraint
    from mgrd import SemanticConstraint as MGRDSemanticConstraint
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
        write_log("starting from:", self.step_start)
        write_log("the new goal for " + self.motion_primitive_name, "is", self.step_goal)
        write_log("arc length is: " + str(self.goal_arc_length))
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
        if not self.is_local:
            #find aligned frames once for all constraints#TODO use splines
            motion_spline.coeffs = align_quaternion_frames(motion_spline.coeffs, prev_frames, self.start_pose)

        #quat_frames = motion_spline.get_motion_vector()#TODO only get the frames that are needed
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

        #quat_frames = motion_spline.get_motion_vector()
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

    def convert_to_mgrd_constraints(self):
        mgrd_constraints = []
        for c in self.constraints:
            if c.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION:
                pose_constraint = MGRDPoseConstraint(c.joint_name, c.weight_factor, c.position, orientation=None)#[None, None, None, None]
                label = c.semantic_annotation["keyframeLabel"]# TODO add "end" annotation label to all motion primitives # "LeftFootContact"
                annotations = {label: True}
                semantic_constraint = MGRDSemanticConstraint(annotations, time=None)
                keyframe_constraint = MGRDKeyframeConstraint(pose_constraint, semantic_constraint)
                mgrd_constraints.append(keyframe_constraint)
        return mgrd_constraints

    def transform_constraints_to_local_cos(self):
        if self.is_local or self.aligning_transform is None:
            return self
        write_log("transform to local cos")
        inv_aligning_transform = np.linalg.inv(self.aligning_transform)
        mp_constraints = MotionPrimitiveConstraints()
        mp_constraints.start_pose = {"orientation": [0,0,0], "position": [0,0,0]}
        mp_constraints.is_local = True
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
                    #position = transform_point(position,  self.aligning_transform["orientation"], self.aligning_transform["translation"])
                    #print "transformed constraint",orig_position, position
                    keyframe_constraint_desc = {"joint": c.joint_name,
                                                "position": position,
                                                "n_canonical_frames": c.n_canonical_frames,
                                                "canonical_keyframe":  c.canonical_keyframe,
                                                "semanticAnnotation": c.semantic_annotation}
                    mp_constraints.constraints.append(GlobalTransformConstraint(self.skeleton, keyframe_constraint_desc, 1.0))
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
                lookat_constraint_desc = {
                                          "canonical_keyframe":c.canonical_keyframe,
                                       "dir_vector": local_target_pos,
                                        "semanticAnnotation": c.semantic_annotation}
                lookat_constraint = LookAtConstraint(self.skeleton, lookat_constraint_desc, c.precision, c.weight_factor)
                mp_constraints.constraints.append(lookat_constraint)


        return mp_constraints

    def convert_to_ik_constraints(self, frame_offset=0, time_function=None):
        ik_constraints = dict()
        for c in self.constraints:
            if (c.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION or c.constraint_type == SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION) \
                and "generated" not in c.semantic_annotation.keys():
                if time_function is not None:
                    keyframe = frame_offset+int(time_function[c.canonical_keyframe]) + 1  # add +1 to map the frame correctly TODO: test and verify for all cases
                else:
                    keyframe = frame_offset+c.canonical_keyframe
                if keyframe not in ik_constraints.keys():
                    ik_constraints[keyframe] = dict()
                    ik_constraints[keyframe]["single"] = []
                    ik_constraints[keyframe]["multiple"] = []
                if c.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION:
                    if c.joint_name in self.skeleton.free_joints_map.keys():
                        free_joints = self.skeleton.free_joints_map[c.joint_name]
                        ik_constraint = JointIKConstraint(c.joint_name, c.position, None, keyframe, free_joints)
                        ik_constraints[keyframe]["single"] .append(ik_constraint)
                elif c.constraint_type == SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION:
                    if c.joint_names[0] in self.skeleton.free_joints_map.keys() and \
                        c.joint_names[1] in self.skeleton.free_joints_map.keys():
                        free_joints = self.skeleton.reduced_free_joints_map[c.joint_names[0]]
                        ik_constraint = JointIKConstraint(c.joint_names[0], c.positions[0], None, keyframe, free_joints)
                        ik_constraints[keyframe]["single"] .append(ik_constraint)
                        free_joints = self.skeleton.reduced_free_joints_map[c.joint_names[1]]
                        ik_constraint = JointIKConstraint(c.joint_names[1], c.positions[1], None, keyframe, free_joints)
                        ik_constraints[keyframe]["single"] .append(ik_constraint)
                        ik_constraints[keyframe]["multiple"].append(None)#TODO replace with TwoJointIKConstraint
                        #free_joints = set()
                        #for joint_name in self.joint_names:
                        #    if joint_name in free_joints_map.keys():
                        #        free_joints.update(free_joints_map[joint_name])

                        #ik_constraint = TwoJointIKConstraint(c.joint_names, c.positions, c.target_center, c.target_delta, c.target_direction, keyframe)
                        #ik_constraints[keyframe]["multiple"].append(ik_constraint)

        return ik_constraints


