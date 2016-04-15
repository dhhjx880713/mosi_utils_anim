# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:39:41 2015

@author: erhe01
"""

from datetime import datetime
import json
import numpy as np
from ..animation_data import MotionVector, align_quaternion_frames
from annotated_motion_vector import AnnotatedMotionVector
from constraints.spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION, SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION
from keyframe_event_list import KeyframeEventList
from ..utilities import write_log


class GraphWalkEntry(object):
    def __init__(self, motion_state_graph, node_key, parameters, arc_length, start_frame, end_frame, motion_primitive_constraints=None):
        self.node_key = node_key
        self.parameters = parameters
        self.arc_length = arc_length
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.motion_primitive_constraints = motion_primitive_constraints
        self.n_spatial_components = motion_state_graph.nodes[node_key].get_n_spatial_components()
        self.n_time_components = motion_state_graph.nodes[node_key].get_n_time_components()


class HighLevelGraphWalkEntry(object):
    def __init__(self, action_name, start_step, end_step, action_constraints):
        self.action_name = action_name
        self.start_step = start_step
        self.end_step = end_step
        self.action_constraints = action_constraints


class GraphWalk(object):
    """ Product of the MotionGenerate class. Contains the graph walk used to generate the frames,
        a mapping of frame segments
        to elementary actions and a list of events for certain frames.
    """
    def __init__(self, motion_state_graph, mg_input, algorithm_config, start_pose=None):
        self.elementary_action_list = []
        self.steps = []
        self.motion_state_graph = motion_state_graph
        self.full_skeleton = motion_state_graph.full_skeleton
        self.step_count = 0
        self.mg_input = mg_input
        self._algorithm_config = algorithm_config
        self.motion_vector = MotionVector(algorithm_config)
        if start_pose is None:
            start_pose = mg_input.get_start_pose()
        self.motion_vector.start_pose = start_pose
        self.motion_vector.apply_spatial_smoothing = False
        self.use_time_parameters = algorithm_config["activate_time_variation"]
        self.apply_smoothing = algorithm_config["smoothing_settings"]["spatial_smoothing"]
        write_log("Use time parameters", self.use_time_parameters)
        self.keyframe_event_list = KeyframeEventList()

    def add_entry_to_action_list(self, action_name, start_step, end_step, action_constraints):
        self.elementary_action_list.append(HighLevelGraphWalkEntry(action_name, start_step, end_step, action_constraints))

    def update_temp_motion_vector(self, start_step=0, create_frame_annotation=True, use_time_parameters=False):
        self._convert_graph_walk_to_quaternion_frames(start_step, use_time_parameters=use_time_parameters)
        if create_frame_annotation:
            self.keyframe_event_list.update_events(self, start_step)

    def convert_to_annotated_motion(self):
        self.motion_vector.apply_spatial_smoothing = self.apply_smoothing
        self.update_temp_motion_vector(use_time_parameters=self.use_time_parameters)
        annotated_motion_vector = AnnotatedMotionVector()
        annotated_motion_vector.frames = self.motion_vector.frames
        annotated_motion_vector.n_frames = self.motion_vector.n_frames
        annotated_motion_vector.keyframe_event_list = self.keyframe_event_list
        annotated_motion_vector.skeleton = self.full_skeleton
        annotated_motion_vector.mg_input = self.mg_input
        frame_offset = 0
        ik_constraints = dict()
        ik_constraints["keyframes"] = dict()
        for step in self.steps:
            time_function = None
            if self.use_time_parameters:
                time_function = self.motion_state_graph.nodes[step.node_key].back_project_time_function(step.parameters)
            step_constraints = step.motion_primitive_constraints.convert_to_ik_constraints(frame_offset, time_function)
            ik_constraints["keyframes"].update(step_constraints)
            frame_offset += step.end_frame - step.start_frame + 1
        ik_constraints["trajectories"] = list()
        if self._algorithm_config["collision_avoidance_constraints_mode"] == "ik":
            ik_constraints["trajectories"] += self._create_ik_trajectory_constraints()

        annotated_motion_vector.ik_constraints = ik_constraints
        return annotated_motion_vector

    def _create_ik_trajectory_constraints(self):
        trajectory_constraints = list()
        n_actions = len(self.elementary_action_list)
        for action_idx in xrange(n_actions):
            frame_annotation = self.keyframe_event_list.frame_annotation['elementaryActionSequence'][action_idx]
            high_level_step = self.elementary_action_list[action_idx]
            for ca_constraint in high_level_step.action_constraints.collision_avoidance_constraints:
                traj_constraint = dict()
                traj_constraint["trajectory"] = ca_constraint
                traj_constraint["start_frame"] = frame_annotation["startFrame"]
                traj_constraint["end_frame"] = frame_annotation["endFrame"]
                if self.mg_input.activate_joint_mapping and ca_constraint.joint_name in self.mg_input.inverse_joint_name_map.keys():
                    joint_name = self.mg_input.inverse_joint_name_map[ca_constraint.joint_name]
                else:
                    joint_name = ca_constraint.joint_name
                traj_constraint["joint_name"] = joint_name
                traj_constraint["delta"] = 1.0
                trajectory_constraints.append(traj_constraint)
        return trajectory_constraints

    def _convert_graph_walk_to_quaternion_frames(self, start_step=0, use_time_parameters=False):
        """
        :param start_step:
        :return:
        """
        if start_step == 0:
            start_frame = 0
        else:
            start_frame = self.steps[start_step].start_frame
        self.motion_vector.clear(end_frame=start_frame)
        for step in self.steps[start_step:]:
            step.start_frame = start_frame
            quat_frames = self.motion_state_graph.nodes[step.node_key].back_project(step.parameters, use_time_parameters).get_motion_vector()
            #write_log(step.node_key, len(quat_frames), use_time_parameters)
            self.motion_vector.append_frames(quat_frames)
            step.end_frame = self.get_num_of_frames()-1
            start_frame = step.end_frame + 1

    def get_global_spatial_parameter_vector(self, start_step=0):
        initial_guess = []
        for step in self.steps[start_step:]:
            initial_guess += step.parameters[:step.n_spatial_components].tolist()
        return initial_guess

    def get_global_time_parameter_vector(self, start_step=0):
        initial_guess = []
        for step in self.steps[start_step:]:
            initial_guess += step.parameters[step.n_spatial_components:].tolist()
        return initial_guess

    def update_spatial_parameters(self, parameter_vector, start_step=0):
        write_log("update spatial parameters")
        offset = 0
        for step in self.steps[start_step:]:
            new_alpha = parameter_vector[offset:offset+step.n_spatial_components]
            step.parameters[:step.n_spatial_components] = new_alpha
            offset += step.n_spatial_components

    def update_time_parameters(self, parameter_vector, start_step=0):
        offset = 0
        for step in self.steps[start_step:]:
            new_gamma = parameter_vector[offset:offset+step.n_time_components]
            #print new_gamma
            step.parameters[step.n_spatial_components:] = new_gamma
            offset += step.n_time_components

    def append_quat_frames(self, new_frames):
        self.motion_vector.append_frames(new_frames)

    def get_quat_frames(self):
        return self.motion_vector.frames

    def get_num_of_frames(self):
        return self.motion_vector.n_frames

    def update_frame_annotation(self, action_name, start_frame, end_frame):
        """ Adds a dictionary to self.frame_annotation marking start and end
            frame of an action.
        """
        self.keyframe_event_list.update_frame_annotation(action_name, start_frame, end_frame)

    def get_average_keyframe_constraint_error(self):
        keyframe_constraint_errors = []
        step_index = 0
        prev_frames = None
        for step in self.steps:
            quat_frames = self.motion_state_graph.nodes[step.node_key].back_project(step.parameters, use_time_parameters=False).get_motion_vector()
            aligned_frames = align_quaternion_frames(quat_frames, prev_frames, self.motion_vector.start_pose)
            for constraint in step.motion_primitive_constraints.constraints:
                if (constraint.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION or constraint.constraint_type == SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION) and\
                    not ("generated" in constraint.semantic_annotation.keys()):
                    #joint_position = self.skeleton.nodes[constraint.joint_name].get_global_position(aligned_frames[constraint.canonical_keyframe])
                    #joint_position = constraint.skeleton.get_cartesian_coordinates_from_quaternion(constraint.joint_name, aligned_frames[constraint.canonical_keyframe])
                    #print "position constraint", joint_position, constraint.position
                    error = constraint.evaluate_motion_sample(aligned_frames)
                    print error
                    keyframe_constraint_errors.append(error)
            prev_frames = aligned_frames
            step_index += 1
        if len(keyframe_constraint_errors) > 0:
            return np.average(keyframe_constraint_errors)
        else:
            return -1

    def get_generated_constraints(self):
        step_count = 0
        generated_constraints = dict()
        for step in self.steps:
            key = str(step.node_key) + str(step_count)
            generated_constraints[key] = []
            for constraint in step.motion_primitive_constraints.constraints:
                if constraint.is_generated():
                    generated_constraints[key].append(constraint.position)
            step_count += 1
        #generated_constraints = np.array(generated_constraints).flatten()
        return generated_constraints

    def get_average_error(self):
        average_error = 0
        for step in self.steps:
            average_error += step.motion_primitive_constraints.min_error
        if average_error > 0:
            average_error /= len(self.steps)
        return average_error

    def get_number_of_object_evaluations(self):
        objective_evaluations = 0
        for step in self.steps:
            objective_evaluations += step.motion_primitive_constraints.evaluations
        return objective_evaluations

    def print_statistics(self):
        print self.get_statistics_string()

    def get_statistics_string(self):
        average_error = self.get_average_error()
        evaluations_string = "total number of objective evaluations " + str(self.get_number_of_object_evaluations())
        error_string = "average error for " + str(len(self.steps)) + \
                       " motion primitives: " + str(average_error)
        average_keyframe_error = self.get_average_keyframe_constraint_error()
        average_keyframe_error_string = "average keyframe constraint error " + str(average_keyframe_error)
        return average_keyframe_error_string + "\n" + evaluations_string + "\n" + error_string

    def export_generated_constraints(self, file_path="goals.path"):
        json_control_point_data = []
        for idx, step in enumerate(self.steps):
            step_constraints = {"semanticAnnotation":{"step": idx}}
            for c in step.motion_primitive_constraints.constraints:
                if c.constraint_type == "keyframe_position":
                    p = c.position
                    if p is not None:
                        step_constraints["position"] = [p[0], -p[2], None]
                elif c.constraint_type == "keyframe_2d_direction":
                        step_constraints["direction"] = c.direction_constraint.tolist()
                json_control_point_data.append(step_constraints)

        constraints = {"tasks": [{"elementaryActions":[{
                                                      "action": "walk",
                                                      "constraints": [{"joint": "Hips",
                                                                       "keyframeConstraints": json_control_point_data  } ]
                                                      }]
                                 }]
                       }

        constraints["startPose"] = {"position":[0,0,0], "orientation": [0,0,0]}
        constraints["session"] = "session"
        with open(file_path, "wb") as out:
            json.dump(constraints, out)
