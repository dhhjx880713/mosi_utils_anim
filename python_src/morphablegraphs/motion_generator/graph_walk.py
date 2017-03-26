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
from ..constraints.spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION, SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION
from keyframe_event_list import KeyframeEventList
from ..utilities import write_log

DEFAULT_PLACE_ACTION_LIST = ["placeRight", "placeLeft","insertRight","insertLeft","screwRight", "screwLeft"] #list of actions in which the orientation constraints are ignored


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
    def __init__(self, motion_state_graph, mg_input, algorithm_config, start_pose=None, create_ca_vis_data=False):
        self.elementary_action_list = []
        self.steps = []
        self.motion_state_graph = motion_state_graph
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
        self.constrain_place_orientation = algorithm_config["inverse_kinematics_settings"]["constrain_place_orientation"]
        write_log("Use time parameters", self.use_time_parameters)
        self.keyframe_event_list = KeyframeEventList(create_ca_vis_data)
        self.place_action_list = DEFAULT_PLACE_ACTION_LIST

    def add_entry_to_action_list(self, action_name, start_step, end_step, action_constraints):
        self.elementary_action_list.append(HighLevelGraphWalkEntry(action_name, start_step, end_step, action_constraints))

    def convert_to_annotated_motion(self):
        self.motion_vector.apply_spatial_smoothing = self.apply_smoothing
        self.convert_graph_walk_to_quaternion_frames(use_time_parameters=self.use_time_parameters)
        self.keyframe_event_list.update_events(self, 0)
        annotated_motion_vector = AnnotatedMotionVector()
        annotated_motion_vector.frames = self.motion_vector.frames
        annotated_motion_vector.n_frames = self.motion_vector.n_frames
        annotated_motion_vector.frame_time = self.motion_state_graph.skeleton.frame_time
        annotated_motion_vector.keyframe_event_list = self.keyframe_event_list
        annotated_motion_vector.skeleton = self.motion_state_graph.skeleton
        annotated_motion_vector.mg_input = self.mg_input
        annotated_motion_vector.ik_constraints = self._create_ik_constraints()
        annotated_motion_vector.graph_walk = self
        return annotated_motion_vector

    def get_action_from_keyframe(self, keyframe):
        found_action_index = -1
        step_index = self.get_step_from_keyframe(keyframe)
        print "found step", step_index
        if step_index < 0:
            return found_action_index
        for action_index, action in enumerate(self.elementary_action_list):
            if action.start_step <= step_index <= action.end_step:
                found_action_index = action_index
        return found_action_index

    def get_step_from_keyframe(self, keyframe):
        found_step_index = -1
        for step_index, step in enumerate(self.steps):
            #Note the start_frame and end_frame are warped in update_temp_motion_vector
            #print step.start_frame, keyframe, step.end_frame
            if step.start_frame <= keyframe <= step.end_frame:
                found_step_index = step_index
        return found_step_index

    def convert_graph_walk_to_quaternion_frames(self, start_step=0, use_time_parameters=False):
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
            #write_log(step.node_key, len(step.parameters))
            quat_frames = self.motion_state_graph.nodes[step.node_key].back_project(step.parameters, use_time_parameters).get_motion_vector()
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

    def update_time_parameters(self, parameter_vector, start_step, end_step):
        offset = 0
        for step in self.steps[start_step:end_step]:
            new_gamma = parameter_vector[offset:offset+step.n_time_components]
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

    def _create_ik_constraints(self):
        ik_constraints = []
        for idx, action in enumerate(self.elementary_action_list):
            print "action", idx, action.start_step,self.steps[action.start_step].start_frame
            if not self.constrain_place_orientation and action.action_name in self.place_action_list:
                constrain_orientation = False
            else:
                constrain_orientation = True
            start_step = action.start_step
            end_step = action.end_step
            elementary_action_ik_constraints = dict()
            elementary_action_ik_constraints["keyframes"] = dict()
            elementary_action_ik_constraints["trajectories"] = list()
            elementary_action_ik_constraints["collision_avoidance"] = list()
            frame_offset = self.steps[start_step].start_frame
            for step in self.steps[start_step: end_step+1]:
                time_function = None
                if self.use_time_parameters:
                    time_function = self.motion_state_graph.nodes[step.node_key].back_project_time_function(step.parameters)
                step_keyframe_constraints = step.motion_primitive_constraints.convert_to_ik_constraints(self.motion_state_graph, frame_offset, time_function, constrain_orientation)
                elementary_action_ik_constraints["collision_avoidance"] += step.motion_primitive_constraints.get_ca_constraints()
                elementary_action_ik_constraints["keyframes"].update(step_keyframe_constraints)
                frame_offset += step.end_frame - step.start_frame + 1

            if self._algorithm_config["collision_avoidance_constraints_mode"] == "ik":
                elementary_action_ik_constraints["trajectories"] += self._create_ik_trajectory_constraints_from_ca_trajectories(idx)
            elementary_action_ik_constraints["trajectories"] += self._create_ik_trajectory_constraints_from_annotated_trajectories(idx)
            ik_constraints.append(elementary_action_ik_constraints)
        return ik_constraints

    def _create_ik_trajectory_constraints_from_ca_trajectories(self, action_idx):
        frame_annotation = self.keyframe_event_list.frame_annotation['elementaryActionSequence'][action_idx]
        trajectory_constraints = list()
        action = self.elementary_action_list[action_idx]
        for ca_constraint in action.action_constraints.collision_avoidance_constraints:
            traj_constraint = dict()
            traj_constraint["trajectory"] = ca_constraint
            traj_constraint["fixed_range"] = False  # search for closer start
            traj_constraint["constrain_orientation"] = False
            traj_constraint["start_frame"] = frame_annotation["startFrame"]
            traj_constraint["end_frame"] = frame_annotation["endFrame"]
            #TODO find a better solution than this workaround that undoes the joint name mapping from hands to tool bones for ca constraints
            if self.mg_input.activate_joint_mapping and ca_constraint.joint_name in self.mg_input.inverse_joint_name_map.keys():
                joint_name = self.mg_input.inverse_joint_name_map[ca_constraint.joint_name]
            else:
                joint_name = ca_constraint.joint_name

            traj_constraint["joint_name"] = joint_name
            traj_constraint["delta"] = 1.0
            trajectory_constraints.append(traj_constraint)
        return trajectory_constraints

    def _create_ik_trajectory_constraints_from_annotated_trajectories(self, action_idx):
        print "extract annotated trajectories"
        frame_annotation = self.keyframe_event_list.frame_annotation['elementaryActionSequence'][action_idx]
        start_frame = frame_annotation["startFrame"]
        trajectory_constraints = list()
        action = self.elementary_action_list[action_idx]
        for constraint in action.action_constraints.annotated_trajectory_constraints:
            label = constraint.semantic_annotation.keys()[0]
            print "trajectory constraint label",constraint.semantic_annotation.keys()
            action_name = action.action_name
            for step in self.steps[action.start_step: action.end_step+1]:
                motion_primitive_name = step.node_key[1]
                print "look for action annotation of",action_name,motion_primitive_name
                if motion_primitive_name not in self.motion_state_graph.node_groups[action_name].motion_primitive_annotation_regions:
                    continue
                annotations = self.motion_state_graph.node_groups[action_name].motion_primitive_annotation_regions[motion_primitive_name]
                print "action annotation",annotations,frame_annotation["startFrame"],frame_annotation["endFrame"]
                if label not in annotations.keys():
                    continue
                annotation_range = annotations[label]
                traj_constraint = dict()
                traj_constraint["trajectory"] = constraint
                traj_constraint["constrain_orientation"] = True
                traj_constraint["fixed_range"] = True
                time_function = None
                if self.use_time_parameters:
                    time_function = self.motion_state_graph.nodes[step.node_key].back_project_time_function(step.parameters)
                if time_function is None:
                    traj_constraint["start_frame"] = start_frame + annotation_range[0]
                    traj_constraint["end_frame"] = start_frame + annotation_range[1]
                else:
                    #add +1 for correct mapping TODO verify for all cases
                    traj_constraint["start_frame"] = start_frame + int(time_function[annotation_range[0]]) + 1
                    traj_constraint["end_frame"] = start_frame + int(time_function[annotation_range[1]]) + 1

                if self.mg_input.activate_joint_mapping and constraint.joint_name in self.mg_input.inverse_joint_name_map.keys():
                    joint_name = self.mg_input.inverse_joint_name_map[constraint.joint_name]
                else:
                    joint_name = constraint.joint_name

                traj_constraint["joint_name"] = joint_name
                traj_constraint["delta"] = 1.0
                print "create ik trajectory constraint from label", label
                trajectory_constraints.append(traj_constraint)
        return trajectory_constraints

    def get_average_keyframe_constraint_error(self):
        keyframe_constraint_errors = []
        step_index = 0
        prev_frames = None
        for step in self.steps:
            quat_frames = self.motion_state_graph.nodes[step.node_key].back_project(step.parameters, use_time_parameters=False).get_motion_vector()
            aligned_frames = align_quaternion_frames(quat_frames, prev_frames, self.motion_vector.start_pose)
            for constraint in step.motion_primitive_constraints.constraints:
                if constraint.constraint_type in [SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION, SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION] and\
                   not "generated" in constraint.semantic_annotation.keys():
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
        average_time_per_step = 0.0
        for step in self.steps:
            average_time_per_step += step.motion_primitive_constraints.time
        average_time_per_step /= len(self.steps)
        average_time_string = "average time per step " + str(average_time_per_step)
        return average_keyframe_error_string + "\n" + evaluations_string + "\n" + average_time_string + "\n" + error_string

    def export_generated_constraints(self, file_path="goals.path"):
        """ Converts constraints that were generated based on input constraints into a json dictionary for a debug visualization
        """
        root_control_point_data = []
        hand_constraint_data = []
        for idx, step in enumerate(self.steps):
            step_constraints = {"semanticAnnotation":{"step": idx}}
            for c in step.motion_primitive_constraints.constraints:
                if c.constraint_type == "keyframe_position" and c.joint_name == self.motion_state_graph.skeleton.root:
                    p = c.position
                    if p is not None:
                        step_constraints["position"] = [p[0], -p[2], None]
                elif c.constraint_type == "keyframe_2d_direction":
                        step_constraints["direction"] = c.direction_constraint.tolist()
                elif c.constraint_type == "ca_constraint":
                    #if c.constraint_type in ["RightHand", "LeftHand"]:
                    position = [c.position[0], -c.position[2], c.position[1]]
                    hand_constraint = {"position": position}
                    hand_constraint_data.append(hand_constraint)
            root_control_point_data.append(step_constraints)


        constraints = {"tasks": [{"elementaryActions":[{
                                                      "action": "walk",
                                                      "constraints": [{"joint": "Hips",
                                                                       "keyframeConstraints": root_control_point_data  },
                                                                      {"joint": "RightHand",
                                                                       "keyframeConstraints": hand_constraint_data}]
                                                      }]
                                 }]
                       }

        constraints["startPose"] = {"position":[0,0,0], "orientation": [0,0,0]}
        constraints["session"] = "session"
        with open(file_path, "wb") as out:
            json.dump(constraints, out)

    def get_number_of_actions(self):
        return len(self.elementary_action_list)

