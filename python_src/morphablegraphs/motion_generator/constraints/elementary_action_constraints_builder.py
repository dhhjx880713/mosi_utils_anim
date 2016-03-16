# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 12:00:15 2015

@author: erhe01
"""
import numpy as np
from elementary_action_constraints import ElementaryActionConstraints
from spatial_constraints import TrajectoryConstraint
from spatial_constraints import TrajectorySetConstraint
from . import *

REFERENCE_2D_OFFSET = np.array([0, -1])# components correspond to x, z - we assume the motions are initially oriented into that direction

class ElementaryActionConstraintsBuilder(object):
    """Generates ElementaryActionConstraints instances based in an MGInputFileReader.
    
    Parameters
    ----------
    * mg_input : MGInputFileReader
        Class to access constraints defined in an input file.
    * motion_state_graph : MotionStateGraph
        Contains a list of motion nodes that can generate short motion clips.
    """
    def __init__(self, motion_state_graph, algorithm_config):
        self.mg_input = None
        self.motion_state_graph = motion_state_graph
        self.set_algorithm_config(algorithm_config)

    def set_algorithm_config(self, algorithm_config):
        self.closest_point_search_accuracy = algorithm_config["trajectory_following_settings"]["closest_point_search_accuracy"]
        self.closest_point_search_max_iterations = algorithm_config["trajectory_following_settings"]["closest_point_search_max_iterations"]
        self.default_spline_type = algorithm_config["trajectory_following_settings"]["spline_type"]
        self.control_point_distance_threshold = algorithm_config["trajectory_following_settings"]["control_point_filter_threshold"]
        self.collision_avoidance_constraints_mode = algorithm_config["collision_avoidance_constraints_mode"]

    def build_list_from_input_file(self, mg_input):
        """
        Returns:
        --------
        * action_constraints : list<ElementarActionConstraints>
          List of constraints for the elementary actions extracted from an input file.
        """
        self.mg_input = mg_input
        self.start_pose = mg_input.get_start_pose()
        action_constaints_list = []
        for idx in xrange(self.mg_input.get_number_of_actions()):
            action_constaints_list.append(self._build_action_constraint(idx))
        return action_constaints_list

    def _build_action_constraint(self, action_index):
        action_constraints = ElementaryActionConstraints()
        action_constraints.motion_state_graph = self.motion_state_graph
        action_constraints.action_name = self.mg_input.get_elementary_action_name(action_index)
        self._add_keyframe_constraints(action_constraints, action_index)
        self._add_keyframe_annotations(action_constraints, action_index)
        self._add_trajectory_constraints(action_constraints, action_index)
        self._set_start_pose(action_constraints)
        action_constraints._initialized = True
        return action_constraints

    def _set_start_pose(self, action_constraints):
        """ Sets the pose at the beginning of the elementary action sequence
        Determines the optimal start orientation from the constraints if none is given.
        :param action_constraints:
        :return:
        """
        if self.start_pose["orientation"] is None:
            if action_constraints.root_trajectory is not None:
                start, tangent, angle = action_constraints.root_trajectory.get_angle_at_arc_length_2d(0.0, REFERENCE_2D_OFFSET)
                self.start_pose["orientation"] = [0, angle, 0]
            else:
                self.start_pose["orientation"] = [0, 0, 0]
            print "set start orientation", self.start_pose["orientation"]
        action_constraints.start_pose = self.start_pose

    def _add_keyframe_annotations(self, action_constraints, index):
        if index > 0:
            action_constraints.prev_action_name = self.mg_input.get_elementary_action_name(index - 1)
        action_constraints.keyframe_annotations = self.mg_input.get_keyframe_annotations(index)

    def _add_keyframe_constraints(self, action_constraints, index):
        node_group = self.motion_state_graph.node_groups[action_constraints.action_name]
        action_constraints.keyframe_constraints = self.mg_input.get_ordered_keyframe_constraints(index, node_group)
        action_constraints.contains_user_constraints = False
        if len(action_constraints.keyframe_constraints) > 0:
            action_constraints.contains_user_constraints = self._has_user_defined_constraints(action_constraints)

            self._merge_two_hand_constraints(action_constraints)
        print action_constraints.action_name, action_constraints.keyframe_constraints, action_constraints.contains_user_constraints, "#######################"

    def _has_user_defined_constraints(self, action_constraints):
        for keyframe_label_constraints in action_constraints.keyframe_constraints.values():
                if len(keyframe_label_constraints) > 0:
                    if len(keyframe_label_constraints[0]) > 0:
                        return True
        return False

    def _merge_two_hand_constraints(self, action_constraints):
        """ Create a special constraint if two hand joints are constrained on the same keyframe
        """
        for motion_primitive_name in action_constraints.keyframe_constraints.keys():
            #separate constraints based on keyframe label
            keyframe_label_lists = dict()
            for desc in action_constraints.keyframe_constraints[motion_primitive_name]:
                print desc
                keyframe_label = desc["semanticAnnotation"]["keyframeLabel"]
                if keyframe_label not in keyframe_label_lists.keys():
                    keyframe_label_lists[keyframe_label] = list()
                keyframe_label_lists[keyframe_label].append(desc)
            #combine them back together and perform the merging for specific keyframe labels
            merged_keyframe_constraints = list()
            for keyframe_label in keyframe_label_lists.keys():
                two_hand_constraint_list, found_two_constraint = self._merge_two_hand_constraint_for_label(keyframe_label_lists[keyframe_label])
                merged_keyframe_constraints += two_hand_constraint_list
                if found_two_constraint:
                    action_constraints.contains_two_hands_constraints = True
            action_constraints.keyframe_constraints[motion_primitive_name] = merged_keyframe_constraints

    def _merge_two_hand_constraint_for_label(self, constraint_list):
        merged_constraint_list = list()
        left_hand_indices = [index for (index, desc) in enumerate(constraint_list) if desc['joint'] == LEFT_HAND_JOINT]
        right_hand_indices = [index for (index, desc) in enumerate(constraint_list) if desc['joint'] == RIGHT_HAND_JOINT]
        if len(left_hand_indices) > 0 and len(right_hand_indices) > 0:

            left_hand_index = left_hand_indices[0]
            right_hand_index = right_hand_indices[0]

            joint_names = [LEFT_HAND_JOINT, RIGHT_HAND_JOINT]
            positions = [constraint_list[left_hand_index]["position"],
                         constraint_list[right_hand_index]["position"]]
            orientations = [constraint_list[left_hand_index]["orientation"],
                            constraint_list[right_hand_index]["orientation"]]
            time = constraint_list[left_hand_index]["time"]
            semantic_annotation = constraint_list[left_hand_index]["semanticAnnotation"]
            merged_constraint_desc = {"joint": joint_names,
                           "positions": positions,
                           "orientations": orientations,
                           "time": time,
                           "merged": True,
                           "semanticAnnotation": semantic_annotation}
            #print "merged keyframe constraint", merged_constraint_desc
            merged_constraint_list.append(merged_constraint_desc)
            merged_constraint_list += [desc for (index, desc) in enumerate(constraint_list)
                                            if index == left_hand_index and \
                                               index == right_hand_index]
            return merged_constraint_list, True
        else:
            #print "did not find two hand keyframe constraint"
            return constraint_list, False

    def _add_trajectory_constraints(self, action_constraints, action_index):
        """ Extracts the root_trajectory if it is found and trajectories for other joints.
            If semanticAnnotation is found they are treated as collision avoidance constraint.
        """
        root_joint_name = self.motion_state_graph.skeleton.root
        action_constraints.root_trajectory = self._create_trajectory_constraint(action_index, root_joint_name)
        action_constraints.trajectory_constraints = []
        action_constraints.collision_avoidance_constraints = []
        for joint_name in self.motion_state_graph.skeleton.node_name_frame_map.keys():
            if joint_name != root_joint_name:
                trajectory_constraint = self._create_trajectory_constraint(action_index, joint_name)
                if trajectory_constraint is not None:
                    # decide if it is a collision avoidance constraint based on whether or not it has a range
                    if trajectory_constraint.range_start is None:
                        action_constraints.trajectory_constraints.append(trajectory_constraint)
                    else:
                        action_constraints.collision_avoidance_constraints.append(trajectory_constraint)
        if self.collision_avoidance_constraints_mode == CA_CONSTRAINTS_MODE_SET and len(action_constraints.collision_avoidance_constraints) > 0:
            if action_constraints.root_trajectory is not None:
                   joint_trajectories = [action_constraints.root_trajectory] + action_constraints.collision_avoidance_constraints
                   joint_names = [action_constraints.root_trajectory.joint_name] + [traj.joint_name for traj in joint_trajectories]
            else:
                   joint_trajectories = action_constraints.collision_avoidance_constraints
                   joint_names = [traj.joint_name for traj in joint_trajectories]

            action_constraints.ca_trajectory_set_constraint = TrajectorySetConstraint(joint_trajectories,
                                                                                    joint_names,
                                                                                    self.motion_state_graph.skeleton, 1.0, 1.0)

    def _create_trajectory_constraint(self, action_index, joint_name, scale_factor=1.0):
        """ Create a spline based on a trajectory constraint definition read from the input file.
            Components containing None are set to 0, but marked as ignored in the unconstrained_indices list.
            Note all elements in constraints_list must have the same dimensions constrained and unconstrained.

        Returns
        -------
        * trajectory: ParameterizedSpline
        \t The trajectory defined by the control points from the trajectory_constraint or None if there is no constraint
        """
        trajectory_constraint = None
        precision = 1.0
        control_points, unconstrained_indices, active_region = self.mg_input.get_trajectory_from_constraint_list(action_index, joint_name, scale_factor, self.control_point_distance_threshold)
        if control_points is not None and unconstrained_indices is not None:

            trajectory_constraint = TrajectoryConstraint(joint_name,
                                                         control_points,
                                                         self.default_spline_type,
                                                         0,
                                                         unconstrained_indices,
                                                         self.motion_state_graph.skeleton,
                                                         precision,
                                                         1.0,
                                                         self.closest_point_search_accuracy,
                                                         self.closest_point_search_max_iterations)
            if active_region is not None and active_region["start_point"] is not None and active_region["end_point"] is not None:
                range_start, closest_point = trajectory_constraint.get_absolute_arc_length_of_point(active_region["start_point"])
                range_end, closest_point = trajectory_constraint.get_absolute_arc_length_of_point(active_region["end_point"])
                trajectory_constraint.set_active_range(range_start, range_end)

        return trajectory_constraint
