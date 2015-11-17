# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 12:00:15 2015

@author: erhe01
"""

from elementary_action_constraints import ElementaryActionConstraints
from spatial_constraints.trajectory_constraint import TrajectoryConstraint
from spatial_constraints.trajectory_set_constraint import TrajectorySetConstraint
from . import *

class ElementaryActionConstraintsBuilder(object):
    """Generates ElementaryActionConstraints instances based in an MGInputFileReader.
    
    Parameters
    ----------
    * mg_input : MGInputFileReader
        Class to access constraints defined in an input file.
    * motion_primitive_graph : MotionPrimitiveGraph
        Contains a list of motion nodes that can generate short motion clips.
    """
    def __init__(self, mg_input_reader, motion_primitive_graph, algorithm_config):
        self.mg_input = mg_input_reader
        self.motion_primitive_graph = motion_primitive_graph
        self.current_action_index = 0
        self.start_pose = self.mg_input.get_start_pose()
        self.n_actions = self.mg_input.get_number_of_actions()
        self.closest_point_search_accuracy = algorithm_config["trajectory_following_settings"]["closest_point_search_accuracy"]
        self.closest_point_search_max_iterations = algorithm_config["trajectory_following_settings"]["closest_point_search_max_iterations"]
        self.default_spline_type = algorithm_config["trajectory_following_settings"]["spline_type"]
        self.control_point_distance_threshold = algorithm_config["trajectory_following_settings"]["control_point_filter_threshold"]
        self.collision_avoidance_constraints_mode = algorithm_config["collision_avoidance_constraints_mode"]

    def reset_counter(self):
        self.current_action_index = 0

    def get_next_elementary_action_constraints(self):
        """
        Returns:
        --------
        * action_constraints : ElementarActionConstraints
          Constraints for the next elementary action extracted from an input file.
        """
        if self.current_action_index < self.n_actions:
            action_constraints = self._build()
            self.current_action_index += 1
            return action_constraints
        else:
            return None

    def _build(self):
        if self.current_action_index < self.n_actions:
            action_constraints = ElementaryActionConstraints()
            action_constraints.motion_primitive_graph = self.motion_primitive_graph
            action_constraints.action_name = self.mg_input.get_elementary_action_name(self.current_action_index)
            action_constraints.start_pose = self.start_pose
            self._add_keyframe_constraints(action_constraints)
            self._add_keyframe_annotations(action_constraints)
            self._add_trajectory_constraints(action_constraints)
            action_constraints._initialized = True
            return action_constraints

    def _add_keyframe_annotations(self, action_constraints):
        if self.current_action_index > 0:
            action_constraints.prev_action_name = self.mg_input.get_elementary_action_name(self.current_action_index - 1)
        action_constraints.keyframe_annotations = self.mg_input.get_keyframe_annotations(self.current_action_index)

    def _add_keyframe_constraints(self, action_constraints):
        node_group = self.motion_primitive_graph.node_groups[action_constraints.action_name]
        action_constraints.keyframe_constraints = self.mg_input.get_ordered_keyframe_constraints(self.current_action_index, node_group)
        action_constraints.contains_user_constraints = False
        if len(action_constraints.keyframe_constraints) > 0:
            for keyframe_label_constraints in  action_constraints.keyframe_constraints.values():
                if len(keyframe_label_constraints[0]) > 0:
                    action_constraints.contains_user_constraints = True
                    break

            self._merge_two_hand_constraints(action_constraints)
        print action_constraints.action_name, action_constraints.keyframe_constraints, action_constraints.contains_user_constraints, "#######################"

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
                merged_keyframe_constraints += self._merge_two_hand_constraint_for_label(keyframe_label_lists[keyframe_label])
            action_constraints.keyframe_constraints[motion_primitive_name] = merged_keyframe_constraints

    def _merge_two_hand_constraint_for_label(self, constraint_list):
        merged_constraint_list = list()
        left_hand_indices = [index for (index, desc) in enumerate(constraint_list) if desc['joint'] == LEFT_HAND_JOINT]
        right_hand_indices = [index for (index, desc) in enumerate(constraint_list) if desc['joint'] == RIGHT_HAND_JOINT]
        print left_hand_indices
        print right_hand_indices
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
            print "merged keyframe constraint", merged_constraint_desc
            merged_constraint_list.append(merged_constraint_desc)
            merged_constraint_list += [desc for (index, desc) in enumerate(constraint_list)
                                            if index == left_hand_index and \
                                               index == right_hand_index]
            return merged_constraint_list
        else:
            print "did not find two hand keyframe constraint"
            return constraint_list

    def _add_trajectory_constraints(self, action_constraints):
        """ Extracts the root_trajectory if it is found and trajectories for other joints.
            If semanticAnnotation is found they are treated as collision avoidance constraint.
        """
        root_joint_name = self.motion_primitive_graph.skeleton.root
        action_constraints.root_trajectory = self._create_trajectory_from_constraint_desc(root_joint_name)
        action_constraints.trajectory_constraints = []
        action_constraints.collision_avoidance_constraints = []
        for joint_name in self.motion_primitive_graph.skeleton.node_name_frame_map.keys():
            if joint_name != root_joint_name:
                trajectory_constraint = self._create_trajectory_from_constraint_desc(joint_name)
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
                                                                                    self.motion_primitive_graph.skeleton, 1.0, 1.0)

    def _create_trajectory_from_constraint_desc(self, joint_name, scale_factor=1.0):
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
        control_points, unconstrained_indices, active_region = self.mg_input.get_trajectory_from_constraint_list(self.current_action_index, joint_name, scale_factor, self.control_point_distance_threshold)
        if control_points is not None and unconstrained_indices is not None:

            trajectory_constraint = TrajectoryConstraint(joint_name,
                                                         control_points,
                                                         self.default_spline_type,
                                                         0,
                                                         unconstrained_indices,
                                                         self.motion_primitive_graph.skeleton,
                                                         precision,
                                                         1.0,
                                                         self.closest_point_search_accuracy,
                                                         self.closest_point_search_max_iterations)
            if active_region is not None:
                range_start, closest_point = trajectory_constraint.get_absolute_arc_length_of_point(active_region["start_point"])
                range_end, closest_point = trajectory_constraint.get_absolute_arc_length_of_point(active_region["end_point"])
                trajectory_constraint.set_active_range(range_start, range_end)

        return trajectory_constraint
