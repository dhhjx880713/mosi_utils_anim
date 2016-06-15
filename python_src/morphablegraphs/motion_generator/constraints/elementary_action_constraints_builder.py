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
from ...utilities.log import write_log

REFERENCE_2D_OFFSET = np.array([0.0, -1.0])# components correspond to x, z - we assume the motions are initially oriented into that direction
LEFT_HAND_JOINT = "LeftToolEndSite"
RIGHT_HAND_JOINT = "RightToolEndSite"


class ElementaryActionConstraintsBuilder(object):
    """Generates ElementaryActionConstraints instances based in an MGInputFormatReader.
    
    Parameters
    ----------
    * mg_input : MGInputFormatReader
        Class to access constraints defined in an input file.
    * motion_state_graph : MotionStateGraph
        Contains a list of motion nodes that can generate short motion clips.
    """
    def __init__(self, motion_state_graph, algorithm_config):
        self.mg_input = None
        self.motion_state_graph = motion_state_graph
        self.default_constraint_weight = 1.0
        self.constraint_precision = 1.0
        self.set_algorithm_config(algorithm_config)

    def set_algorithm_config(self, algorithm_config):
        self.closest_point_search_accuracy = algorithm_config["trajectory_following_settings"]["closest_point_search_accuracy"]
        self.closest_point_search_max_iterations = algorithm_config["trajectory_following_settings"]["closest_point_search_max_iterations"]
        self.default_spline_type = algorithm_config["trajectory_following_settings"]["spline_type"]
        self.control_point_distance_threshold = algorithm_config["trajectory_following_settings"]["control_point_filter_threshold"]
        self.collision_avoidance_constraints_mode = algorithm_config["collision_avoidance_constraints_mode"]
        self.spline_arc_length_parameter_granularity = algorithm_config["trajectory_following_settings"]["arc_length_granularity"]

    def build_list_from_input_file(self, mg_input):
        """
        Returns:
        --------
        * action_constraints : list<ElementarActionConstraints>
          List of constraints for the elementary actions extracted from an input file.
        """
        self.mg_input = mg_input
        self._init_start_pose(mg_input)
        action_constraints_list = []
        for idx in xrange(self.mg_input.get_number_of_actions()):
            action_constraints_list.append(self._build_action_constraint(idx))
        return action_constraints_list

    def _build_action_constraint(self, action_index):
        action_constraints = ElementaryActionConstraints()
        action_constraints.motion_state_graph = self.motion_state_graph
        action_constraints.action_name = self.mg_input.get_elementary_action_name(action_index)
        action_constraints.start_pose = self.get_start_pose()
        self._add_keyframe_constraints(action_constraints, action_index)
        self._add_keyframe_annotations(action_constraints, action_index)
        self._add_trajectory_constraints(action_constraints, action_index)
        action_constraints._initialized = True
        return action_constraints

    def _init_start_pose(self, mg_input):
        """ Sets the pose at the beginning of the elementary action sequence
            Estimates the optimal start orientation from the constraints if none is given.
        """
        self.start_pose = mg_input.get_start_pose()
        if self.start_pose["orientation"] is None:
            root_trajectories = self._create_trajectory_constraints_for_joint(0, self.motion_state_graph.skeleton.root)
            if len(root_trajectories) > 0:
                if root_trajectories[0] is None:
                    self.start_pose["orientation"] = [0, 0, 0]
                else:
                    self.start_pose["orientation"] = self.get_start_orientation_from_trajectory(root_trajectories[0])
            write_log("Set start orientation from trajectory to", self.start_pose["orientation"])

    def get_start_pose(self):
        return self.start_pose

    def get_start_orientation_from_trajectory(self, root_trajectory):
        start, tangent, angle = root_trajectory.get_angle_at_arc_length_2d(0.0, REFERENCE_2D_OFFSET)
        return [0, angle, 0]

    def _add_keyframe_annotations(self, action_constraints, index):
        if index > 0:
            action_constraints.prev_action_name = self.mg_input.get_elementary_action_name(index - 1)
        action_constraints.keyframe_annotations = self.mg_input.get_keyframe_annotations(index)

    def _add_keyframe_constraints(self, action_constraints, index):
        node_group = self.motion_state_graph.node_groups[action_constraints.action_name]
        action_constraints.keyframe_constraints = self.mg_input.get_ordered_keyframe_constraints(index, node_group)
        if len(action_constraints.keyframe_constraints) > 0:
            action_constraints.contains_user_constraints = self._has_user_defined_constraints(action_constraints)
            self._merge_two_hand_constraints(action_constraints)
        #print action_constraints.action_name, action_constraints.keyframe_constraints, action_constraints.contains_user_constraints

    def _has_user_defined_constraints(self, action_constraints):
        for keyframe_label_constraints in action_constraints.keyframe_constraints.values():
                if len(keyframe_label_constraints) > 0:
                    if len(keyframe_label_constraints[0]) > 0:
                        return True
        return False

    def _merge_two_hand_constraints(self, action_constraints):
        """ Create a special constraint if two hand joints are constrained on the same keyframe
        """
        for mp_name in action_constraints.keyframe_constraints.keys():
            keyframe_constraints_map = self._map_constraints_by_label(action_constraints.keyframe_constraints[mp_name])
            action_constraints.keyframe_constraints[mp_name], merged_constraints = \
                self._merge_two_hand_constraints_in_keyframe_label_map(keyframe_constraints_map)
            if merged_constraints:
                action_constraints.contains_two_hands_constraints = True

    def _map_constraints_by_label(self, keyframe_constraints):
        """ separate constraints based on keyframe label
        """
        keyframe_constraints_map = dict()
        for desc in keyframe_constraints:
            keyframe_label = desc["semanticAnnotation"]["keyframeLabel"]
            if keyframe_label not in keyframe_constraints_map.keys():
                keyframe_constraints_map[keyframe_label] = list()
            keyframe_constraints_map[keyframe_label].append(desc)
        return keyframe_constraints_map

    def _merge_two_hand_constraints_in_keyframe_label_map(self, keyframe_constraints_map):
        """perform the merging for specific keyframe labels
        """
        merged_constraints = False
        merged_keyframe_constraints = list()
        for keyframe_label in keyframe_constraints_map.keys():
            new_constraint_list, is_merged = self._merge_two_hand_constraint_for_label(keyframe_constraints_map[keyframe_label])
            merged_keyframe_constraints += new_constraint_list
            if is_merged:
                merged_constraints = True
        return merged_keyframe_constraints, merged_constraints

    def _merge_two_hand_constraint_for_label(self, constraint_list):
        left_hand_indices = [index for (index, desc) in enumerate(constraint_list) if desc['joint'] == LEFT_HAND_JOINT]
        right_hand_indices = [index for (index, desc) in enumerate(constraint_list) if desc['joint'] == RIGHT_HAND_JOINT]
        if len(left_hand_indices) == 0 or len(right_hand_indices) == 0:
            #print "did not find two hand keyframe constraint"
            return constraint_list, False

        merged_constraint_list = list()
        left_hand_index = left_hand_indices[0]
        right_hand_index = right_hand_indices[0]
        merged_constraint_list.append(self._create_two_hand_constraint_definition(constraint_list, left_hand_index, right_hand_index))
        merged_constraint_list += [desc for (index, desc) in enumerate(constraint_list)
                                        if index != left_hand_index and index != right_hand_index]
        return merged_constraint_list, True

    def _create_two_hand_constraint_definition(self, constraint_list, left_hand_index, right_hand_index):
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
        return merged_constraint_desc

    def _add_trajectory_constraints(self, action_constraints, action_index):
        """ Extracts the root_trajectory if it is found and trajectories for other joints.
            If semanticAnnotation is found they are treated as collision avoidance constraint.
        """
        action_constraints.trajectory_constraints = list()
        action_constraints.collision_avoidance_constraints = list()
        action_constraints.annotated_trajectory_constraints = list()

        root_trajectories = self._create_trajectory_constraints_for_joint(action_index, self.motion_state_graph.skeleton.root)
        if len(root_trajectories) > 0:
            action_constraints.root_trajectory = root_trajectories[0]
        for joint_name in self.motion_state_graph.skeleton.node_name_frame_map.keys():
            if joint_name != self.motion_state_graph.skeleton.root:
                self._add_trajectory_constraint(action_constraints, action_index, joint_name)
        if self.collision_avoidance_constraints_mode == CA_CONSTRAINTS_MODE_SET and len(action_constraints.collision_avoidance_constraints) > 0:
            self._add_ca_trajectory_constraint_set(action_constraints)

    def _add_trajectory_constraint(self, action_constraints, action_index, joint_name):
        trajectory_constraints = self._create_trajectory_constraints_for_joint(action_index, joint_name)
        for c in trajectory_constraints:
            if c is not None:
                if c.is_collision_avoidance_constraint:
                    action_constraints.collision_avoidance_constraints.append(c)
                if c.semantic_annotation is not None:
                    action_constraints.annotated_trajectory_constraints.append(c)
                else:
                    action_constraints.trajectory_constraints.append(c)

    def _add_ca_trajectory_constraint_set(self, action_constraints):
        if action_constraints.root_trajectory is not None:
           joint_trajectories = [action_constraints.root_trajectory] + action_constraints.collision_avoidance_constraints
           joint_names = [action_constraints.root_trajectory.joint_name] + [traj.joint_name for traj in joint_trajectories]
        else:
           joint_trajectories = action_constraints.collision_avoidance_constraints
           joint_names = [traj.joint_name for traj in joint_trajectories]

        action_constraints.ca_trajectory_set_constraint = TrajectorySetConstraint(joint_trajectories, joint_names,
                                                                                  self.motion_state_graph.skeleton,
                                                                                  self.constraint_precision,
                                                                                  self.default_constraint_weight)

    def _create_trajectory_constraints_for_joint(self, action_index, joint_name):
        """ Create a spline based on a trajectory constraint definition read from the input file.
            Components containing None are set to 0, but marked as ignored in the unconstrained_indices list.
            Note all elements in constraints_list must have the same dimensions constrained and unconstrained.

        Returns
        -------
        * trajectory: List(TrajectoryConstraint)
        \t The trajectory constraints defined by the control points from the
            trajectory_constraint or an empty list if there is no constraint
        """
        desc = self.mg_input.extract_trajectory_desc(action_index, joint_name, self.control_point_distance_threshold)
        traj_constraints = list()
        for idx, control_points in enumerate(desc["control_points_list"]):
            if control_points is None:
                continue
            else:
                traj_constraint = TrajectoryConstraint(joint_name, control_points,
                                              self.default_spline_type, 0.0,
                                              desc["unconstrained_indices"],
                                              self.motion_state_graph.skeleton,
                                              self.constraint_precision, self.default_constraint_weight,
                                              self.closest_point_search_accuracy,
                                              self.closest_point_search_max_iterations,
                                              self.spline_arc_length_parameter_granularity)
                traj_constraint.semantic_annotation = desc["semantic_annotation"]
                if desc["active_regions"][idx] is not None:
                    # only collision avoidance constraints have a active regions
                    traj_constraint.is_collision_avoidance_constraint = True
                    self._set_active_range_from_region(traj_constraint, desc["active_regions"][idx])
                traj_constraints.append(traj_constraint)
        return traj_constraints

    def _set_active_range_from_region(self, traj_constraint, active_region):
        if active_region["start_point"] is not None and active_region["end_point"] is not None:
            range_start, closest_point = traj_constraint.get_absolute_arc_length_of_point(active_region["start_point"])
            range_end, closest_point = traj_constraint.get_absolute_arc_length_of_point(active_region["end_point"])
            traj_constraint.set_active_range(range_start, range_end)
