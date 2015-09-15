# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 12:00:15 2015

@author: erhe01
"""

from elementary_action_constraints import ElementaryActionConstraints
from spatial_constraints.trajectory_constraint import TrajectoryConstraint


class ElementaryActionConstraintsBuilder(object):
    """Generates ElementaryActionConstraints instances based in an MGInputFileReader.
    
    Parameters
    ----------
    * mg_input : MGInputFileReader
        Class to access constraints defined in an input file.
    * motion_primitive_graph : MotionPrimitiveGraph
        Contains a list of motion nodes that can generate short motion clips.
    """
    def __init__(self, mg_input_reader, motion_primitive_graph):
        self.mg_input = mg_input_reader
        self.motion_primitive_graph = motion_primitive_graph
        self.current_action_index = 0
        self.start_pose = self.mg_input.get_start_pose()
        self.n_actions = self.mg_input.get_number_of_actions()

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

    def get_mg_input_file(self):
        return self.mg_input.mg_input_file

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
        ### extract event
        #action_constraints.keyframe_event_list = dict()
        #for annotation in action_constraints.keyframe_annotations.values():
        #    if "keyframe" in annotation.keys() and "annotations" in annotation.keys():
         #       action_constraints.keyframe_event_list[annotation["keyframe"]] = annotation["annotations"]

    def _add_keyframe_constraints(self, action_constraints):
        node_group = self.motion_primitive_graph.node_groups[action_constraints.action_name]
        action_constraints.keyframe_constraints = self.mg_input.get_keyframe_constraints(self.current_action_index, node_group)
        if len(action_constraints.keyframe_constraints) > 0:
                action_constraints.contains_user_constraints = True

    def _add_trajectory_constraints(self, action_constraints):
        """ Note: only trajectories on the Hips joint are supported for path following with direction constraints
           the other trajectories are only used  for calculating the euclidean distance
        """
        root_joint_name = self.motion_primitive_graph.skeleton.root
        action_constraints.root_trajectory = self._create_trajectory_from_constraint_desc(root_joint_name)
        action_constraints.trajectory_constraints = []
        for joint_name in self.motion_primitive_graph.skeleton.node_name_frame_map.keys():
            if joint_name != root_joint_name:
                trajectory_constraint = self._create_trajectory_from_constraint_desc(joint_name)
                if trajectory_constraint is not None:
                    action_constraints.trajectory_constraints.append(trajectory_constraint)

    def _create_trajectory_from_constraint_desc(self, joint_name, scale_factor=1.0):
        """ Create a spline based on a trajectory constraint definition read from the input file.
            Components containing None are set to 0, but marked as ignored in the unconstrained_indices list.
            Note all elements in constraints_list must have the same dimensions constrained and unconstrained.

        Returns
        -------
        * trajectory: ParameterizedSpline
        \t The trajectory defined by the control points from the trajectory_constraint
        """
        precision = 1.0
        control_points, unconstrained_indices = self.mg_input.get_trajectory_from_constraint_list(self.current_action_index, joint_name, scale_factor)
        if control_points is not None and unconstrained_indices is not None:
            trajectory_constraint = TrajectoryConstraint(joint_name, control_points, 0, unconstrained_indices, self.motion_primitive_graph.skeleton, precision)
            return trajectory_constraint
        else:
            return None
