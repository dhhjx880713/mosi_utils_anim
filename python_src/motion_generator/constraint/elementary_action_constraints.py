# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:42:41 2015

@author: erhe01
"""
from constraint_extraction import extract_trajectory_constraint,\
                            create_trajectory_from_constraint,\
                            extract_keyframe_constraint,\
                            extract_all_keyframe_constraints

class ElementaryActionConstraints(object):
    def __init__(self,action_index, motion_constraints):
        self.parent_constraint = motion_constraints
        self.action_name = motion_constraints.elementary_action_list[action_index]["action"]
        self.keyframe_annotations = motion_constraints.keyframe_annotations[action_index]
        self.constraints = motion_constraints.elementary_action_list[action_index]["constraints"]
        self.max_step = motion_constraints.max_step
        self.start_pose = motion_constraints.start_pose
        self._extract_constraints_from_motion_constraint_list()
                                         
    def get_subgraph(self):
        return self.parent_constraint.morphable_graph.subgraphs[self.action_name]
        
    def get_skeleton(self):
        return self.parent_constraint.morphable_graph.skeleton
        
    def _extract_constraints_from_motion_constraint_list(self):
        """ Extracts keyframe and trajectory constraints from constraint_list
        Returns:
        -------
        * trajectory: ParameterizedSpline
            Spline parameterized by arc length.
        * unconstrained_indices: list of indices
            lists of indices of degrees of freedom to ignore in the constraint evaluation.
        * keyframe_constraints: dict of lists
            Lists of constraints for each motion primitive in the subgraph.
        """
        morphable_subgraph = self.parent_constraint.morphable_graph.subgraphs[self.action_name]
        root_joint_name = self.parent_constraint.morphable_graph.skeleton.root# currently only trajectories on the Hips joint are supported
        self.trajectory, self.unconstrained_indices = self._extract_trajectory_from_constraint_list(self.constraints, root_joint_name)
    
        keyframe_constraints = extract_all_keyframe_constraints(self.constraints,
                                                                morphable_subgraph)
        self.keyframe_constraints = self._reorder_keyframe_constraints_for_motion_primitves(morphable_subgraph,
                                                                                 keyframe_constraints)



    def _reorder_keyframe_constraints_for_motion_primitves(self, morphable_subgraph, keyframe_constraints):
         """ Order constraints extracted by extract_all_keyframe_constraints for each state
         """
         constraints = {}#dict of lists
         #iterate over keyframe labels
         for label in keyframe_constraints.keys():
            state = morphable_subgraph.annotation_map[label]
            time_information = morphable_subgraph.motion_primitive_annotations[state][label]
            constraints[state] = []
            # iterate over joints constrained at that keyframe
            for joint_name in keyframe_constraints[label].keys():
                # iterate over constraints for that joint
                for c in keyframe_constraints[label][joint_name]:
                    # create constraint definition usable by the algorithm
                    # and add it to the list of constraints for that state
                    constraint_desc = extract_keyframe_constraint(joint_name,c,\
                                                morphable_subgraph,time_information)
                    constraints[state].append(constraint_desc)
         return constraints


    def _extract_trajectory_from_constraint_list(self, constraint_list, joint_name):
        """ Extract the trajectory information from the constraints and constructs
            a trajectory as an ParameterizedSpline instance.
        Returns:
        -------
        * trajectory: ParameterizedSpline
            Spline parameterized by arc length.
        * unconstrained_indices: list of indices
            Lists of indices of degrees of freedom to ignore in the constraint evaluation.
        """
        trajectory_constraint = extract_trajectory_constraint(constraint_list,joint_name)
        if  trajectory_constraint is not None:
            #print "found trajectory constraint"
            return create_trajectory_from_constraint(trajectory_constraint)
        else:
            return None, None
