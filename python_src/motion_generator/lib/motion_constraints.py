# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:26:29 2015

@author: erhe01
"""

import copy
from lib.input_processing import extract_keyframe_annotations, transform_point_from_cad_to_opengl_cs, \
                                    extract_all_keyframe_constraints,\
                                    extract_trajectory_constraint,\
                                    create_trajectory_from_constraint,\
                                    extract_keyframe_constraint
                                    
             




def prepare_keyframe_constraints_for_motion_primitves(morphable_subgraph,keyframe_constraints):
     """ Order constraints extracted by extract_all_keyframe_constraints for each state
     """
     constraints = {}#dict of lists
     #iterate over keyframe labels
     for label in keyframe_constraints.keys():
        state = morphable_subgraph.annotation_map[label]
        time_information = morphable_subgraph.mp_annotations[state][label]
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


def extract_trajectory_from_constraint_list(constraint_list,joint_name):
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

def extract_constraints_of_elementary_action(skeleton, morphable_subgraph, constraint_list):
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
    root_joint_name = skeleton.root# currently only trajectories on the Hips joint are supported
    trajectory, unconstrained_indices = extract_trajectory_from_constraint_list(constraint_list, root_joint_name)

    keyframe_constraints = extract_all_keyframe_constraints(constraint_list,
                                                            morphable_subgraph)
    keyframe_constraints = prepare_keyframe_constraints_for_motion_primitves(morphable_subgraph,
                                                                             keyframe_constraints)
    return trajectory,unconstrained_indices, keyframe_constraints
      
class ElementaryActionConstraints(object):
    def __init__(self,action_index,motion_constraints):
        self.action_name = motion_constraints.elementary_action_list[action_index]["action"]
        self.keyframe_annotations = motion_constraints.keyframe_annotations[action_index]
        self.constraints = motion_constraints.elementary_action_list[action_index]["constraints"]
        self.max_step = motion_constraints.max_step
        self.start_pose = motion_constraints.start_pose
        self.trajectory,self.unconstrained_indices, self.keyframe_constraints = \
            extract_constraints_of_elementary_action(motion_constraints.morphable_graph.skeleton, \
                                                     motion_constraints.morphable_graph.subgraphs[self.action_name],\
                                                     self.constraints)

class MotionConstraints(object):
    """
    Parameters
    ----------
    * mg_input : json data read from a file
        Contains elementary action list with constraints, start pose and keyframe annotations.
    * max_step : integer
        Sets the maximum number of graph walk steps to be performed. If less than 0
        then it is unconstrained
    """
    def __init__(self, mg_input, morphable_graph, max_step=-1):
        self.morphable_graph = morphable_graph
        self.max_step = max_step
        self.elementary_action_list = mg_input["elementaryActions"]
        self.keyframe_annotations = extract_keyframe_annotations(self.elementary_action_list)
        self.start_pose = mg_input["startPose"]
        self._transform_from_left_to_right_handed_cs()
        self.action_index = 0
        self.n_actions = len(self.elementary_action_list)

            
    
    def _transform_from_left_to_right_handed_cs(self):
        """ Transform transition and rotation of the start pose from CAD to Opengl 
            coordinate system.
        """
        start_pose_copy = copy.copy(self.start_pose)
        self.start_pose["orientation"] = transform_point_from_cad_to_opengl_cs(start_pose_copy["orientation"])
        self.start_pose["position"] = transform_point_from_cad_to_opengl_cs(start_pose_copy["position"])
        
        
    def get_next_elementary_action_constraints(self):
        
        if self.action_index < self.n_actions:
            action_constraints = ElementaryActionConstraints(self.action_index, self)
            self.action_index+=1
            return action_constraints
        else:
            return None
  