# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 12:00:15 2015

@author: erhe01
"""

import numpy as np
from splines.parameterized_spline import ParameterizedSpline
TRAJECTORY_DIM = 3 # spline in cartesian space

from elementary_action_constraints import ElementaryActionConstraints
        
class ElementaryActionConstraintsBuilder():
    """Implements functions used for the processing of the constraints from the input file
    generated by CNL processing.
    
    Parameters
    ----------
    * mg_input : json data read from a file
        Contains elementary action list with constraints, start pose and keyframe annotations.
    * max_step : integer
        Sets the maximum number of graph walk steps to be performed. If less than 0
        then it is unconstrained
    """
    def __init__(self, mg_input, morphable_graph):
        self.mg_input = mg_input
        self.morphable_graph = morphable_graph
        self.elementary_action_list = mg_input["elementaryActions"]
        self.keyframe_annotations = self._extract_keyframe_annotations(self.elementary_action_list)
        self.start_pose = {}
        self.start_pose["orientation"] = self._transform_point_from_cad_to_opengl_cs(mg_input["startPose"]["orientation"])
        self.start_pose["position"] = self._transform_point_from_cad_to_opengl_cs(mg_input["startPose"]["position"])
        self.n_actions = len(self.elementary_action_list)
        self.reset_counter()
   
    def reset_counter(self):
        self.current_action_index = 0

    def _extract_keyframe_annotations(self, elementary_action_list):
        """
        Returns
        ------
        * keyframe_annotations : a list of dicts
          Contains for every elementary action a dict that associates of events/actions with certain keyframes
        """
        keyframe_annotations = []
        for entry in elementary_action_list:
            print  "entry#################",entry
            if "keyframeAnnotations" in entry.keys():
                annotations = {}
           
                for annotation in entry["keyframeAnnotations"]:
                    key = annotation["keyframe"]
                    annotations[key] = annotation
                keyframe_annotations.append(annotations)
            else:
                keyframe_annotations.append({})
        return keyframe_annotations


    def get_next_elementary_action_constraints(self):
        """
        Returns:
        --------
        * action_constraints : ElementarActionConstraints
          Constraints for the next elementary action extracted from an input file.
        """
        if self.current_action_index < self.n_actions:
            action_constraints = self._build()
            self.current_action_index+=1
            return action_constraints
        else:
            return None
    
      
        
    def _build(self):
        if self.current_action_index < len(self.elementary_action_list):
            action_constraints = ElementaryActionConstraints()
            action_constraints.parent_constraint = self
            action_constraints.action_name = self.elementary_action_list[self.current_action_index]["action"]
            action_constraints.keyframe_annotations = self.keyframe_annotations[self.current_action_index]
            action_constraints.constraints = self.elementary_action_list[self.current_action_index]["constraints"]
            action_constraints.start_pose = self.start_pose
            morphable_subgraph = self.morphable_graph.subgraphs[action_constraints.action_name]
            root_joint_name = self.morphable_graph.skeleton.root# currently only trajectories on the Hips joint are supported
            
            action_constraints.trajectory, action_constraints.unconstrained_indices = self._extract_trajectory_from_constraint_list(action_constraints.constraints, root_joint_name)
        
            keyframe_constraints = self._extract_all_keyframe_constraints(action_constraints.constraints,
                                                                    morphable_subgraph)
            action_constraints.keyframe_constraints = self._reorder_keyframe_constraints_for_motion_primitves(morphable_subgraph,
                                                                                     keyframe_constraints)
            action_constraints._initialized = True
            return action_constraints
        

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
        trajectory_constraint = self._extract_trajectory_constraint(constraint_list,joint_name)
        if  trajectory_constraint is not None:
            #print "found trajectory constraint"
            return self._create_trajectory_from_constraint(trajectory_constraint)
        else:
            return None, None



    def _extract_trajectory_constraint(self, constraints, joint_name="Hips"):
        """Returns a single trajectory constraint definition for joint joint out of a elementary action constraint list
        """
        for c in constraints:
            if joint_name == c["joint"]: 
                if "trajectoryConstraints" in c.keys():
                    return c["trajectoryConstraints"]
        return None
    
    
    def _create_trajectory_from_constraint(self, trajectory_constraint,scale_factor=1.0):
        """ Create a spline based on a trajectory constraint. 
            Components containing None are set to 0, but marked as ignored in the unconstrained_indices list.
            Note all elements in constraints_list must have the same dimensions constrained and unconstrained.
        
        Parameters
        ----------
        * trajectory_constraint: list of dict
        \t Defines a list of control points
        * scale_factor: float
        \t Scaling applied on the control points
        
        Returns
        -------
        * trajectory: ParameterizedSpline
        \t The trajectory defined by the control points from the trajectory_constraint
        * unconstrained_indices : list
        \t List of indices of unconstrained dimensions
        """
       
        
        assert len(trajectory_constraint)>0  and "position" in trajectory_constraint[0].keys()
               
        #extract unconstrained dimensions
        unconstrained_indices = []
        idx = 0
        for v in trajectory_constraint[0]["position"]:
            if v == None:
                unconstrained_indices.append(idx)
            idx += 1            
        unconstrained_indices = self._transform_unconstrained_indices_from_cad_to_opengl_cs(unconstrained_indices)
        #create control points by setting constrained dimensions to 0
        control_points = []
    
        for c in trajectory_constraint:
            point = [ p*scale_factor if p is not None else 0 for p in c["position"] ]# else 0  where the array is None set it to 0
            point = self._transform_point_from_cad_to_opengl_cs(point)
            control_points.append(point)
        #print "####################################################"
        #print "control points are: "
        #print control_points
        trajectory =  ParameterizedSpline(control_points, TRAJECTORY_DIM)
        
        return trajectory, unconstrained_indices
    
    
        
    def _extract_keyframe_constraint(self, joint_name,constraint,morphable_subgraph,time_information):
        """ Creates a dict containing all properties stated explicitly or implicitly in the input constraint
        Parameters
        ----------
        * joint_name : string
          Name of the joint
        * constraint : dict
          Read from json input file 
        * time_information : string
          Time information corresponding to an annotation read from morphable graph meta information 
          
         Returns
         -------
         *constraint_desc : dict
          Contains the keys joint, position, orientation, semanticAnnotation
        """
       
        position = [None, None, None]       
        orientation = [None, None, None]
        first_frame = None
        last_frame = None
        if "position" in constraint.keys():
             position = constraint["position"]
        if "orientation" in constraint.keys():
            orientation =constraint["orientation"]
        #check if last or fist frame from annotation
    
        position = self._transform_point_from_cad_to_opengl_cs(position)
     
        if time_information == "lastFrame":
            last_frame = True
        elif time_information == "firstFrame":
            first_frame = True
        if "semanticAnnotation" in constraint.keys():
            semanticAnnotation = constraint["semanticAnnotation"]
        else:
            semanticAnnotation = {}
            
        semanticAnnotation["firstFrame"] = first_frame
        semanticAnnotation["lastFrame"] = last_frame
        constraint_desc = {"joint":joint_name,"position":position,"orientation":orientation,"semanticAnnotation": semanticAnnotation}
        return constraint_desc
        
        
    def _constraint_definition_has_label(self, constraint_definition, label):
        """ Checks if the label is in the semantic annotation dict of a constraint
        """
        #print "check######",constraint_definition
        if "semanticAnnotation" in constraint_definition.keys():
            annotation = constraint_definition["semanticAnnotation"]
            #print "semantic Annotation",annotation
            if label in annotation.keys():
                return True
        return False
            
    def _extract_keyframe_constraints_for_label(self, constraint_list, label):
        """ Returns the constraints associated with the given label. Ordered 
            based on joint names.
        Returns
        ------
        * key_constraints : dict of lists
        \t contains the list of the constrained joints
        """
        key_constraints= {}
        for c in constraint_list:
            joint_name = c["joint"]
            #print "read constraint"
            #print c
            if "keyframeConstraints" in c.keys():# there are keyframe constraints
                key_constraints[joint_name] = []
                for constraint_definition in c["keyframeConstraints"]:
                    print "read constraint",constraint_definition
                    if self._constraint_definition_has_label(constraint_definition,label):
                        key_constraints[joint_name].append(constraint_definition)
        return key_constraints
                    
    def _extract_all_keyframe_constraints(self, constraint_list,morphable_subgraph):
        """Orders the keyframe constraint for the labels found in the metainformation of
           the elementary actions based on labels as keys
        Returns
        -------
        * keyframe_constraints : dict of dict of lists
          Lists of constraints for each motion primitive in the subgraph.
          access as keyframe_constraints["label"]["joint"][index]
        """
        keyframe_constraints = {}
        annotations = morphable_subgraph.annotation_map.keys()#["start_contact",]
        for label in annotations:
    #        print "extract constraints for annotation",label
            keyframe_constraints[label] = self._extract_keyframe_constraints_for_label(constraint_list,label)
            #key_frame_constraints = extract_keyframe_constraints(constraints,annotion)
        return keyframe_constraints
    
    


    def _reorder_keyframe_constraints_for_motion_primitves(self, morphable_subgraph, keyframe_constraints):
         """ Order constraints extracted by _extract_all_keyframe_constraints for each state
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
                    constraint_desc = self._extract_keyframe_constraint(joint_name,c,\
                                                morphable_subgraph,time_information)
                    constraints[state].append(constraint_desc)
         return constraints
         

    def _transform_point_from_cad_to_opengl_cs(self, point):
        """ Transforms a 3d point represented as a list from a left handed cad to a
            right handed opengl coordinate system
        """
    
        transform_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        return np.dot(transform_matrix, point).tolist()
    
    
    def _transform_unconstrained_indices_from_cad_to_opengl_cs(self, indices):
        """ Transforms a list indicating unconstrained dimensions from cad to opengl
            coordinate system.
        """
        new_indices = []
        for i in indices:
            if i == 0:
                new_indices.append(0)
            elif i == 1:
                new_indices.append(2)
            elif i == 2:
                new_indices.append(1)
        return new_indices
