# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:57:42 2015

@author: erhe01
"""

import os
from utilities.io_helper_functions import load_json_file

from motion_primitive_node import MotionPrimitiveNode
from motion_primitive_node_group import MotionPrimitiveNodeGroup

 
class MotionPrimitiveNodeGroupBuilder(object):
    """ Contains a dictionary of motion primitives of an elementary action as nodes  meta information. 

    """
    def __init__(self):
        self.elementary_action_name = None
        self.morphable_model_directory = None
        self.has_transition_models = False
        self.meta_information = None
        self.annotation_map = {}
        self.start_states = []
        self.end_states = []
        self.motion_primitive_annotations = {}
        self.build_from_directory = False
        self.load_transition_models = False
        self.transition_model_directory = None

    def set_properties(self, transition_model_directory=None, load_transition_models=False):
        self.load_transition_models = load_transition_models
        self.transition_model_directory = transition_model_directory
        return

    def set_data_source(self, elementary_action_name, morphable_model_directory, subgraph_desc=None, graph_definition=None):
        """Sets the directory with the motion primitives    
        Parameters
        ----------
        * elementary_action_name: string
            The name of the elementary action that the subgraph represents
        
        * morphable_model_directory: string
            The directory of the morphable models of an elementary action.
        
        * subgraph_desc: dict
            Optionla: Description loaded from zip file
            
        * transition_model_directory: string
            Optional: The directory of the transition models.
        
        * graph_definition: dict
           Optional: Description of the graph
         """
        self.subgraph_desc = subgraph_desc
        self.elementary_action_name = elementary_action_name
        self.morphable_model_directory = morphable_model_directory
        self.graph_definition = graph_definition
        if self.subgraph_desc is not None:
            self.build_from_directory = False
        else:
            self.build_from_directory = True

    def build(self):
        if self.build_from_directory:
            motion_primitive_node_group = self._init_from_directory()
        else:
            motion_primitive_node_group = self._init_from_dict()
        return motion_primitive_node_group
        
    def _init_from_dict(self):
        motion_primitive_node_group = MotionPrimitiveNodeGroup()
        motion_primitive_node_group.nodes = {}
        motion_primitive_node_group.loaded_from_dict = True
        motion_primitive_node_group.elementary_action_name = self.subgraph_desc["name"]

        for motion_primitive_name in self.subgraph_desc["nodes"].keys():
             node_key = (self.subgraph_desc["name"], motion_primitive_name)
             motion_primitive_node_group.nodes[node_key] = MotionPrimitiveNode()
             motion_primitive_node_group.nodes[node_key].init_from_dict(self.elementary_action_name, self.subgraph_desc["nodes"][motion_primitive_name])
        
        if "info" in self.subgraph_desc.keys():
            motion_primitive_node_group._set_meta_information(self.subgraph_desc["info"])
        else:
            motion_primitive_node_group._set_meta_information() 
        return motion_primitive_node_group

    def _init_from_directory(self):
        
        motion_primitive_node_group = MotionPrimitiveNodeGroup()
        motion_primitive_node_group.nodes = {}
        motion_primitive_node_group.loaded_from_dict = False
        motion_primitive_node_group.elementary_action_name = self.elementary_action_name
        motion_primitive_node_group.morphable_model_directory = self.morphable_model_directory
   
        #load morphable models
        temp_file_list =  []#for files containing additional information that require the full graph to be constructed first
        meta_information = None
        motion_primitive_node_group.annotation_map = {}
        for root, dirs, files in os.walk(self.morphable_model_directory):
            for file_name in files:#for each morphable model 
                if file_name == "meta_information.json":
                    print "found meta information for",  self.elementary_action_name
                    meta_information = load_json_file(self.morphable_model_directory+os.sep+file_name)
                    
                elif file_name.endswith("mm.json"):
                    print "found motion primitive",file_name  
                    motion_primitive_name = file_name.split("_")[1]
                    #print motion_primitve_file_name
                    motion_primitive_file_name = self.morphable_model_directory+os.sep+file_name
                    node_key = (self.elementary_action_name, motion_primitive_name)
                    motion_primitive_node_group.nodes[node_key] = MotionPrimitiveNode()
                    motion_primitive_node_group.nodes[node_key].init_from_file(motion_primitive_node_group.elementary_action_name, motion_primitive_name, motion_primitive_file_name)
                    
                elif file_name.endswith(".stats"):
                    print "found stats",file_name
                    temp_file_list.append(file_name)

                else:
                    print "ignored", file_name

        motion_primitive_node_group._set_meta_information(meta_information)
        
        #load information about training data if available
        for file_name in temp_file_list:
            motion_primitive = file_name.split("_")[1][:-6]
            if motion_primitive in motion_primitive_node_group.nodes.keys():
                info = load_json_file(self.morphable_model_directory+os.sep+file_name, use_ordered_dict=True)
                motion_primitive_node_group.nodes[motion_primitive].parameter_bb = info["pose_bb"]
                motion_primitive_node_group.nodes[motion_primitive].cartesian_bb = info["cartesian_bb"]
                motion_primitive_node_group.nodes[motion_primitive].velocity_data = info["pose_velocity"]

        return  motion_primitive_node_group

