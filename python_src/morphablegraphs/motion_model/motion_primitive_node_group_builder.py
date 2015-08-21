# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:57:42 2015

@author: erhe01
"""

import os
from ..utilities.io_helper_functions import load_json_file

from motion_primitive_node import MotionPrimitiveNode
from motion_primitive_node_group import MotionPrimitiveNodeGroup

 
class MotionPrimitiveNodeGroupBuilder(object):
    """ Contains a dictionary of motion primitives of an elementary action as nodes  meta information.
    """
    def __init__(self):
        self.elementary_action_name = None
        self.elementary_action_directory = None
        self.has_transition_models = False
        self.meta_information = None
        self.annotation_map = {}
        self.start_states = []
        self.end_states = []
        self.motion_primitive_annotations = {}
        self.build_from_directory = False
        self.load_transition_models = False
        self.transition_model_directory = None
        self.graph_definition = None

    def set_properties(self, transition_model_directory=None, load_transition_models=False):
        self.load_transition_models = load_transition_models
        self.transition_model_directory = transition_model_directory
        return

    def set_data_source(self, elementary_action_name, elementary_action_directory, elementary_action_dict=None, graph_definition=None):
        """Sets the directory with the motion primitives    
        Parameters
        ----------
        * elementary_action_name: string
            The name of the elementary action
        * elementary_action_directory: string
            The directory of the morphable models of an elementary action.
        * elementary_action_dict: dict
            Optionl: Description loaded from zip file
        * transition_model_directory: string
            Optional: The directory of the transition models.
        * graph_definition: dict
           Optional: Description of the graph
         """
        self.elementary_action_dict = elementary_action_dict
        self.elementary_action_name = elementary_action_name
        self.elementary_action_directory = elementary_action_directory
        self.graph_definition = graph_definition
        if self.elementary_action_dict is not None:
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
        motion_primitive_node_group.elementary_action_name = self.elementary_action_dict["name"]
        for motion_primitive_name in self.elementary_action_dict["nodes"].keys():
            node_key = (self.elementary_action_dict["name"], motion_primitive_name)
            motion_primitive_node_group.nodes[node_key] = MotionPrimitiveNode()
            motion_primitive_node_group.nodes[node_key].init_from_dict(self.elementary_action_name, self.elementary_action_dict["nodes"][motion_primitive_name])
        if "info" in self.elementary_action_dict.keys():
            motion_primitive_node_group.set_meta_information(self.elementary_action_dict["info"])
        else:
            motion_primitive_node_group.set_meta_information()
        return motion_primitive_node_group

    def _init_from_directory(self):
        motion_primitive_node_group = MotionPrimitiveNodeGroup()
        motion_primitive_node_group.nodes = {}
        motion_primitive_node_group.loaded_from_dict = False
        motion_primitive_node_group.elementary_action_name = self.elementary_action_name
        motion_primitive_node_group.elementary_action_directory = self.elementary_action_directory
        #load morphable models
        temp_file_list =  []#for files containing additional information that require the full graph to be constructed first
        meta_information = None
        motion_primitive_node_group.annotation_map = {}
        for root, dirs, files in os.walk(self.elementary_action_directory):
            for file_name in files:#for each morphable model 
                if file_name == "meta_information.json":
                    print "found meta information for",  self.elementary_action_name
                    meta_information = load_json_file(self.elementary_action_directory+os.sep+file_name)
                    
                elif file_name.endswith("mm.json"):
                    print "found motion primitive", file_name
                    motion_primitive_name = file_name.split("_")[1]
                    #print motion_primitve_file_name
                    motion_primitive_file_name = self.elementary_action_directory+os.sep+file_name
                    node_key = (self.elementary_action_name, motion_primitive_name)
                    motion_primitive_node_group.nodes[node_key] = MotionPrimitiveNode()
                    motion_primitive_node_group.nodes[node_key].init_from_file(motion_primitive_node_group.elementary_action_name, motion_primitive_name, motion_primitive_file_name)
                    
                elif file_name.endswith(".stats"):
                    print "found stats",file_name
                    temp_file_list.append(file_name)

                else:
                    print "ignored", file_name
        motion_primitive_node_group.set_meta_information(meta_information)

        #load information about training data if available
        for file_name in temp_file_list:
            motion_primitive = file_name.split("_")[1][:-6]
            if motion_primitive in motion_primitive_node_group.nodes.keys():
                info = load_json_file(self.elementary_action_directory+os.sep+file_name, use_ordered_dict=True)
                motion_primitive_node_group.nodes[motion_primitive].parameter_bb = info["pose_bb"]
                motion_primitive_node_group.nodes[motion_primitive].cartesian_bb = info["cartesian_bb"]
                motion_primitive_node_group.nodes[motion_primitive].velocity_data = info["pose_velocity"]
        return motion_primitive_node_group


