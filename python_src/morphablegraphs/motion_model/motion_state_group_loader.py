# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:57:42 2015

@author: erhe01
"""

import os
from ..utilities.io_helper_functions import load_json_file
from motion_state import MotionState
from motion_state_group import MotionStateGroup
from . import META_INFORMATION_FILE_NAME


class MotionStateGroupLoader(object):
    """ Creates an instance of a MotionStateGroup from a data source
    """
    def __init__(self):
        self.elementary_action_name = None
        self.elementary_action_directory = None
        self.has_transition_models = False
        self.meta_information = None
        self.start_states = []
        self.end_states = []
        self.motion_primitive_annotations = {}
        self.build_from_directory = False
        self.load_transition_models = False
        self.elementary_action_data = None

    def set_properties(self, load_transition_models=False):
        self.load_transition_models = load_transition_models
        return

    def set_data_from_zip(self, elementary_action_data):
        """Sets the directory with the motion primitives
        Parameters
        ----------
        * elementary_action_data: dict
            Description loaded from zip file
         """
        self.elementary_action_name = elementary_action_data["name"]
        self.elementary_action_data = elementary_action_data
        self.build_from_directory = False

    def set_directory_as_data_source(self, elementary_action_name, elementary_action_directory):
        """Sets the directory with the motion primitives
        Parameters
        ----------
        * elementary_action_name: string
            The name of the elementary action
        * elementary_action_directory: string
            The directory of the morphable models of an elementary action.
         """
        self.elementary_action_name = elementary_action_name
        self.elementary_action_directory = elementary_action_directory
        self.build_from_directory = True

    def build(self):
        if self.build_from_directory:
            motion_primitive_node_group = self._init_from_directory()
        else:
            motion_primitive_node_group = self._init_from_dict()
        return motion_primitive_node_group
        
    def _init_from_dict(self):
        motion_primitive_node_group = MotionStateGroup(self.elementary_action_data["name"], None)
        for motion_primitive_name in self.elementary_action_data["nodes"].keys():
            node_key = (self.elementary_action_data["name"], motion_primitive_name)
            motion_primitive_node_group.nodes[node_key] = MotionState()
            motion_primitive_node_group.nodes[node_key].init_from_dict(self.elementary_action_name, self.elementary_action_data["nodes"][motion_primitive_name])
        if "info" in self.elementary_action_data.keys():
            motion_primitive_node_group.set_meta_information(self.elementary_action_data["info"])
        else:
            motion_primitive_node_group.set_meta_information()
        return motion_primitive_node_group

    def _init_from_directory(self):
        motion_primitive_node_group = MotionStateGroup(self.elementary_action_name, self.elementary_action_directory)
        #load morphable models
        temp_file_list = []#for files containing additional information that require the full graph to be constructed first
        meta_information = None
        motion_primitive_node_group.label_to_motion_primitive_map = {}
        for root, dirs, files in os.walk(self.elementary_action_directory):
            for file_name in files:
                if file_name == META_INFORMATION_FILE_NAME:
                    print "found meta information for", self.elementary_action_name
                    meta_information = load_json_file(self.elementary_action_directory+os.sep+file_name)
                    
                elif file_name.endswith("mm.json"):
                    print "found motion primitive", file_name
                    motion_primitive_name = file_name.split("_")[1]
                    motion_primitive_file_name = self.elementary_action_directory+os.sep+file_name
                    node_key = (self.elementary_action_name, motion_primitive_name)
                    motion_primitive_node_group.nodes[node_key] = MotionState()
                    motion_primitive_node_group.nodes[node_key].init_from_file(motion_primitive_node_group.elementary_action_name, motion_primitive_name, motion_primitive_file_name)
                elif file_name.endswith(".stats"):
                    print "found stats", file_name
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
