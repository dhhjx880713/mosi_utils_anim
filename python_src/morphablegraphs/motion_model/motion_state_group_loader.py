# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:57:42 2015

@author: erhe01
"""

import os
from ..utilities.io_helper_functions import load_json_file
from .motion_state import MotionState
from .motion_state_group import MotionStateGroup
from . import META_INFORMATION_FILE_NAME
from ..utilities import write_log


class MotionStateGroupLoader(object):
    """ Creates an instance of a MotionStateGroup from a data source
    """
    def __init__(self):
        self.has_transition_models = False
        self.meta_information = None
        self.start_states = list()
        self.end_states = list()
        self.labeled_frames = dict()
        self.load_transition_models = False
        self.motion_state_graph = None

    def set_properties(self, load_transition_models=False):
        self.load_transition_models = load_transition_models

    def build_from_dict(self, ea_data, graph):
        mp_node_group = MotionStateGroup(ea_data["name"], None, graph)

        for mp_name in list(ea_data["nodes"].keys()):
            node_key = (ea_data["name"], mp_name)
            mp_node_group.nodes[node_key] = MotionState(mp_node_group)
            mp_node_group.nodes[node_key].init_from_dict(ea_data["name"], ea_data["nodes"][mp_name])

        if "info" in list(ea_data.keys()):
            mp_node_group.set_meta_information(ea_data["info"])
        else:
            mp_node_group.set_meta_information()

        for mp_name in ea_data["nodes"]:
            if "keyframes" in ea_data["nodes"][mp_name]["mm"]:
                keyframes = ea_data["nodes"][mp_name]["mm"]["keyframes"]
                for label, frame_idx in keyframes.items():
                    if label not in mp_node_group.label_to_motion_primitive_map:
                        mp_node_group.label_to_motion_primitive_map[label] = list()
                    mp_node_group.label_to_motion_primitive_map[label].append(mp_name)
                if mp_name not in mp_node_group.labeled_frames:
                    mp_node_group.labeled_frames[mp_name] = dict()
                mp_node_group.labeled_frames[mp_name].update(keyframes)

        return mp_node_group

    def build_from_directory(self, ea_name, ea_directory, graph):
        mp_node_group = MotionStateGroup(ea_name, ea_directory, graph)
        temp_file_list = []#for files containing additional information that require the full graph to be constructed first
        meta_information = None
        mp_node_group.label_to_motion_primitive_map = {}
        for root, dirs, files in os.walk(ea_directory):
            for file_name in files:
                if file_name == META_INFORMATION_FILE_NAME:
                    write_log("found meta information for", ea_name)
                    meta_information = load_json_file(ea_directory+os.sep+file_name)
                elif file_name.endswith("mm.json"):
                    write_log("found motion primitive", file_name)
                    mp_name = file_name.split("_")[1]
                    mp_file_name = ea_directory+os.sep+file_name
                    node_key = (ea_name, mp_name)
                    mp_node_group.nodes[node_key] = MotionState(mp_node_group)
                    mp_node_group.nodes[node_key].init_from_file(mp_node_group.ea_name, mp_name, mp_file_name)
                elif file_name.endswith(".stats"):
                    write_log("found stats", file_name)
                    temp_file_list.append(file_name)
                else:
                    write_log("ignored", file_name)
            mp_node_group.set_meta_information(meta_information)
        #load information about training data if available
        for file_name in temp_file_list:
            motion_primitive = file_name.split("_")[1][:-6]
            if motion_primitive in list(mp_node_group.nodes.keys()):
                info = load_json_file(ea_directory+os.sep+file_name, use_ordered_dict=True)
                mp_node_group.nodes[motion_primitive].parameter_bb = info["pose_bb"]
                mp_node_group.nodes[motion_primitive].cartesian_bb = info["cartesian_bb"]
                mp_node_group.nodes[motion_primitive].velocity_data = info["pose_velocity"]
        return mp_node_group
