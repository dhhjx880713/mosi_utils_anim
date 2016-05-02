# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:57:51 2015

@author: erhe01
"""

import os
from ..animation_data.bvh import BVHReader
from ..animation_data.skeleton import Skeleton
from ..animation_data.motion_editing import euler_to_quaternion
from ..utilities.io_helper_functions import load_json_file
from gp_mixture import GPMixture
from motion_state_group_loader import MotionStateGroupLoader
from ..utilities.zip_io import ZipReader
from motion_state_transition import MotionStateTransition
from motion_state_graph import MotionStateGraph
from ..motion_generator.hand_pose_generator import HandPoseGenerator
from . import ELEMENTARY_ACTION_DIRECTORY_NAME, TRANSITION_MODEL_DIRECTORY_NAME, NODE_TYPE_START, NODE_TYPE_STANDARD, NODE_TYPE_END, TRANSITION_DEFINITION_FILE_NAME, TRANSITION_MODEL_FILE_ENDING

SKELETON_FILE = "skeleton.bvh"  # TODO replace with standard skeleton in data directory



class MotionStateGraphLoader(object):
    """   Constructs a MotionPrimitiveGraph instance from a zip file or directory as data source
    """  
    def __init__(self):
        self.graph_data = None # used to store the zip file content
        self.load_transition_models = False
        self.update_stats = False
        self.motion_state_graph_path = None
        self.elementary_action_directory = None
        self.motion_primitive_node_group_builder = MotionStateGroupLoader()
        self.animated_joints = ["Hips", "Spine", "Spine_1", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"]


    def set_data_source(self, motion_state_graph_path, load_transition_models=False, update_stats=False):
        """ Set the source which is used to load the data structure into memory.
        Parameters
        ----------
        * elementary_action_directory: string
        \tThe root directory of the morphable models of all elementary actions.
        * transition_model_directory: string
        \tThe directory of the morphable models of an elementary action.
        * transition_model_directory: string
        \tThe directory of the transition models.
        """

        self.load_transition_models = load_transition_models
        self.update_stats = update_stats
        self.motion_state_graph_path = motion_state_graph_path
        self.motion_primitive_node_group_builder.set_properties(self.load_transition_models)

    def build(self):
        motion_state_graph = MotionStateGraph()

        if os.path.isfile(self.motion_state_graph_path+".zip"):
            self._init_from_zip_file(motion_state_graph)
        else:
            self._init_from_directory(motion_state_graph)

        return motion_state_graph

    def _init_from_zip_file(self, motion_state_graph):
        zip_path = self.motion_state_graph_path+".zip"
        zip_reader = ZipReader(zip_path, pickle_objects=True)
        graph_data = zip_reader.get_graph_data()
        motion_state_graph.skeleton = Skeleton(BVHReader("").init_from_string(graph_data["skeletonString"]), self.animated_joints)
        #motion_state_graph.skeleton = motion_state_graph.full_skeleton.create_reduced_copy()
        motion_state_graph.mgrd_skeleton = motion_state_graph.skeleton.convert_to_mgrd_skeleton()

        #skeleton_path = self.motion_state_graph_path + os.sep + SKELETON_FILE
        #motion_state_graph.skeleton = Skeleton(BVHReader(skeleton_path))
        transition_dict = graph_data["transitions"]
        elementary_actions = graph_data["subgraphs"]
        for action_name in elementary_actions.keys():
            self.motion_primitive_node_group_builder.set_data_from_zip(elementary_actions[action_name])
            node_group = self.motion_primitive_node_group_builder.build(motion_state_graph)
            motion_state_graph.nodes.update(node_group.nodes)
            motion_state_graph.node_groups[node_group.elementary_action_name] = node_group
        #print "add transitions between nodes from", transition_dict
        self._set_transitions_from_dict(motion_state_graph, transition_dict)

        self._update_motion_state_stats(motion_state_graph, recalculate=False)

        if "handPoseInfo" in graph_data.keys():
            motion_state_graph.hand_pose_generator = HandPoseGenerator(motion_state_graph.skeleton)
            motion_state_graph.hand_pose_generator.init_from_desc(graph_data["handPoseInfo"])

    def _init_from_directory(self, motion_state_graph, recalculate_motion_stats=True):
        """ Initializes the class
        """
        skeleton_path = self.motion_state_graph_path + os.sep + SKELETON_FILE
        motion_state_graph.skeleton = Skeleton(BVHReader(skeleton_path), self.animated_joints)
        #motion_state_graph.skeleton = motion_state_graph.full_skeleton.create_reduced_copy()
        motion_state_graph.mgrd_skeleton = motion_state_graph.skeleton.convert_to_mgrd_skeleton()

        #load graphs representing elementary actions including transitions between actions
        for key in next(os.walk(self.motion_state_graph_path + os.sep + ELEMENTARY_ACTION_DIRECTORY_NAME))[1]:
            subgraph_path = self. motion_state_graph_path + os.sep + ELEMENTARY_ACTION_DIRECTORY_NAME + os.sep + key
            print subgraph_path
            name = key.split("_")[-1]
            self.motion_primitive_node_group_builder.set_directory_as_data_source(name, subgraph_path)
            node_group = self.motion_primitive_node_group_builder.build(motion_state_graph)
            motion_state_graph.nodes.update(node_group.nodes)
            motion_state_graph.node_groups[node_group.elementary_action_name] = node_group

        graph_definition_file = self.motion_state_graph_path+os.sep+TRANSITION_DEFINITION_FILE_NAME
        #add transitions between subgraphs and load transition models
        if os.path.isfile(graph_definition_file):
            graph_definition = load_json_file(graph_definition_file)
            if "transitions" in graph_definition.keys():
                print "add transitions between subgraphs from", graph_definition_file
                self._set_transitions_from_dict(motion_state_graph, graph_definition["transitions"])
        else:
            print "did not find graph definition file", graph_definition_file

        self._update_motion_state_stats(motion_state_graph, recalculate=recalculate_motion_stats)

    def _update_motion_state_stats(self, motion_state_graph, recalculate=False):
        for keys in motion_state_graph.node_groups.keys():
            motion_state_graph.node_groups[keys]._update_motion_state_stats(recalculate=recalculate)

    def _set_transitions_from_dict(self, motion_state_graph, transition_dict):
        for node_key in transition_dict:
            from_action_name = node_key.split("_")[0]
            from_motion_primitive_name = node_key.split("_")[1]
            from_node_key = (from_action_name, from_motion_primitive_name)
            if from_node_key in motion_state_graph.nodes.keys():
                #print "add action transitions for", subgraph_key,"###############################"
                for to_key in transition_dict[node_key]:
                    to_action_name = to_key.split("_")[0]
                    to_motion_primitive_name = to_key.split("_")[1]
                    to_node_key = (to_action_name, to_motion_primitive_name)
                    if to_node_key in motion_state_graph.nodes.keys():
                        self._add_transition(motion_state_graph, from_node_key, to_node_key)

    def _load_transition_model(self, motion_state_graph, from_node_key, to_node_key):
        transition_model_file = self.motion_state_graph_path + os.sep + TRANSITION_MODEL_DIRECTORY_NAME\
        + os.sep + from_node_key + "_to_" + to_node_key[0] + "_" + to_node_key[1] + TRANSITION_MODEL_FILE_ENDING
        if os.path.isfile(transition_model_file):
            output_gmm = motion_state_graph.nodes[to_node_key].gaussian_mixture_model
            return GPMixture.load(transition_model_file, motion_state_graph.nodes[from_node_key].gaussian_mixture_model,output_gmm)
        else:
            print "did not find transition model file", transition_model_file
            return None

    def _get_transition_type(self, motion_state_graph, from_node_key, to_node_key):
        if to_node_key[0] == from_node_key[0]:
            if motion_state_graph.nodes[to_node_key].node_type in [NODE_TYPE_START, NODE_TYPE_STANDARD]:
                transition_type = "standard"
            else:
                transition_type = "end"
        else:
            transition_type = "action_transition"
        return transition_type

    def _add_transition(self, motion_state_graph, from_node_key, to_node_key):
        transition_model = None
        if self.load_transition_models:
            transition_model = self._load_transition_model(motion_state_graph, from_node_key, to_node_key)
        transition_type = self._get_transition_type(motion_state_graph, from_node_key, to_node_key)
        #print "create edge", from_node_key, to_node_key
        motion_state_graph.nodes[from_node_key].outgoing_edges[to_node_key] = MotionStateTransition(from_node_key, to_node_key, transition_type, transition_model)
