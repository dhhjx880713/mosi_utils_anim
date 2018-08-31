# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:57:51 2015

@author: erhe01
"""

import os
from ..animation_data.bvh import BVHReader
from ..animation_data.skeleton_builder import SkeletonBuilder
from ..animation_data.utils import euler_to_quaternion
from ..utilities.io_helper_functions import load_json_file
from .gp_mixture import GPMixture
from .motion_state_group_loader import MotionStateGroupLoader
from ..utilities.zip_io import ZipReader, SKELETON_BVH_STRING_KEY, SKELETON_JSON_KEY
from .motion_state_transition import MotionStateTransition
from .motion_state_graph import MotionStateGraph
from ..motion_generator.hand_pose_generator import HandPoseGenerator
from . import ELEMENTARY_ACTION_DIRECTORY_NAME, TRANSITION_MODEL_DIRECTORY_NAME, NODE_TYPE_START, NODE_TYPE_STANDARD,NODE_TYPE_CYCLE_END, NODE_TYPE_END, TRANSITION_DEFINITION_FILE_NAME, TRANSITION_MODEL_FILE_ENDING, NODE_TYPE_IDLE
from ..utilities import write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_ERROR, LOG_MODE_INFO

SKELETON_FILE = "skeleton"  # TODO replace with standard skeleton in data directory


class MotionStateGraphLoader(object):
    """   Constructs a MotionPrimitiveGraph instance from a zip file or directory as data source
    """  
    def __init__(self):
        self.graph_data = None  # used to store the zip file content
        self.load_transition_models = False
        self.update_stats = False
        self.motion_state_graph_path = None
        self.ea_directory = None
        self.use_all_joints = False
        self.mp_node_group_builder = MotionStateGroupLoader()

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
        self.mp_node_group_builder.set_properties(self.load_transition_models)

    def build(self):
        graph = MotionStateGraph()
        if os.path.isfile(self.motion_state_graph_path+".zip"):
            self._build_from_zip_file(graph)
        else:
            self._build_from_directory(graph)
        return graph

    def _build_from_zip_file(self, ms_graph):
        zip_path = self.motion_state_graph_path+".zip"
        zip_reader = ZipReader(zip_path, pickle_objects=True)
        graph_data = zip_reader.get_graph_data()
        if SKELETON_BVH_STRING_KEY in list(graph_data.keys()):
            bvh_reader = BVHReader("").init_from_string(graph_data[SKELETON_BVH_STRING_KEY])
            ms_graph.skeleton = SkeletonBuilder().load_from_bvh(bvh_reader)
        elif SKELETON_JSON_KEY in list(graph_data.keys()):
            if self.use_all_joints and "animated_joints" in graph_data[SKELETON_JSON_KEY]:
                del graph_data[SKELETON_JSON_KEY]["animated_joints"]
            ms_graph.skeleton = SkeletonBuilder().load_from_json_data(graph_data[SKELETON_JSON_KEY])
        else:
            raise Exception("There is no skeleton defined in the graph file")
            return

        ms_graph.animated_joints = ms_graph.skeleton.animated_joints
        ms_graph.mgrd_skeleton = ms_graph.skeleton.convert_to_mgrd_skeleton()

        #skeleton_path = self.motion_state_graph_path + os.sep + SKELETON_FILE
        #motion_state_graph.skeleton = Skeleton(BVHReader(skeleton_path))
        transition_dict = graph_data["transitions"]
        actions = graph_data["subgraphs"]
        for action_name in list(actions.keys()):
            node_group = self.mp_node_group_builder.build_from_dict(actions[action_name], ms_graph)
            ms_graph.nodes.update(node_group.nodes)
            ms_graph.node_groups[node_group.ea_name] = node_group

        self._set_transitions_from_dict(ms_graph, transition_dict)

        self._update_motion_state_stats(ms_graph, recalculate=False)

        if "handPoseInfo" in list(graph_data.keys()):
            ms_graph.hand_pose_generator = HandPoseGenerator(ms_graph.skeleton)
            ms_graph.hand_pose_generator.init_from_desc(graph_data["handPoseInfo"])

    def _build_from_directory(self, ms_graph, recalculate_motion_stats=True):
        """ Initializes the class
        """
        graph_definition_file = self.motion_state_graph_path + os.sep + TRANSITION_DEFINITION_FILE_NAME
        # add transitions between subgraphs and load transition models
        if os.path.isfile(graph_definition_file):
            graph_definition = load_json_file(graph_definition_file)
            skeleton_path = self.motion_state_graph_path + os.sep + SKELETON_FILE
            if SKELETON_JSON_KEY in list(graph_definition.keys()):
                ms_graph.skeleton = SkeletonBuilder().load_from_json_data(graph_definition[SKELETON_JSON_KEY])
            elif os.path.isfile(skeleton_path+"bvh"):
                ms_graph.skeleton = SkeletonBuilder().load_from_bvh(BVHReader(skeleton_path+"bvh"))
            else:
                raise Exception("There is no skeleton defined in the graph directory")
                return

            ms_graph.animated_joints = ms_graph.skeleton.animated_joints
            # ms_graph.skeleton = motion_state_graph.full_skeleton.create_reduced_copy()
            ms_graph.mgrd_skeleton = ms_graph.skeleton.convert_to_mgrd_skeleton()

            #load graphs representing elementary actions including transitions between actions
            for key in next(os.walk(self.motion_state_graph_path + os.sep + ELEMENTARY_ACTION_DIRECTORY_NAME))[1]:
                subgraph_path = self. motion_state_graph_path + os.sep + ELEMENTARY_ACTION_DIRECTORY_NAME + os.sep + key
                name = key.split("_")[-1]
                node_group = self.mp_node_group_builder.build_from_directory(name, subgraph_path, ms_graph)
                ms_graph.nodes.update(node_group.nodes)
                ms_graph.node_groups[node_group.ea_name] = node_group

            if "transitions" in list(graph_definition.keys()):
                write_message_to_log("add transitions between actions from" + graph_definition_file, LOG_MODE_DEBUG)
                self._set_transitions_from_dict(ms_graph, graph_definition["transitions"])
        else:
            write_message_to_log("Error: Did not find graph definition file " + graph_definition_file, LOG_MODE_ERROR)

        self._update_motion_state_stats(ms_graph, recalculate=recalculate_motion_stats)

    def _update_motion_state_stats(self, motion_state_graph, recalculate=False):
        for keys in list(motion_state_graph.node_groups.keys()):
            motion_state_graph.node_groups[keys]._update_motion_state_stats(recalculate=recalculate)

    def _set_transitions_from_dict(self, motion_state_graph, transition_dict):
        for node_key in transition_dict:
            from_action_name = node_key.split("_")[0]
            from_mp_key = node_key.split("_")[1]
            from_node_key = (from_action_name, from_mp_key)
            if from_node_key in motion_state_graph.nodes.keys():
                for to_key in transition_dict[node_key]:
                    to_action_name = to_key.split("_")[0]
                    to_mp_name = to_key.split("_")[1]
                    to_node_key = (to_action_name, to_mp_name)
                    if to_node_key in motion_state_graph.nodes.keys():
                        self._add_transition(motion_state_graph, from_node_key, to_node_key)

    def _load_transition_model(self, motion_state_graph, from_node_key, to_node_key):
        transition_model_file = self.motion_state_graph_path + os.sep + TRANSITION_MODEL_DIRECTORY_NAME\
        + os.sep + from_node_key + "_to_" + to_node_key[0] + "_" + to_node_key[1] + TRANSITION_MODEL_FILE_ENDING
        if os.path.isfile(transition_model_file):
            output_gmm = motion_state_graph.nodes[to_node_key].gaussian_mixture_model
            return GPMixture.load(transition_model_file, motion_state_graph.nodes[from_node_key].gaussian_mixture_model,output_gmm)
        else:
            write_message_to_log("Error: Did not find transition model file " + transition_model_file, LOG_MODE_ERROR)
            return None

    def _get_transition_type(self, graph, from_node_key, to_node_key):
        if to_node_key[0] == from_node_key[0]:
            if graph.nodes[from_node_key].node_type == NODE_TYPE_IDLE:
                if graph.nodes[to_node_key].node_type == NODE_TYPE_START:
                    t_type = NODE_TYPE_START
                elif graph.nodes[to_node_key].node_type == NODE_TYPE_IDLE:
                    t_type = NODE_TYPE_IDLE
                elif graph.nodes[to_node_key].node_type == NODE_TYPE_END:
                    t_type = NODE_TYPE_END
            else:
                if graph.nodes[to_node_key].node_type == NODE_TYPE_STANDARD:
                    t_type = NODE_TYPE_STANDARD
                elif graph.nodes[to_node_key].node_type == NODE_TYPE_START:
                    t_type = NODE_TYPE_START
                elif graph.nodes[to_node_key].node_type == NODE_TYPE_CYCLE_END:
                    t_type = "cycle_end"
                elif graph.nodes[to_node_key].node_type == NODE_TYPE_IDLE:
                    t_type = NODE_TYPE_IDLE
                else:
                    t_type = NODE_TYPE_END
        else:
            t_type = "action_transition"
        return t_type

    def _add_transition(self, graph, from_key, to_key):
        transition_model = None
        if self.load_transition_models:
            transition_model = self._load_transition_model(graph, from_key, to_key)
        transition_type = self._get_transition_type(graph, from_key, to_key)
        #print("add transition", from_key, to_key, transition_type)
        graph.nodes[from_key].outgoing_edges[to_key] = MotionStateTransition(from_key, to_key, transition_type, transition_model)
