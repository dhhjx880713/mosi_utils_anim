# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:57:51 2015

@author: erhe01
"""

import os
from ..animation_data.bvh import BVHReader
from ..animation_data.skeleton import Skeleton
from ..utilities.io_helper_functions import load_json_file
from gp_mixture import GPMixture
from motion_primitive_node_group_loader import MotionPrimitiveNodeGroupLoader
from ..utilities.zip_io import ZipReader
from graph_edge import GraphEdge
from motion_primitive_graph import MotionPrimitiveGraph
from . import NODE_TYPE_START, NODE_TYPE_STANDARD, TRANSITION_DEFINITION_FILE_NAME, TRANSITION_MODEL_FILE_ENDING

        
class MotionPrimitiveGraphLoader(object):
    """   Constructs a MotionPrimitiveGraph instance from a zip file or directory as data source
    """  
    def __init__(self):
        self.skeleton = None
        self.load_transition_models = False
        self.update_stats = False
        self.motion_primitive_graph_path = None
        self.elementary_action_directory = None
        self.transition_model_directory = None
        self.motion_primitive_node_group_builder = MotionPrimitiveNodeGroupLoader()

    def set_data_source(self, skeleton_path, motion_primitive_graph_path, transition_model_directory, load_transition_models=False, update_stats=False):
        """ Set the source which is used to load the data structure into memory.
        Parameters
        ----------
        *skeleton_path: string
        \tpath to a reference BVH file with the skeleton used with the motion data.
        * elementary_action_directory: string
        \tThe root directory of the morphable models of all elementary actions.
        * transition_model_directory: string
        \tThe directory of the morphable models of an elementary action.
        * transition_model_directory: string
        \tThe directory of the transition models.
        """
        self.skeleton = Skeleton(BVHReader(skeleton_path))
        self.load_transition_models = load_transition_models
        self.update_stats = update_stats
        self.motion_primitive_graph_path = motion_primitive_graph_path
        self.transition_model_directory = transition_model_directory
        self.motion_primitive_node_group_builder.set_properties(self.transition_model_directory,
                                                                self.load_transition_models)

    def build(self):
        motion_primitive_graph = MotionPrimitiveGraph()
        motion_primitive_graph.skeleton = self.skeleton
        if os.path.isfile(self.motion_primitive_graph_path+".zip"):
            self._init_from_zip_file(motion_primitive_graph)
        else:
            self._init_from_directory(motion_primitive_graph)
        return motion_primitive_graph

    def _init_from_zip_file(self, motion_primitive_graph):
        zip_path = self.motion_primitive_graph_path+".zip"
        zip_reader = ZipReader(zip_path, pickle_objects=True)
        graph_data = zip_reader.get_graph_data()
        graph_definition = graph_data["transitions"]
        elementary_actions = graph_data["subgraphs"]
        for action_name in elementary_actions.keys():
            self.motion_primitive_node_group_builder.set_data_from_zip(elementary_actions[action_name])
            node_group = self.motion_primitive_node_group_builder.build()
            motion_primitive_graph.nodes.update(node_group.nodes)
            motion_primitive_graph.node_groups[node_group.elementary_action_name] = node_group
        print "add transitions between nodes from", graph_definition
        self._set_transitions_from_dict(motion_primitive_graph, graph_definition)

        self._update_attributes(motion_primitive_graph, update_stats=False)

    def _init_from_directory(self, motion_primitive_graph, update_stats=True):
        """ Initializes the class
        """
        #load graphs representing elementary actions including transitions between actions
        for key in next(os.walk(self.motion_primitive_graph_path))[1]:
            subgraph_path = self. motion_primitive_graph_path+os.sep+key
            name = key.split("_")[-1]
            self.motion_primitive_node_group_builder.set_directory_as_data_source(name, subgraph_path)
            node_group = self.motion_primitive_node_group_builder.build()
            motion_primitive_graph.nodes.update(node_group.nodes)
            motion_primitive_graph.node_groups[node_group.elementary_action_name] = node_group

        graph_definition_file = self.motion_primitive_graph_path+os.sep+TRANSITION_DEFINITION_FILE_NAME
        #add transitions between subgraphs and load transition models
        if os.path.isfile(graph_definition_file):
            graph_definition = load_json_file(graph_definition_file)
            print "add transitions between subgraphs from", graph_definition_file
            self._set_transitions_from_dict(motion_primitive_graph, graph_definition)
        else:
            print "did not find graph definition file", graph_definition_file

        self._update_attributes(motion_primitive_graph, update_stats=update_stats)

    def _update_attributes(self, motion_primitive_graph, update_stats=False):
        for keys in motion_primitive_graph.node_groups.keys():
            motion_primitive_graph.node_groups[keys].update_attributes(update_stats=update_stats)

    def _set_transitions_from_dict(self, motion_primitive_graph, graph_definition):
        transition_dict = graph_definition["transitions"]
        
        for node_key in transition_dict:
            from_action_name = node_key.split("_")[0]
            from_motion_primitive_name = node_key.split("_")[1]
            from_node_key = (from_action_name, from_motion_primitive_name)
            if from_node_key in motion_primitive_graph.nodes.keys():
                #print "add action transitions for", subgraph_key,"###############################"
                for to_key in transition_dict[node_key]:
                    to_action_name = to_key.split("_")[0]
                    to_motion_primitive_name = to_key.split("_")[1]
                    to_node_key = (to_action_name, to_motion_primitive_name)
                    if to_node_key in motion_primitive_graph.nodes.keys():
                        self._add_transition(motion_primitive_graph, from_node_key, to_node_key)

    def _add_transition(self, motion_primitive_graph, from_node_key, to_node_key):
        transition_model = None
        if self.load_transition_models and self.transition_model_directory is not None:
            transition_model_file = self.transition_model_directory\
            +os.sep+from_node_key+"_to_"+to_node_key[0]+"_"+to_node_key[1]+TRANSITION_MODEL_FILE_ENDING
            if os.path.isfile(transition_model_file):
                output_gmm = motion_primitive_graph.nodes[to_node_key].gaussian_mixture_model
                transition_model = GPMixture.load(transition_model_file,\
                motion_primitive_graph.nodes[from_node_key].gaussian_mixture_model,output_gmm)
            else:
                print "did not find transition model file", transition_model_file
        self._create_edge(motion_primitive_graph, from_node_key, to_node_key, transition_model)

    def _get_transition_type(self, motion_primitive_graph, from_node_key, to_node_key):
        if to_node_key[0] == from_node_key[0]:
            if motion_primitive_graph.nodes[to_node_key].node_type in [NODE_TYPE_START, NODE_TYPE_STANDARD]:
                transition_type = "standard"
            else:
                transition_type = "end"
        else:
            transition_type = "action_transition"
        return transition_type

    def _create_edge(self, motion_primitive_graph, from_node_key, to_node_key, transition_model=None):
        transition_type = self._get_transition_type(motion_primitive_graph, from_node_key, to_node_key)
        edge = GraphEdge(from_node_key, to_node_key, transition_type, transition_model)
        motion_primitive_graph.nodes[from_node_key].outgoing_edges[to_node_key] = edge
