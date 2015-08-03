# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 12:31:00 2015

@author: erhe01
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:57:51 2015

@author: erhe01
"""

import os
from animation_data.bvh import BVHReader
from animation_data.skeleton import Skeleton
from utilities.io_helper_functions import load_json_file
from gp_mixture import GPMixture
from motion_primitive_graph_builder import MotionPrimitiveGraphBuilder
from utilities.zip_io import read_graph_data_from_zip
from graph_edge import GraphEdge
from elementary_action_graph import ElementaryActionGraph


        
class ElementaryActionGraphBuilder(object):
    """   Builds an ElementaryActionGraph
    """  
    def __init__(self):
        self.skeleton = None
        self.load_transition_models = False
        self.update_stats = False
        self.morphable_model_directory = None
        self.transition_model_directory = None
        self.motion_primitive_graph_builder = MotionPrimitiveGraphBuilder()
     
        return
        
    def set_data_source(self, skeleton_path, morphable_model_directory, transition_model_directory, load_transition_models=False, update_stats=False):
     
        """ Initializes the class
        
        Parameters
        ----------
        *skeleton_path: string
        \tpath to a reference BVH file with the skeleton used with the motion data.
        
        * morphable_model_directory: string
        \tThe root directory of the morphable models of all elementary actions.
        
        * transition_model_directory: string
        \tThe directory of the morphable models of an elementary action.
        
        * transition_model_directory: string
        \tThe directory of the transition models.
        """
        self.skeleton = Skeleton(BVHReader(skeleton_path))
        self.load_transition_models = load_transition_models
        self.update_stats = update_stats
        self.morphable_model_directory = morphable_model_directory
        self.transition_model_directory = transition_model_directory
        self.motion_primitive_graph_builder.set_properties(transition_model_directory=self.transition_model_directory, load_transition_models=self.load_transition_models)
    def build(self):
        elementary_action_graph = ElementaryActionGraph()
        elementary_action_graph.skeleton = self.skeleton
        if os.path.isfile(self.morphable_model_directory+".zip"):
            self._init_from_zip_file(elementary_action_graph)
        else:
            self._init_from_directory(elementary_action_graph)
        return elementary_action_graph
        
    def _init_from_zip_file(self, elementary_action_graph):
        
        zip_path = self.morphable_model_directory+".zip"
        graph_data = read_graph_data_from_zip(zip_path, pickle_objects=True)
        graph_definition = graph_data["transitions"]
        subgraph_desc = graph_data["subgraphs"]
        for el_action in subgraph_desc.keys():
            subgraph_path = self.morphable_model_directory+os.sep+el_action
            self.motion_primitive_graph_builder.set_data_source(el_action, subgraph_path, subgraph_desc=subgraph_desc[el_action], graph_definition=graph_definition)
            elementary_action_graph.subgraphs[el_action] = self.motion_primitive_graph_builder.build()
            #for m_primitive in subgraph_desc[el_action].keys():
        self._set_transitions_from_dict(elementary_action_graph, graph_definition)
  
    def _init_from_directory(self, elementary_action_graph):
        """ Initializes the class
        
        Parameters
        ----------
        * morphable_model_directory: string
        \tThe root directory of the morphable models of all elementary actions.
        
        * transition_model_directory: string
        \tThe directory of the morphable models of an elementary action.
        
        * transition_model_directory: string
        \tThe directory of the transition models.
        """
        
        #load graphs representing elementary actions including transitions between actions
        for key in next(os.walk(self.morphable_model_directory))[1]:
            subgraph_path = self. morphable_model_directory+os.sep+key
            name = key.split("_")[-1]
            self.motion_primitive_graph_builder.set_data_source(name, subgraph_path)
            elementary_action_graph.subgraphs[name] = self.motion_primitive_graph_builder.build()

        graph_definition_file = self.morphable_model_directory+os.sep+"graph_definition.json"
        
        #add transitions between subgraphs and load transition models           
        if os.path.isfile(graph_definition_file):
            graph_definition = load_json_file(graph_definition_file)
            print "add transitions between subgraphs from",graph_definition_file," ##################################"
            self._set_transitions_from_dict(elementary_action_graph, graph_definition)          
        else:
            print "did not find graph definition file",graph_definition_file," #####################"
  

            
    def _set_transitions_from_dict(self, elementary_action_graph, graph_definition):
        transition_dict = graph_definition["transitions"]
        
        for subgraph_key in elementary_action_graph.subgraphs.keys(): 
            for node_key in transition_dict:
                from_action_name = node_key.split("_")[0]
                from_motion_primitive_name = node_key.split("_")[1]
                if from_action_name == subgraph_key and \
                   from_motion_primitive_name in elementary_action_graph.subgraphs[from_action_name].nodes.keys():
                    #print "add action transitions for", subgraph_key,"###############################"
                    for to_key in transition_dict[node_key]:
                        
                        to_action_name = to_key.split("_")[0]
                        to_motion_primitive_name = to_key.split("_")[1]
                        if to_action_name != from_action_name:
                            print "add action transition",node_key,to_key 
                            transition_model = None
                            if self.load_transition_models and self.transition_model_directory is not None:
                                transition_model_file = self.transition_model_directory\
                                +os.sep+node_key+"_to_"+to_key+".GPM"
                                if  os.path.isfile(transition_model_file):
                                    output_gmm =  elementary_action_graph.subgraphs[to_action_name].nodes[to_motion_primitive_name].motion_primitive.gmm
                                    transition_model = GPMixture.load(transition_model_file,\
                                    elementary_action_graph.subgraphs[from_action_name].nodes[from_motion_primitive_name].motion_primitive.gmm,output_gmm)
                                else:
                                    print "did not find transition model file", transition_model_file
                            transition_type = "action_transition"
                           
                            edge = GraphEdge(from_action_name,from_motion_primitive_name,to_action_name,\
                            to_motion_primitive_name,transition_type,transition_model)
                            elementary_action_graph.subgraphs[from_action_name].nodes[from_motion_primitive_name].outgoing_edges[to_key] = edge