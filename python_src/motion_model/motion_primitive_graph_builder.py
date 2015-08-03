# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 13:59:46 2015

@author: erhe01
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:57:42 2015

@author: erhe01
"""

import os
from . import NODE_TYPE_START, NODE_TYPE_STANDARD, NODE_TYPE_END
from utilities.io_helper_functions import load_json_file

from gp_mixture import GPMixture
from motion_primitive_node import MotionPrimitiveNode
from graph_edge import GraphEdge
from motion_primitive_graph import MotionPrimitiveGraph

 
class MotionPrimitiveGraphBuilder(object):
    """ Contains the motion primitives of an elementary action as nodes and
    transition models as edges. 
             
    Parameters
    ----------
    * elementary_action_name: string
    \t The name of the elementary action that the subgraph represents
    
    * morphable_model_directory: string
    \tThe directory of the morphable models of an elementary action.
    
    * transition_model_directory: string
    \tThe directory of the transition models.
    """
    def __init__(self):
        self.elementary_action_name = None
        self.nodes = {}
        self.morphable_model_directory = None
        self.has_transition_models = False
        self.meta_information = None
        self.annotation_map = {}
        self.start_states = []
        self.end_states = []
        self.motion_primitive_annotations = {}
        self.loaded_from_dict = False
   
    def set_properties(self, transition_model_directory=None, load_transition_models=False, update_stats=False):
        self.load_transition_models = load_transition_models
        self.transition_model_directory = transition_model_directory
        self.update_stats = update_stats
        return

    def set_data_source(self, elementary_action_name, morphable_model_directory, subgraph_desc=None, graph_definition=None):
        self.subgraph_desc = subgraph_desc
        self.elementary_action_name = elementary_action_name
        self.morphable_model_directory = morphable_model_directory
        self.graph_definition = graph_definition

        
    def build(self):
        motion_primitive_graph = MotionPrimitiveGraph()
        if self.subgraph_desc is not None:
            self._init_from_dict(motion_primitive_graph)
        else:
            self._init_from_directory(motion_primitive_graph)
        return motion_primitive_graph
        
    def _init_from_dict(self, motion_primitive_graph):
        motion_primitive_graph.loaded_from_dict = True
        motion_primitive_graph.elementary_action_name = self.subgraph_desc["name"]

        for m_primitive in self.subgraph_desc["nodes"].keys():
             motion_primitive_graph.nodes[m_primitive] = MotionPrimitiveNode()
             motion_primitive_graph.nodes[m_primitive].init_from_dict(self.elementary_action_name,self.subgraph_desc["nodes"][m_primitive])
        
        if "info" in self.subgraph_desc.keys():
            motion_primitive_graph._set_meta_information(self.subgraph_desc["info"])
        else:
            motion_primitive_graph._set_meta_information() 
        #self._set_transitions_from_dict(motion_primitive_graph)
        motion_primitive_graph._update_attributes(update_stats=False)
        return

    def _init_from_directory(self, motion_primitive_graph):
        motion_primitive_graph.loaded_from_dict = False
        motion_primitive_graph.elementary_action_name = self.elementary_action_name
        motion_primitive_graph.nodes = {}
        motion_primitive_graph.morphable_model_directory = self.morphable_model_directory
   
        #load morphable models
        temp_file_list =  []#for files containing additional information that require the full graph to be constructed first
        meta_information = None
        motion_primitive_graph.annotation_map = {}
        for root, dirs, files in os.walk(self.morphable_model_directory):
            for file_name in files:#for each morphable model 
                if file_name == "meta_information.json":
                    meta_information = load_json_file(self.morphable_model_directory+os.sep+file_name)
                    print "found meta information"
                elif file_name.endswith("mm.json"):
                    print "found motion primitive",file_name  
                    motion_primitive_name = file_name.split("_")[1]  
                    #print motion_primitve_file_name
                    motion_primitive_file_name = self.morphable_model_directory+os.sep+file_name
                    motion_primitive_graph.nodes[motion_primitive_name] = MotionPrimitiveNode()
                    motion_primitive_graph.nodes[motion_primitive_name].init_from_file(motion_primitive_graph.elementary_action_name,motion_primitive_name,motion_primitive_file_name)
                    
                elif file_name.endswith(".stats"):
                    print "found stats",file_name
                    temp_file_list.append(file_name)

                else:
                    print "ignored",file_name

        motion_primitive_graph._set_meta_information(meta_information)
        
        #load information about training data if available
        for file_name in temp_file_list:
            motion_primitive = file_name.split("_")[1][:-6]
            if motion_primitive in self.nodes.keys():
                info = load_json_file(self.morphable_model_directory+os.sep+file_name,use_ordered_dict=True)
                motion_primitive_graph.nodes[motion_primitive].parameter_bb = info["pose_bb"]
                motion_primitive_graph.nodes[motion_primitive].cartesian_bb = info["cartesian_bb"]
                motion_primitive_graph.nodes[motion_primitive].velocity_data = info["pose_velocity"]

      
               
       # self._set_transitions_from_directory(motion_primitive_graph)
       
        motion_primitive_graph._update_attributes(update_stats=self.update_stats)     


