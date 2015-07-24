# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:57:51 2015

@author: erhe01
"""

import os
import time
import collections
import numpy as np
import datetime
from animation_data.bvh import BVHReader, BVHWriter
from animation_data.skeleton import Skeleton
from utilities.io_helper_functions import get_morphable_model_directory, \
                             get_transition_model_directory, \
                             load_json_file
from GPMixture import GPMixture
from animation_data.motion_editing import align_frames, convert_quaternion_to_euler
from motion_primitive_graph import MotionPrimitiveGraph
from utilities.zip_io import read_graph_data_from_zip
from graph_edge import GraphEdge


        
class ElementaryActionGraph(object):
    """ Contains a list of MotionPrimitiveGraphs for each elementary action and
         transition models between them.
    """
     
    def __init__(self, skeleton_path, morphable_model_directory,transition_model_directory, load_transition_models=False, update_stats=False):
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
        self.subgraphs = collections.OrderedDict()
        zip_path = morphable_model_directory+".zip"
        if os.path.isfile(zip_path):
            self.init_from_zip_file(zip_path, transition_model_directory, load_transition_models)
        else:
            self.init_from_directory(morphable_model_directory
                                    ,transition_model_directory, 
                                    load_transition_models=load_transition_models,
                                    update_stats=update_stats)
        return
        
    def init_from_zip_file(self,zip_path, transition_model_directory, load_transition_models):
        graph_data = read_graph_data_from_zip(zip_path,pickle_objects=True)
        graph_definition = graph_data["transitions"]
        subgraph_desc = graph_data["subgraphs"]
        for el_action in subgraph_desc.keys():
            self.subgraphs[el_action] = MotionPrimitiveGraph()
            self.subgraphs[el_action].init_from_dict(subgraph_desc[el_action],graph_definition, transition_model_directory, load_transition_models)
            #for m_primitive in subgraph_desc[el_action].keys():
        self._set_transitions_from_dict(graph_definition,transition_model_directory, load_transition_models)          
        
    def init_from_directory(self,morphable_model_directory,transition_model_directory, load_transition_models=False, update_stats=False):
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
        self.subgraphs = collections.OrderedDict()
        if morphable_model_directory is not None:
            self.morphable_model_directory = morphable_model_directory
            
            for key in next(os.walk(morphable_model_directory))[1]:
                subgraph_path = morphable_model_directory+os.sep+key
                name = key.split("_")[-1]
                self.subgraphs[name] = MotionPrimitiveGraph()
                self.subgraphs[name].init_from_directory(name,subgraph_path,transition_model_directory, load_transition_models, update_stats)
                
        graph_definition_file = morphable_model_directory+os.sep+"graph_definition.json"
        
        #add transitions between subgraphs and load transition models           
        if os.path.isfile(graph_definition_file):
            graph_definition = load_json_file(graph_definition_file)
            print "add transitions between subgraphs from",graph_definition_file," ##################################"
            self._set_transitions_from_dict(graph_definition, transition_model_directory, load_transition_models)          
        else:
            print "did not find graph definition file",graph_definition_file," #####################"


            
    def _set_transitions_from_dict(self, graph_definition, transition_model_directory= None,load_transition_models=False):
        transition_dict = graph_definition["transitions"]
        
        for subgraph_key in self.subgraphs.keys():
            
            for node_key in transition_dict:
                from_action_name = node_key.split("_")[0]
                from_motion_primitive_name = node_key.split("_")[1]
                if from_action_name == subgraph_key and \
                   from_motion_primitive_name in self.subgraphs[from_action_name].nodes.keys():
                    #print "add action transitions for", subgraph_key,"###############################"
                    for to_key in transition_dict[node_key]:
                        
                        to_action_name = to_key.split("_")[0]
                        to_motion_primitive_name = to_key.split("_")[1]
                        if to_action_name != from_action_name:
                            print "add action transition",node_key,to_key 
                            transition_model = None
                            if load_transition_models and transition_model_directory is not None:
                                transition_model_file = transition_model_directory\
                                +os.sep+node_key+"_to_"+to_key+".GPM"
                                if  os.path.isfile(transition_model_file):
                                    output_gmm =  self.subgraphs[to_action_name].nodes[to_motion_primitive_name].motion_primitive.gmm
                                    transition_model = GPMixture.load(transition_model_file,\
                                    self.subgraphs[from_action_name].nodes[from_motion_primitive_name].motion_primitive.gmm,output_gmm)
                                else:
                                    print "did not find transition model file", transition_model_file
                            transition_type = "action_transition"
                           
                            edge = GraphEdge(from_action_name,from_motion_primitive_name,to_action_name,\
                            to_motion_primitive_name,transition_type,transition_model)
                            self.subgraphs[from_action_name].nodes[from_motion_primitive_name].outgoing_edges[to_key] = edge


    @classmethod
    def init_from_dict2(cls,subgraph_dict,subgraph_transitions = None):
        """ Intitializes the class using a dictionary of subgraphs
        
        Parameters
        ----------
        * subgraph_dict : dict
        \t A dictionary containing MotionPrimitiveGraph instances
        * subgraph_transitions : dict
        \t A dictionary containing additional transitions between subgraphs
        
        Returns
        -------
        *morphable_graph: ElementaryActionGraph
        \t The ElementaryActionGraph instance
        """
        morphable_graph = cls(None,None,False)
        morphable_graph.subgraphs = subgraph_dict
        
        return morphable_graph

    def generate_random_walk(self, start_subgraph,number_of_steps, use_transition_model = True):
        """ Generates a random graph walk
            
        Parameters
        ----------
        * start_subgraph: string
        \tInitial subgraph.
        
        * number_of_steps: integer
        \tNumber of transitions
        
        * use_transition_model: bool
        \tSets whether or not the transition model should be used in parameter prediction
        
        Returns
        -------
        *graph_walk: a list of dictionaries
        \t The graph walk is defined by a list of dictionaries containing entries for "subgraph","state" and "parameters"
        """
        assert start_subgraph in self.subgraphs.keys()
        print "generate random graph walk for",start_subgraph
        current_subgraph = start_subgraph
        start_state = self.subgraphs[current_subgraph].get_random_start_state()
        return self.subgraphs[current_subgraph].generate_random_walk(start_state, number_of_steps, use_transition_model)
    
    def print_information(self):
        """
        Prints out information on the graph structure and properties of the motion primitives
        """
        for s in self.subgraphs.keys():
            print s
            for n in self.subgraphs[s].nodes.keys():
                print "\t"+ n
                print "\t"+"n canonical frames",self.subgraphs[s].nodes[n].motion_primitive.n_canonical_frames
                print "\t"+"n latent spatial dimensions",self.subgraphs[s].nodes[n].motion_primitive.s_pca["n_components"]
                print "\t"+"n latent time dimensions",self.subgraphs[s].nodes[n].motion_primitive.t_pca["n_components"]
                print "\t"+"n basis spatial ",self.subgraphs[s].nodes[n].motion_primitive.s_pca["n_basis"]
                print "\t"+"n basis time ",self.subgraphs[s].nodes[n].motion_primitive.t_pca["n_basis"]
                print "\t"+"n clusters",len(self.subgraphs[s].nodes[n].motion_primitive.gmm.weights_)
                print "\t"+"average length", self.subgraphs[s].nodes[n].average_step_length
                for e in self.subgraphs[s].nodes[n].outgoing_edges.keys():
                    print "\t \t to "+ e
                print "\t##########"       
                
                
    def get_random_action_transition(self, motion, action_name):
        """ Get random start state based on edge from previous elementary action if possible
        """
        next_state = ""
        if motion.step_count > 0:
            prev_action_name = motion.graph_walk[-1].action_name
            prev_mp_name = motion.graph_walk[-1].motion_primitive_name
      
            if prev_action_name in self.subgraphs.keys() and \
                   prev_mp_name in self.subgraphs[prev_action_name].nodes.keys():
                                       
               to_key = self.subgraphs[prev_action_name]\
                               .nodes[prev_mp_name].generate_random_action_transition(action_name)
               if to_key is not None:
                   next_state = to_key.split("_")[1]
                   return next_state
               else:
                   return None
               print "generate start from transition of last action", prev_action_name, prev_mp_name, to_key
           
        # if there is no previous elementary action or no action transition
        #  use transition to random start state
        if next_state == "" or next_state not in self.subgraphs[action_name].nodes.keys():
            print next_state,"not in", action_name#,prev_action_name,prev_mp_name
            next_state = self.subgraphs[action_name].get_random_start_state()
            print "generate random start",next_state
        return next_state


def print_morphable_graph_structure(morphable_graph):
    for s in morphable_graph.subgraphs.keys():
        print s
        for n in morphable_graph.subgraphs[s].nodes.keys():
            print "\t" + n
            print "\t" + "canonical frames", morphable_graph.subgraphs[s].nodes[n].motion_primitive.n_canonical_frames
            for e in morphable_graph.subgraphs[s].nodes[n].outgoing_edges.keys():
                print "\t \t to " + e
    return
    
    
def convert_graph_walk_to_euler_frames(skeleton, morphable_graph, graph_walk, node_name_map, smooth = True):
    """
     Converts a graph walk to a list of euler frames and concatenates the frames 
     to one list by aligning them at transitions.
     
     Parameters
    ----------
    * skeleton: Skeleton
    \tUsed for to extract the skeleton hierarchy information.
    * morphable_graph: ElementaryActionGraph
    \tData structure containing the morphable models
    * graph_walk: list of dict
    \tContains a list of dictionaries with the entries for "subgraph","state" and "parameters"
    * node_name_map: dict
    \tMaps joint names to indices in their original loading sequence ignoring the "Bip" joints
    * smooth: bool
    \tSets whether or not smoothing is applied on transitions
    Returns
    -------
    * concatenated_frames: numpy.ndarray
    \tA list of euler frames representing a motion.
    """
    
    concatenated_frames = None
    print "export graph walk with length",len(graph_walk)
    for entry in graph_walk:
        print entry["state"]#,entry["parameters"]
        subgraph = entry["subgraph"]
        state = entry["state"]
        parameters = entry["parameters"]
        quaternion_frames = morphable_graph.subgraphs[subgraph].nodes[state].motion_primitive.back_project(parameters).get_motion_vector()
        euler_frames = convert_quaternion_to_euler(quaternion_frames.tolist())
        if concatenated_frames is not None:
            concatenated_frames = align_frames(skeleton,concatenated_frames,euler_frames, smooth) 
        else:
            concatenated_frames = euler_frames
  
    return concatenated_frames

def export_graph_walk_to_bvh(skeleton, graph_walk, morphable_graph, concatenate=True , apply_smoothing= True, prefix= "",out_dir ="."):
    """Saves a graph walk as a single bvh file or as a list of bvh files
    
    """
    if concatenate:
        concatenated_frames = convert_graph_walk_to_euler_frames(skeleton, morphable_graph, \
                                                graph_walk,apply_smoothing)
    
        filepath = out_dir+os.sep+prefix+"output_" + unicode(datetime.datetime.now().strftime("%d%m%y_%H%M%S"))+".bvh"
        BVHWriter(filepath,skeleton, concatenated_frames,skeleton.frame_time,is_quaternion=False)
    else:
        time_code = out_dir+os.sep+prefix+"output_" + unicode(datetime.datetime.now().strftime("%d%m%y_%H%M%S"))
        i = 0
        for entry in graph_walk:
            filepath = time_code+"_"+str(i)+".bvh"
            subgraph = entry["subgraph"]
            state = entry["state"]
            parameters = entry["parameters"]
            quaternion_frames = morphable_graph.subgraphs[subgraph].nodes[state].motion_primitive.back_project(parameters).get_motion_vector()
            euler_frames = convert_quaternion_to_euler(quaternion_frames.tolist())
            BVHWriter(filepath,skeleton, euler_frames,skeleton.frame_time,is_quaternion=False)
            i += 1
    
def test_morphable_graph(load_transition_models = False):
    np.random.seed(int(time.time()))
    skeleton_file = "skeleton.bvh"

    start = time.time()
    mm_directory =".."+ os.sep + get_morphable_model_directory()
    transition_directory =".."+ os.sep +  get_transition_model_directory()
    
    morphable_graph = ElementaryActionGraph(skeleton_file, mm_directory,transition_directory, load_transition_models)


    print_morphable_graph_structure(morphable_graph)
    print "loaded the morphable graph in",time.time()-start,"seconds"

    n_steps = 10
    elementary_action = "walk"
    graph_walk = morphable_graph.generate_random_walk(elementary_action,n_steps,load_transition_models)
    export_graph_walk_to_bvh(morphable_graph.skeleton, graph_walk, morphable_graph, True, True, prefix = "smoothed_",out_dir="random_walks")
    #export_graph_walk_to_bvh(bvh_reader, graph_walk, morphable_graph, True, False, prefix = "",out_dir="random_walks")
    
def load_subgraph(elementary_action = "walk",load_transition_models = True):
    np.random.seed(int(time.time()))
    mm_directory = get_morphable_model_directory() 
    transition_directory = get_transition_model_directory()
    start = time.time()
    #search for morphable models
    subgraph = None
    for key in next(os.walk(mm_directory))[1]:
        name = key.split("_")[-1]
        print name
        if name == elementary_action:
            subgraph_path = mm_directory+os.sep+key
            subgraph = MotionPrimitiveGraph()
            subgraph.init_from_directory(name,subgraph_path,transition_directory, load_transition_models)
            break
    
    
    if subgraph is not None:
        print "loaded the morphable sub graph for", elementary_action,"in",time.time()-start,"seconds"
    else:
        print elementary_action, "not found"
    return subgraph
    
    
def test_morphable_subgraph(elementary_action = "walk",load_transition_models = True, apply_smoothing = True):
  
    skeleton_file = "skeleton.bvh"
    skeleton = Skeleton(BVHReader(skeleton_file))
    subgraph = load_subgraph(elementary_action,load_transition_models)
    n_steps = 2
    if subgraph is not None:
        start_state = subgraph.get_random_start_state()
       
        graph_walk = subgraph.generate_random_walk(start_state,n_steps,load_transition_models)
        print graph_walk
        subgraph_dict = {elementary_action:subgraph}
        morphable_graph = ElementaryActionGraph.init_from_dict2(subgraph_dict)
        #print print_morphable_graph_structure(morphable_graph)
        
        prefix = elementary_action+"_"
        if load_transition_models:
            prefix  += "from_transition_"
        export_graph_walk_to_bvh(skeleton, graph_walk, morphable_graph, True, apply_smoothing, prefix = prefix,out_dir="random_walks")
        
        

    
def main():
    test_morphable_graph(False)


if __name__ ==  "__main__":
    main()
    