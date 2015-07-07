# -*- coding: utf-8 -*-
"""
Created on Sun Mar 01 12:33:44 2015

@author: erhe01
"""
import os
import time
import collections
import random
import numpy as np
import datetime
from bvh import BVHReader, BVHWriter
from helper_functions import get_morphable_model_directory, \
                             get_transition_model_directory, \
                             gen_file_paths, \
                             load_json_file, \
                             write_to_json_file
from motion_primitive import MotionPrimitive
from GPMixture import GPMixture
from motion_editing import align_frames, convert_quaternion_to_euler
from math import sqrt


def extract_root_positions(euler_frames):
    roots_2D = []
    for i in xrange(len(euler_frames)):
        position_2D = np.array([ euler_frames[i][0],euler_frames[i][1], euler_frames[i][2] ])
        #print "sample",position2D
        roots_2D.append(position_2D)
    return np.array(roots_2D) 


def get_arc_length_from_points(points):
    """
    Note: accuracy depends on the granulariy of points
    """
    arc_length = 0.0
    last_p = None
    for p in points:
        if last_p != None:
            delta = p - last_p
            #print delta
            arc_length += sqrt( delta[0]**2 + delta[1]**2 +delta[2]**2) #-arcLength
        else:
            delta = p
        last_p = p            
    return arc_length

def get_step_length(motion_primitive,method = "arc_length"):
    """Backproject the motion and get the step length and the last keyframe on the canonical timeline
    Parameters
    ----------
    * morphable_subgraph : MorphableSubgraph
      Represents an elementary action
    * motion_primitive_name : string
      Identifier of the morphable model
    * method : string
      Can have values arc_length or distance. If any other value distance is used.
    Returns
    -------
    *step_length: float
    \tThe arc length of the path of the motion primitive
    """

    current_parameters = motion_primitive.sample(return_lowdimvector=True)
    quat_frames = motion_primitive.back_project(current_parameters,use_time_parameters=False).get_motion_vector()
    if method == "arc_length":
        root_pos = extract_root_positions(quat_frames)
        #print root_pos
        step_length = get_arc_length_from_points(root_pos)
    else:# use distance
        vector = quat_frames[-1][:3] - quat_frames[0][:3] 
        magnitude = 0
        for v in vector:
            magnitude += v**2
        step_length = sqrt(magnitude)
    return step_length



class GraphEdge(object):
    """ Contains a transition model. 
    """
    def __init__(self,from_action,from_motion_primitive,to_action,to_motion_primitive,transition_type = "standard",transition_model = None):
        self.from_action = from_action
        self.to_action = to_action
        self.from_motion_primitive = from_motion_primitive
        self.to_motion_primitive = to_motion_primitive
        self.transition_type = transition_type
        self.transition_model = transition_model

class GraphNode(object):
    """ Contains a motion primitive and all its outgoing transitions. 

    Parameters
    ----------
    * motion_primitive_filename: string
    \tThe filename with the saved data in json format.
    
    Attributes
    ----------
    * mp: MotionPrimitive
    \tThe motion primitive instance that is wrapped by the node class.
    * outgoing_edges: OrderedDict containing tuples
    \tEach entry contains a tuple (transition model, transition type)
    """
    def __init__(self, action_name,primitive_name,motion_primitive_filename, node_type="standard"):
        self.mp = MotionPrimitive(motion_primitive_filename)
        self.outgoing_edges = {}
        self.node_type = node_type
        self.n_standard_transitions = 0
        self.parameter_bb = None
        self.cartesian_bb = None
        self.velocity_data = None
        self.cluster_annotation = None
        self.average_step_length = 0 
        self.action_name = action_name
        self.primitive_name = primitive_name
        
    def sample_parameters(self):
         """ Samples a low dimensional vector.
         Returns
         -------
         * parameters: numpy.ndarray
         \tLow dimensional motion parameters.
        
         """
         return self.mp.sample(return_lowdimvector=True)
        
    def generate_random_transition(self, transition_type ="standard"):
        """ Returns the key of a random transition.

        Parameters
        ----------
        * transition_type: string
        \t Idententifies edges as either standard or end transitions
        """
        if self.outgoing_edges:
            edges = [e for e in self.outgoing_edges.keys() if self.outgoing_edges[e].transition_type == transition_type]
            if len(edges) > 0:
                random_index = random.randrange(0, len(edges), 1)
                to_key = edges[random_index]
                #print "to",to_key
                return to_key
        return None
        
    def generate_random_action_transition(self,elementary_action):
        """ Returns the key of a random transition to the given elementary action.

        Parameters
        ----------
        * elementary_action: string
        \t Identifies an elementary action
        """
        if self.outgoing_edges:
            edges = [e for e in self.outgoing_edges.keys() if e.split("_")[0] == elementary_action]
            if len(edges) > 0:
                random_index = random.randrange(0, len(edges), 1)
                to_key = edges[random_index]
                #print "to",to_key
                return to_key
        return None
        
    
    def update_attributes(self):
        """ Updates attributes for faster look up
        """
        self.n_standard_transitions = len([e for e in self.outgoing_edges.keys() if self.outgoing_edges[e].transition_type == "standard"])
        n_samples = 50 
        sample_lengths = [get_step_length(self.mp)for i in xrange(n_samples)]
        method = "median"
        if method == "average":
            self.average_step_length = sum(sample_lengths)/n_samples
        else:
            self.average_step_length = np.median(sample_lengths)
        
 
    def predict_parameters(self, to_key, current_parameters):
        """ Predicts parameters for a transition using the transition model.
        
        Parameters
        ----------
        * to_key: string
        \t Name of the action and motion primitive we want to transition to. 
        \t Should have the format "action_motionprimitive" 
        
        * current_parameters: numpy.ndarray
        \tLow dimensional motion parameters.
        
        Returns
        -------
        * next_parameters: numpy.ndarray
        \tThe predicted parameters for the next state.
        """
        gmm = self.outgoing_edges[to_key].transition_model.predict(current_parameters)
        next_parameters = np.ravel(gmm.sample())  
        return next_parameters
        
    def predict_gmm(self, to_key, current_parameters):
        """ Predicts a Gaussian Mixture Model for a transition using the transition model.
        
        Parameters
        ----------
        * to_key: string
        \t Name of the action and motion primitive we want to transition to. 
        \t Should have the format "action_motionprimitive" 
        
        * current_parameters: numpy.ndarray
        \tLow dimensional motion parameters.
        
        Returns
        -------
        * gmm: sklearn.mixture.GMM
        \tThe predicted Gaussian Mixture Model.
        """
        return self.outgoing_edges[to_key].transition_model.predict(current_parameters)

 
class MorphableSubgraph(object):
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
    def __init__(self,elementary_action_name, morphable_model_directory,transition_model_directory, load_transition_models= False, update_stats=False):
        self.elementary_action_name = elementary_action_name
        self.nodes = {}
        self.morphable_model_directory = morphable_model_directory
        
        
        #load morphable models
        temp_file_list =  []#for files containing additional information that require the full graph to be constructed first
        self.meta_information = None
        self.annotation_map = {}
        for root, dirs, files in os.walk(morphable_model_directory):
            for file_name in files:#for each morphable model 
                if file_name == "meta_information.json":
                    self.meta_information = load_json_file(morphable_model_directory+os.sep+file_name)
                    print "found meta information"
                elif file_name.endswith("mm.json"):
                    print "found motion primitive",file_name  
                    motion_primitive_name = file_name.split("_")[1]
                    motion_primitve_file_name = morphable_model_directory+os.sep+file_name
                    #print motion_primitve_file_name
                    self.nodes[motion_primitive_name] = GraphNode(elementary_action_name,motion_primitive_name,motion_primitve_file_name)
                elif file_name.endswith(".stats"):
                    print "found stats",file_name
                    temp_file_list.append(file_name)

                else:
                    print "ignored",file_name

        #identify start and end states from meta information
        if self.meta_information == None:
            self.start_states = [k for k in self.nodes.keys() if k.startswith("begin") or k == "first"]
            self.end_states = [k for k in self.nodes.keys() if k.startswith("end") or k == "second"]
        else:
            for key in ["annotations", "start_states", "end_states" ]:
                assert key in self.meta_information.keys() 

            self.start_states = self.meta_information["start_states"]
            self.end_states = self.meta_information["end_states"]
            self.mp_annotations = self.meta_information["annotations"]
            if "pattern_constraints" in self.meta_information.keys():
                pattern_constraints = self.meta_information["pattern_constraints"]
                for k in pattern_constraints.keys():
                    if k in self.nodes.keys():
                        self.nodes[k].cluster_annotation = pattern_constraints[k]
            #create a map from semantic label to motion primitive
            for motion_primitive in self.meta_information["annotations"].keys():
                if motion_primitive != "all_primitives":
                    motion_primitve_annotations  = self.meta_information["annotations"][motion_primitive]
                    for label in motion_primitve_annotations.keys():
                        self.annotation_map[label] = motion_primitive
             
         
     
        print "elementary_action",self.elementary_action_name     
        print "start states",self.start_states
        for k in self.start_states:
            self.nodes[k].node_type = "start"
        print "end states",self.end_states
        for k in self.end_states:
            self.nodes[k].node_type = "end"
             
             
        #load information about training data if available
        for file_name in temp_file_list:
            motion_primitive = file_name.split("_")[1][:-6]
            if motion_primitive in self.nodes.keys():
                info = load_json_file(morphable_model_directory+os.sep+file_name,use_ordered_dict=True)
                self.nodes[motion_primitive].parameter_bb = info["pose_bb"]
                self.nodes[motion_primitive].cartesian_bb = info["cartesian_bb"]
                self.nodes[motion_primitive].velocity_data = info["pose_velocity"]

        self.has_transition_models = load_transition_models
               
        #load information about training data if available
        for file_name in temp_file_list:
            motion_primitive = file_name.split("_")[1][:-6]
            if motion_primitive in self.nodes.keys():
                info = load_json_file(morphable_model_directory+os.sep+file_name,use_ordered_dict=True)
                self.nodes[motion_primitive].parameter_bb = info["pose_bb"]
                self.nodes[motion_primitive].cartesian_bb = info["cartesian_bb"]
                self.nodes[motion_primitive].velocity_data = info["pose_velocity"]
        
        #define transitions and load transiton models
        self.has_transition_models = load_transition_models
        if os.path.isfile(morphable_model_directory+os.sep+".."+os.sep+"graph_definition.json"):
            graph_definition = load_json_file(morphable_model_directory+os.sep+".."+os.sep+"graph_definition.json")
            transition_dict = graph_definition["transitions"]
            for node_key in transition_dict:
                from_action_name = node_key.split("_")[0]
                from_motion_primitive_name = node_key.split("_")[1]
                if from_action_name == elementary_action_name and \
                   from_motion_primitive_name in self.nodes.keys():
                    for to_key in transition_dict[node_key]:
                        to_action_name = to_key.split("_")[0]
                        to_motion_primitive_name = to_key.split("_")[1]
                        if to_action_name == elementary_action_name:
                            transition_model_file = transition_model_directory\
                            +os.sep+node_key+"_to_"+to_key+".GPM"
                            if load_transition_models and os.path.isfile(transition_model_file):
                                output_gmm = self.nodes[to_motion_primitive_name].mp.gmm
                                transition_model = GPMixture.load(transition_model_file,\
                                self.nodes[from_motion_primitive_name].mp.gmm,output_gmm)
                            else:
                                transition_model = None
                            if self.nodes[to_motion_primitive_name].node_type in ["start","standard"]: 
                                transition_type = "standard"
                            else:
                                transition_type = "end"
                            print "add",transition_type
                            edge = GraphEdge(elementary_action_name,node_key,to_action_name,\
                            to_motion_primitive_name,transition_type,transition_model)
                            self.nodes[from_motion_primitive_name].outgoing_edges[to_key] = edge  
        else:
            print "Warning: no transitions were found in the directory"
                    
                    
#        
#        if load_transition_models:
#            #load transiton models
#            for filepath in gen_file_paths(transition_model_directory, mask="*.GPM"):
#                filename = filepath.split(os.sep)[-1]
#                splitted_file_name = filename.split("_")
#                from_action_name = splitted_file_name[0]
#                to_action_name = splitted_file_name[3]
#                #print "action name",action_name
#                from_motion_primitive_name = splitted_file_name[1]
#    
#                to_motion_primitive_name = splitted_file_name[4][:-4]
#    
#                #check if transition starts from this primitive
#                if from_action_name == self.elementary_action_name and from_motion_primitive_name in self.nodes.keys():
#                    #check transition type
#                    if to_action_name == self.elementary_action_name:
#                        transition_type = self.nodes[to_motion_primitive_name].node_type
#                    else:
#                        transition_type = "action_transition"
#                      
#                    transition_model =  None
#                    #if load_transition_models: #load mixture of gps and add output gmm for internal transitions
#                    
#                    
#                    if transition_type != "action_transition":
#                        output_gmm =  self.nodes[to_motion_primitive_name].mp.gmm
#                    else:
#                        output_gmm = None
#                     #print "transition model filename",filename
#                    transition_model = GPMixture.load(filepath, self.nodes[from_motion_primitive_name].mp.gmm,output_gmm)
#        
#                    print "add edge",from_action_name,from_motion_primitive_name, to_action_name, to_motion_primitive_name, transition_type
#                    edge = GraphEdge(from_action_name,from_motion_primitive_name,to_action_name,to_motion_primitive_name,transition_type,transition_model)
#                    to_key = to_action_name+"_"+to_motion_primitive_name
#                    self.nodes[from_motion_primitive_name].outgoing_edges[to_key] = edge
                    
        #update attributes for faster lookup
        changed_meta_info = False
        if update_stats:
            changed_meta_info = True
            self.meta_information["stats"] = {}
            for k in self.nodes.keys():
                 self.nodes[k].update_attributes()
                 self.meta_information["stats"][k]={"average_step_length":self.nodes[k].average_step_length,"n_standard_transitions": self.nodes[k].n_standard_transitions }
                 print"n standard transitions",k,self.nodes[k].n_standard_transitions
            print "updated meta information",self.meta_information
        else:
            
            if "stats" not in self.meta_information.keys():
                self.meta_information["stats"] = {}
            for k in self.nodes.keys():
                if k in self.meta_information["stats"].keys():
                    self.nodes[k].n_standard_transitions = self.meta_information["stats"][k]["n_standard_transitions"] 
                    self.nodes[k].average_step_length = self.meta_information["stats"][k]["average_step_length"]
                else:
                    self.nodes[k].update_attributes()
                    self.meta_information["stats"][k]={"average_step_length":self.nodes[k].average_step_length,"n_standard_transitions": self.nodes[k].n_standard_transitions }
                    changed_meta_info = True
            print "loaded stats from meta information file",self.meta_information
        if changed_meta_info:
            self.save_updated_meta_info()
            
    def get_random_start_state(self):
        """ Returns the name of a random start state. """
        random_index = random.randrange(0, len(self.start_states), 1)
        start_state = self.start_states[random_index]
        return start_state
            
    def get_random_end_state(self):
        """ Returns the name of a random start state."""
        random_index = random.randrange(0, len(self.end_states), 1)
        start_state = self.end_states[random_index]
        return start_state
        
    def generate_random_walk(self,start_state, number_of_steps, use_transition_model = True):
        """ Generates a random graph walk to be converted into a BVH file
    
        Parameters
        ----------
        * start_state: string
        \tInitial state.
        
        * number_of_steps: integer
        \tNumber of transitions
        
        * use_transition_model: bool
        \tSets whether or not the transition model should be used in parameter prediction
        """
        assert start_state in self.nodes.keys()
        graph_walk = []
        count = 0
        print "start",start_state
        current_state = start_state
        current_parameters = self.nodes[current_state].sample_parameters()
        entry = {"subgraph": self.elementary_action_name,"state": current_state,"parameters":current_parameters}
        graph_walk.append(entry)
        
        if self.nodes[current_state].n_standard_transitions > 0:
            while count < number_of_steps:
                #sample transition
                #print current_state
                to_key = self.nodes[current_state].generate_random_transition("standard") 
                next_parameters = self.generate_next_parameters(current_state,current_parameters,to_key,use_transition_model)
                #add entry to graph walk
                to_action  = to_key.split("_")[0]
                to_motion_primitive  = to_key.split("_")[1]
                entry = {"subgraph": to_action,"state": to_motion_primitive,"parameters":next_parameters}
                graph_walk.append(entry)
                current_parameters = next_parameters
                current_state = to_motion_primitive 
                count += 1
            
        #add end state
        to_key = self.nodes[current_state].generate_random_transition("end")
        next_parameters = self.generate_next_parameters(current_state,current_parameters,to_key,use_transition_model)
        to_action  = to_key.split("_")[0]
        to_motion_primitive  = to_key.split("_")[1]
        entry = {"subgraph": to_action,"state": to_motion_primitive,"parameters":next_parameters}
        graph_walk.append(entry)
        return graph_walk
        
    def generate_next_parameters(self,current_state,current_parameters,to_key,use_transition_model):
        """ Generate parameters for transitions.
        
        Parameters
        ----------
        * current_state: string
        \tName of the current motion primitive
        * current_parameters: np.ndarray
        \tParameters of the current state
        * to_key: string
        \t Name of the action and motion primitive we want to transition to. 
        \t Should have the format "action_motionprimitive" 
        * use_transition_model: bool
        \t flag to set whether a prediction from the transition model should be made or not.
        """
        splitted_key = to_key.split("_")
        action = splitted_key[0]
        assert action == self.elementary_action_name
        if  self.has_transition_models and use_transition_model:
            print "use transition model",current_state,to_key
            next_parameters = self.nodes[current_state].predict_parameters(to_key,current_parameters)
            
        else:
            motion_primitive = splitted_key[1]
            next_parameters = self.nodes[motion_primitive].sample_parameters()
        return next_parameters

    def save_updated_meta_info(self):
        """ Save updated meta data to a json file
        """
        if self.meta_information is not None:
            path = self.morphable_model_directory + os.sep + "meta_information.json"
            meta_info = self.meta_information
            write_to_json_file(path,meta_info)
        return        
        
class MorphableGraph(object):
    """ Contains a list of MorphableSubgraphs for each elementary action and
         transition models between them.
    """
     
    def __init__(self,morphable_model_directory,transition_model_directory, load_transition_models=False, update_stats=False):
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
        if morphable_model_directory != None:
            self.morphable_model_directory = morphable_model_directory
            
            for key in next(os.walk(morphable_model_directory))[1]:
                subgraph_path = morphable_model_directory+os.sep+key
                name = key.split("_")[-1]
                self.subgraphs[name] = MorphableSubgraph(name,subgraph_path,transition_model_directory, load_transition_models, update_stats)
                
        graph_definition_file = morphable_model_directory+os.sep+"graph_definition.json"
        
        #add transitions between subgraphs and load transition models           
        if os.path.isfile(graph_definition_file):
            graph_definition = load_json_file(graph_definition_file)
            transition_dict = graph_definition["transitions"]
            print "add transitions between subgraphs from",graph_definition_file," ##################################"
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
                                transition_model_file = transition_model_directory\
                                +os.sep+node_key+"_to_"+to_key+".GPM"
                                if load_transition_models and os.path.isfile(transition_model_file):
                                    output_gmm =  self.nodes[to_motion_primitive_name].mp.gmm
                                    transition_model = GPMixture.load(transition_model_file,\
                                    self.subgraphs[from_action_name].nodes[from_motion_primitive_name].mp.gmm,output_gmm)
                                else:
                                    transition_model = None
                                transition_type = "action_transition"
                               
                                edge = GraphEdge(from_action_name,from_motion_primitive_name,to_action_name,\
                                to_motion_primitive_name,transition_type,transition_model)
                                self.subgraphs[from_action_name].nodes[from_motion_primitive_name].outgoing_edges[to_key] = edge
                                
        else:
            print "did not find graph definition file",graph_definition_file," #####################"
#        #add output gmms for outgoing transiton models
#        if load_transition_models:
#            for key in self.subgraphs.keys():
#                for n in self.subgraphs[key].nodes.keys():
#                    for e in self.subgraphs[key].nodes[n].outgoing_edges.keys():
#                        edge =  self.subgraphs[key].nodes[n].outgoing_edges[e]
#                        if edge.transition_type == "action_transition":
#                            if edge.to_action in self.subgraphs.keys() and edge.to_motion_primitive in self.subgraphs[edge.to_action].nodes.keys():
#                                edge.transition_model.output_gmm = \
#                                    self.subgraphs[edge.to_action].nodes[edge.to_motion_primitive].mp.gmm
                            
        return

    @classmethod
    def init_from_dict(cls,subgraph_dict,subgraph_transitions = None):
        """ Intitializes the class using a dictionary of subgraphs
        
        Parameters
        ----------
        * subgraph_dict : dict
        \t A dictionary containing MorphableSubgraph instances
        * subgraph_transitions : dict
        \t A dictionary containing additional transitions between subgraphs
        
        Returns
        -------
        *morphable_graph: MorphableGraph
        \t The MorphableGraph instance
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
                print "\t"+"n canonical frames",self.subgraphs[s].nodes[n].mp.n_canonical_frames
                print "\t"+"n latent spatial dimensions",self.subgraphs[s].nodes[n].mp.s_pca["n_components"]
                print "\t"+"n latent time dimensions",self.subgraphs[s].nodes[n].mp.t_pca["n_components"]
                print "\t"+"n basis spatial ",self.subgraphs[s].nodes[n].mp.s_pca["n_basis"]
                print "\t"+"n basis time ",self.subgraphs[s].nodes[n].mp.t_pca["n_basis"]
                print "\t"+"n clusters",len(self.subgraphs[s].nodes[n].mp.gmm.weights_)
                print "\t"+"average length", self.subgraphs[s].nodes[n].average_step_length
                for e in self.subgraphs[s].nodes[n].outgoing_edges.keys():
                    print "\t \t to "+ e
                print "\t##########"       
                
                
def test_path():
    path = get_morphable_model_directory()
    print next(os.walk(path))[1]
    paths = []
    for root, dirs, files in  os.walk(path):
        for mm_file in files:
            paths.append(root+os.sep+mm_file)
    print paths

def print_morphable_graph_structure(morphable_graph):
    for s in morphable_graph.subgraphs.keys():
        print s
        for n in morphable_graph.subgraphs[s].nodes.keys():
            print "\t" + n
            print "\t" + "canonical frames", morphable_graph.subgraphs[s].nodes[n].mp.n_canonical_frames
            for e in morphable_graph.subgraphs[s].nodes[n].outgoing_edges.keys():
                print "\t \t to " + e
    return
    
    
def convert_graph_walk_to_euler_frames(bvh_reader, morphable_graph, graph_walk, node_name_map, smooth = True):
    """
     Converts a graph walk to a list of euler frames and concatenates the frames 
     to one list by aligning them at transitions.
     
     Parameters
    ----------
    * bvh_reader: BVHReader
    \tUsed for to extract the skeleton hierarchy information.
    * morphable_graph: MorphableGraph
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
        quaternion_frames = morphable_graph.subgraphs[subgraph].nodes[state].mp.back_project(parameters).get_motion_vector()
        euler_frames = convert_quaternion_to_euler(quaternion_frames.tolist())
        if concatenated_frames != None:
            concatenated_frames = align_frames(bvh_reader,concatenated_frames,euler_frames, node_name_map,smooth) 
        else:
            concatenated_frames = euler_frames
  
    return concatenated_frames

def export_graph_walk_to_bvh(bvh_reader, graph_walk, morphable_graph, concatenate=True , apply_smoothing= True, prefix= "",out_dir ="."):
    """Saves a graph walk as a single bvh file or as a list of bvh files
    
    """
    node_name_map = create_filtered_node_name_map(bvh_reader)
    if concatenate:
        concatenated_frames = convert_graph_walk_to_euler_frames(bvh_reader, morphable_graph, \
                                                graph_walk,node_name_map,apply_smoothing)
    
        filepath = out_dir+os.sep+prefix+"output_" + unicode(datetime.datetime.now().strftime("%d%m%y_%H%M%S"))+".bvh"
        BVHWriter(filepath,bvh_reader, concatenated_frames,bvh_reader.frame_time,is_quaternion=False)
    else:
        time_code = out_dir+os.sep+prefix+"output_" + unicode(datetime.datetime.now().strftime("%d%m%y_%H%M%S"))
        i = 0
        for entry in graph_walk:
            filepath = time_code+"_"+str(i)+".bvh"
            subgraph = entry["subgraph"]
            state = entry["state"]
            parameters = entry["parameters"]
            quaternion_frames = morphable_graph.subgraphs[subgraph].nodes[state].mp.back_project(parameters).get_motion_vector()
            euler_frames = convert_quaternion_to_euler(quaternion_frames.tolist())
            BVHWriter(filepath,bvh_reader, euler_frames,bvh_reader.frame_time,is_quaternion=False)
            i += 1
    
def test_morphable_graph(load_transition_models = False):
    np.random.seed(int(time.time()))
    skeleton_file = "skeleton.bvh"
    bvh_reader = BVHReader(skeleton_file)
    start = time.time()
    mm_directory =".."+ os.sep + get_morphable_model_directory()
    transition_directory =".."+ os.sep +  get_transition_model_directory()
    
    morphable_graph = MorphableGraph(mm_directory,transition_directory, load_transition_models)
    #morphable_graph = MorphableGraph(mm_directory,transition_directory)
    print_morphable_graph_structure(morphable_graph)
    print "loaded the morphable graph in",time.time()-start,"seconds"
    text_input = raw_input("Enter Data: ")

    n_steps = 10
    elementary_action = "walk"
    #graph_walk = morphable_graph.generate_random_walk(elementary_action,n_steps,load_transition_models)
    #export_graph_walk_to_bvh(bvh_reader, graph_walk, morphable_graph, True, True, prefix = "smoothed_",out_dir="random_walks")
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
            subgraph = MorphableSubgraph(name,subgraph_path,transition_directory, load_transition_models)
            break
    
    
    if subgraph != None:
        print "loaded the morphable sub graph for", elementary_action,"in",time.time()-start,"seconds"
    else:
        print elementary_action, "not found"
    return subgraph
    
    
def test_morphable_subgraph(elementary_action = "walk",load_transition_models = True, apply_smoothing = True):
  
    skeleton_file = "skeleton.bvh"
    bvh_reader = BVHReader(skeleton_file)
    subgraph = load_subgraph(elementary_action,load_transition_models)
    n_steps = 2
    if subgraph != None:
        start_state = subgraph.get_random_start_state()
       
        graph_walk = subgraph.generate_random_walk(start_state,n_steps,load_transition_models)
        print graph_walk
        subgraph_dict = {elementary_action:subgraph}
        morphable_graph = MorphableGraph.init_from_dict(subgraph_dict)
        #print print_morphable_graph_structure(morphable_graph)
        
        prefix = elementary_action+"_"
        if load_transition_models:
            prefix  += "from_transition_"
        export_graph_walk_to_bvh(bvh_reader, graph_walk, morphable_graph, True, apply_smoothing, prefix = prefix,out_dir="random_walks")
    
def main():
    #test_morphable_graph(load_transition_models=False)
    
    elementary_action= "pick"#"pick"
    load_transition_models = False
    apply_smoothing = True
    #test_morphable_subgraph(elementary_action,load_transition_models,apply_smoothing)
    test_morphable_graph(load_transition_models)


if __name__ ==  "__main__":
    #test_path()
    main()
    
    