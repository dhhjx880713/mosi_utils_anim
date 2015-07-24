# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:57:42 2015

@author: erhe01
"""

import os
import random
from . import NODE_TYPE_START, NODE_TYPE_STANDARD, NODE_TYPE_END
from utilities.io_helper_functions import load_json_file, \
                                         write_to_json_file

from GPMixture import GPMixture
from motion_primitive_node import MotionPrimitiveNode
from graph_edge import GraphEdge


 
class MotionPrimitiveGraph(object):
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
   
    def init_from_dict(self, subgraph_desc, graph_definition, transition_model_directory=None, load_transition_models=False):
        self.loaded_from_dict = True
        self.elementary_action_name = subgraph_desc["name"]

        for m_primitive in subgraph_desc["nodes"].keys():
             self.nodes[m_primitive] = MotionPrimitiveNode()
             self.nodes[m_primitive].init_from_dict(self.elementary_action_name,subgraph_desc["nodes"][m_primitive])
        
        if "info" in subgraph_desc.keys():
            self._set_meta_information(subgraph_desc["info"])
        else:
            self._set_meta_information() 
        self._set_transitions_from_dict(graph_definition, transition_model_directory, load_transition_models)
        self._update_attributes(update_stats=False)
        return

    def init_from_directory(self,elementary_action_name, morphable_model_directory,transition_model_directory, load_transition_models=False, update_stats=False):
        self.loaded_from_dict = False
        self.elementary_action_name = elementary_action_name
        self.nodes = {}
        self.morphable_model_directory = morphable_model_directory
   
        #load morphable models
        temp_file_list =  []#for files containing additional information that require the full graph to be constructed first
        meta_information = None
        self.annotation_map = {}
        for root, dirs, files in os.walk(morphable_model_directory):
            for file_name in files:#for each morphable model 
                if file_name == "meta_information.json":
                    meta_information = load_json_file(morphable_model_directory+os.sep+file_name)
                    print "found meta information"
                elif file_name.endswith("mm.json"):
                    print "found motion primitive",file_name  
                    motion_primitive_name = file_name.split("_")[1]  
                    #print motion_primitve_file_name
                    motion_primitive_file_name = morphable_model_directory+os.sep+file_name
                    self.nodes[motion_primitive_name] = MotionPrimitiveNode()
                    self.nodes[motion_primitive_name].init_from_file(elementary_action_name,motion_primitive_name,motion_primitive_file_name)
                    
                elif file_name.endswith(".stats"):
                    print "found stats",file_name
                    temp_file_list.append(file_name)

                else:
                    print "ignored",file_name

        self._set_meta_information(meta_information)
        
        #load information about training data if available
        for file_name in temp_file_list:
            motion_primitive = file_name.split("_")[1][:-6]
            if motion_primitive in self.nodes.keys():
                info = load_json_file(morphable_model_directory+os.sep+file_name,use_ordered_dict=True)
                self.nodes[motion_primitive].parameter_bb = info["pose_bb"]
                self.nodes[motion_primitive].cartesian_bb = info["cartesian_bb"]
                self.nodes[motion_primitive].velocity_data = info["pose_velocity"]

      
               
        self._set_transitions_from_directory(morphable_model_directory, transition_model_directory, load_transition_models)
       
        self._update_attributes(update_stats=update_stats)     


    def _set_transitions_from_directory(self, morphable_model_directory, transition_model_directory, load_transition_models=False):
        """
        Define transitions and load transiton models.
        """
        

        self.has_transition_models = load_transition_models
        if os.path.isfile(morphable_model_directory+os.sep+".."+os.sep+"graph_definition.json"):
            graph_definition = load_json_file(morphable_model_directory+os.sep+".."+os.sep+"graph_definition.json")
            self._set_transitions_from_dict(graph_definition, transition_model_directory, load_transition_models)
         
        else:
            print "Warning: no transitions were found in the directory"
                    
        return
        
    def _set_transitions_from_dict(self,graph_definition, transition_model_directory=None, load_transition_models=False):
        """Define transitions  and load transiton models
            TODO split once into tuples when loaded
            TODO factor out into its own class
        """
        transition_dict = graph_definition["transitions"]
        for node_key in transition_dict:
            from_action_name = node_key.split("_")[0]
            from_motion_primitive_name = node_key.split("_")[1]
            if from_action_name == self.elementary_action_name and \
               from_motion_primitive_name in self.nodes.keys():
                for to_key in transition_dict[node_key]:
                    to_action_name = to_key.split("_")[0]
                    to_motion_primitive_name = to_key.split("_")[1]
                    
                    if to_action_name == self.elementary_action_name:
                        transition_model = None
                        if transition_model_directory is not None and load_transition_models:
                            transition_model_file = transition_model_directory\
                            +os.sep+node_key+"_to_"+to_key+".GPM"
                            if  os.path.isfile(transition_model_file):
                                output_gmm = self.nodes[to_motion_primitive_name].motion_primitive.gmm
                                transition_model = GPMixture.load(transition_model_file,\
                                self.nodes[from_motion_primitive_name].motion_primitive.gmm,output_gmm)
                            else:
                                print "did not find transition model file",transition_model_file
                            
                        if self.nodes[to_motion_primitive_name].node_type in [NODE_TYPE_START,NODE_TYPE_STANDARD]: 
                            transition_type = "standard"
                        else:
                            transition_type = "end"
                        print "add",transition_type
                        edge = GraphEdge(self.elementary_action_name,node_key,to_action_name,\
                        to_motion_primitive_name,transition_type,transition_model)
                        self.nodes[from_motion_primitive_name].outgoing_edges[to_key] = edge  
        return
        
    def _set_meta_information(self, meta_information=None):
        """
        Identify start and end states from meta information.
        """
        
        self.meta_information = meta_information
        
        if self.meta_information is None:
            self.start_states = [k for k in self.nodes.keys() if k.startswith("begin") or k == "first"]
            self.end_states = [k for k in self.nodes.keys() if k.startswith("end") or k == "second"]
        else:
            for key in ["annotations", "start_states", "end_states" ]:
                assert key in self.meta_information.keys() 

            self.start_states = self.meta_information["start_states"]
            self.end_states = self.meta_information["end_states"]
            self.motion_primitive_annotations = self.meta_information["annotations"]
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
            self.nodes[k].node_type = NODE_TYPE_START
        print "end states",self.end_states
        for k in self.end_states:
            self.nodes[k].node_type = NODE_TYPE_END
             
        return
                      
        
    def _update_attributes(self, update_stats=False):
        """
        Update attributes of motion primitives for faster lookup. #
        """
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
        if changed_meta_info and not self.loaded_from_dict:
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
        
    def generate_random_walk(self,start_state, number_of_steps, use_transition_model=True):
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
                to_key = self.nodes[current_state].generate_random_transition(NODE_TYPE_STANDARD) 
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
        to_key = self.nodes[current_state].generate_random_transition(NODE_TYPE_END)
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
        
        
    def get_canonical_keyframe_labels(self, motion_primitive_name):
        if motion_primitive_name in self.motion_primitive_annotations.keys():
            keyframe_labels = self.motion_primitive_annotations[motion_primitive_name]
        else:
            keyframe_labels = {}
        return keyframe_labels



        
        
    def get_random_transition(self, motion, action_constraint, travelled_arc_length, arc_length_of_end):
        """ Get next state of the elementary action based on previous iteration.
        """
        prev_mp_name = motion.graph_walk[-1].motion_primitive_name
            
        if action_constraint.trajectory is not None :
                
             #test end condition for trajectory constraints
            if not action_constraint.check_end_condition(motion.quat_frames,\
                                    travelled_arc_length,arc_length_of_end):            

                #make standard transition to go on with trajectory following
                next_mp_type = NODE_TYPE_STANDARD
            else:
                # threshold was overstepped. remove previous step before 
                # trying to reach the goal using a last step
                #TODO replace with more efficient solution or optimization
    
                next_mp_type = NODE_TYPE_END
                
            print "generate",next_mp_type,"transition from trajectory"
        else:
            n_standard_transitions = len([e for e in self.nodes[prev_mp_name].outgoing_edges.keys() if self.nodes[prev_mp_name].outgoing_edges[e].transition_type == NODE_TYPE_STANDARD])
            if n_standard_transitions > 0:
                next_mp_type = NODE_TYPE_STANDARD
            else:
                next_mp_type = NODE_TYPE_END
            print "generate",next_mp_type,"transition without trajectory",n_standard_transitions
    
        to_key = self.nodes[prev_mp_name].generate_random_transition(next_mp_type)
        
        if to_key is not None:
            current_motion_primitive = to_key.split("_")[1]
            return current_motion_primitive, next_mp_type
        else:
            return None, next_mp_type
           
