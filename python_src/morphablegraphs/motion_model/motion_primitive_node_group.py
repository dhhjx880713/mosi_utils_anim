# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:57:42 2015

@author: erhe01
"""

import os
import random
from . import NODE_TYPE_START, NODE_TYPE_STANDARD, NODE_TYPE_END
from ..utilities.io_helper_functions import write_to_json_file

 
class MotionPrimitiveNodeGroup(object):
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
   

    def _set_meta_information(self, meta_information=None):
        """
        Identify start and end states from meta information.
        """
        
        self.meta_information = meta_information
        
        if self.meta_information is None:
            self.start_states = [k[1] for k in self.nodes.keys() if k[1].startswith("begin") or k[1] == "first"]
            self.end_states = [k[1] for k in self.nodes.keys() if k[1].startswith("end") or k[1] == "second"]
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
             
        self._set_node_attributes()
        return
         
    def _set_node_attributes(self):
        print "elementary_action",self.elementary_action_name     
        print "start states",self.start_states
        for k in self.start_states:
            self.nodes[(self.elementary_action_name, k)].node_type = NODE_TYPE_START
        print "end states",self.end_states
        for k in self.end_states:
            self.nodes[(self.elementary_action_name, k)].node_type = NODE_TYPE_END
                          
        
    def update_attributes(self, update_stats=False):
        """
        Update attributes of motion primitives for faster lookup. #
        """
        changed_meta_info = False
        if update_stats:
            changed_meta_info = True
            self.meta_information["stats"] = {}
            for k in self.nodes.keys():
                 self.nodes[k].update_attributes()
                 self.meta_information["stats"][k[1]]={"average_step_length":self.nodes[k].average_step_length,"n_standard_transitions": self.nodes[k].n_standard_transitions }
                 print"n standard transitions",k,self.nodes[k].n_standard_transitions
            print "updated meta information",self.meta_information
        else:
            if self.meta_information is None:
                self.meta_information = {}
            if "stats" not in self.meta_information.keys():
                self.meta_information["stats"] = {}
            for k in self.nodes.keys():
                if k[1] in self.meta_information["stats"].keys():
                    self.nodes[k].n_standard_transitions = self.meta_information["stats"][k[1]]["n_standard_transitions"] 
                    self.nodes[k].average_step_length = self.meta_information["stats"][k[1]]["average_step_length"]
                else:
                    self.nodes[k].update_attributes()
                    self.meta_information["stats"][k[1]]={"average_step_length":self.nodes[k].average_step_length,"n_standard_transitions": self.nodes[k].n_standard_transitions }
                    changed_meta_info = True
            print "loaded stats from meta information file",self.meta_information
        if changed_meta_info and not self.loaded_from_dict:
            self.save_updated_meta_info()
            
    def get_random_start_state(self):
        """ Returns the name of a random start state. """
        random_index = random.randrange(0, len(self.start_states), 1)
        start_state = (self.elementary_action_name, self.start_states[random_index])
        return start_state
            
    def get_random_end_state(self):
        """ Returns the name of a random start state."""
        random_index = random.randrange(0, len(self.end_states), 1)
        start_state = (self.elementary_action_name, self.end_states[random_index])
        return start_state
        
    def generate_random_walk(self, start_state, number_of_steps, use_transition_model=True):
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
                next_parameters = self.generate_next_parameters(self.nodes, current_state,current_parameters,to_key,use_transition_model)
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
        
    def generate_next_parameters(self, current_state, current_parameters, to_key, use_transition_model):
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

    def _convert_keys_to_strings(self, mydict):
        copy_dict = {}
        for key in mydict.keys():
              if type(key) is tuple:
                try:
                  copy_dict[key[1]] = mydict[key]
                except:
                    continue
              else:
                   copy_dict[key] =  mydict[key]
        return copy_dict

    def save_updated_meta_info(self):
        """ Save updated meta data to a json file
        """
        if self.meta_information is not None:
            path = self.morphable_model_directory + os.sep + "meta_information.json"
            write_to_json_file(path, self._convert_keys_to_strings(self.meta_information))
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
        
        prev_state = motion.graph_walk[-1].node_key
            
        if action_constraint.trajectory is not None:
                
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
            n_standard_transitions = len([e for e in self.nodes[prev_state].outgoing_edges.keys() if self.nodes[prev_state].outgoing_edges[e].transition_type == NODE_TYPE_STANDARD])
            if n_standard_transitions > 0:
                next_mp_type = NODE_TYPE_STANDARD
            else:
                next_mp_type = NODE_TYPE_END
            print "generate",next_mp_type,"transition without trajectory",n_standard_transitions
    
        to_key = self.nodes[prev_state].generate_random_transition(next_mp_type)
        
        if to_key is not None:
            print to_key
            return to_key, next_mp_type
        else:
            return None, next_mp_type
           
