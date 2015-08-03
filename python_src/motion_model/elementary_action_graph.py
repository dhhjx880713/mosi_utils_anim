# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:57:51 2015

@author: erhe01
"""


import collections


        
class ElementaryActionGraph(object):
    """ Contains a list of MotionPrimitiveGraphs for each elementary action and
         transition models between them.
    """
     
    def __init__(self):
        """ Initializes the class
        """
       
        self.skeleton = None
        self.subgraphs = collections.OrderedDict()

        
 

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
    