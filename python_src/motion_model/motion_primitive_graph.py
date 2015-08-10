# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:57:51 2015

@author: erhe01
"""


import collections


        
class MotionPrimitiveGraph(object):
    """ Contains a dict of MotionPrimitiveNodes and MotionPrimitiveNodeGroups for each elementary action,
         transitions between them are stored as outgoing edges in the nodes.
    """
     
    def __init__(self):
        """ Initializes the class
        """
        self.skeleton = None
        self.node_groups = collections.OrderedDict()
        self.nodes = collections.OrderedDict()

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
        assert start_subgraph in self.node_groups.keys()
        print "generate random graph walk for",start_subgraph
        current_subgraph = start_subgraph
        start_state = self.node_groups[current_subgraph].get_random_start_state()
        return self.node_groups[current_subgraph].generate_random_walk(self.nodes, start_state, number_of_steps, use_transition_model)
    
    def print_information(self):
        """
        Prints out information on the graph structure and properties of the motion primitives
        """
        for s in self.node_groups.keys():
            print s
            for n in self.node_groups[s].nodes.keys():
                print "\t"+ n
                print "\t"+"n canonical frames",self.nodes[n].motion_primitive.n_canonical_frames
                print "\t"+"n latent spatial dimensions",self.nodes[n].motion_primitive.s_pca["n_components"]
                print "\t"+"n latent time dimensions",self.nodes[n].motion_primitive.t_pca["n_components"]
                print "\t"+"n basis spatial ",self.nodes[n].motion_primitive.s_pca["n_basis"]
                print "\t"+"n basis time ",self.nodes[n].motion_primitive.t_pca["n_basis"]
                print "\t"+"n clusters",len(self.nodes[n].motion_primitive.gmm.weights_)
                print "\t"+"average length", self.nodes[n].average_step_length
                for e in self.nodes[n].outgoing_edges.keys():
                    print "\t \t to "+ e
                print "\t##########"       
                
                
    def get_random_action_transition(self, motion, action_name):
        """ Get random start state based on edge from previous elementary action if possible
        """
        next_state = None
        if motion.step_count > 0:
            prev_action_name = motion.graph_walk[-1].action_name
            prev_mp_name = motion.graph_walk[-1].motion_primitive_name
      
            if (prev_action_name,prev_mp_name) in self.nodes.keys():
                                       
               to_key = self.nodes[(prev_action_name,prev_mp_name)].generate_random_action_transition(action_name)
               if to_key is not None:
                   next_state = to_key
                   return next_state
               else:
                   return None
               print "generate start from transition of last action", prev_action_name, prev_mp_name, to_key
           
        # if there is no previous elementary action or no action transition
        #  use transition to random start state
        if next_state == "" or next_state not in self.node_groups[action_name].nodes:
            print next_state,"not in", action_name#,prev_action_name,prev_mp_name
            next_state = self.node_groups[action_name].get_random_start_state()
            print "generate random start",next_state
            return next_state


def print_morphable_graph_structure(morphable_graph):
    for a in morphable_graph.node_groups.keys():
        print a
        for n in morphable_graph.node_groups[a].nodes:
            print "\t" + n
            print "\t" + "canonical frames", morphable_graph.nodes[(a, n)].motion_primitive.n_canonical_frames
            for e in morphable_graph.nodes[(a, n)].outgoing_edges.keys():
                print "\t \t to " + e
    return
    