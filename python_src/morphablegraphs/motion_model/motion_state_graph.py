# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:57:51 2015

@author: erhe01
"""

import collections
from ..utilities import write_message_to_log, LOG_MODE_DEBUG
import random


class MotionStateGraph(object):
    """ Contains a dict of MotionPrimitiveNodes and MotionPrimitiveNodeGroups for each elementary action,
         transitions between them are stored as outgoing edges in the nodes.
    """
     
    def __init__(self):
        """ Initializes the class
        """
        self.skeleton = None
        self.mgrd_skeleton = None
        self.node_groups = collections.OrderedDict()
        self.nodes = collections.OrderedDict()
        self.hand_pose_generator = None
        self.animated_joints = None

    def generate_random_walk(self, start_action, number_of_steps, use_transition_model=True):
        """ Generates a random graph walk
        Parameters
        ----------
        * start_action: string
            Initial action.
        * number_of_steps: integer
            Number of transitions
        * use_transition_model: bool
            Sets whether or not the transition model should be used in parameter prediction
        Returns
        -------
        *graph_walk: a list of dictionaries
            The graph walk is defined by a list of dictionaries containing entries for "action","motion primitive" and "parameters"
        """
        assert start_action in self.node_groups.keys()
        print "generate random graph walk for", start_action
        start_state = self.node_groups[start_action].get_random_start_state()
        print start_state
        return self.node_groups[start_action].generate_random_walk(self.nodes, start_state, number_of_steps, use_transition_model)
    
    def print_information(self):
        """
        Prints out information on the graph structure and properties of the motion primitives
        """
        for s in self.node_groups.keys():
            print s
            for n in self.node_groups[s].nodes.keys():
                print "\t"+ str(n)
                print "\t"+"n canonical frames", self.nodes[n].n_canonical_frames
                print "\t"+"n latent spatial dimensions", self.nodes[n].s_pca["n_components"]
                print "\t"+"n latent time dimensions", self.nodes[n].t_pca["n_components"]
                print "\t"+"n basis spatial ", self.nodes[n].s_pca["n_basis"]
                print "\t"+"n basis time ", self.nodes[n].t_pca["n_basis"]
                print "\t"+"n clusters", len(self.nodes[n].gaussian_mixture_model.weights_)
                print "\t"+"average length", self.nodes[n].average_step_length
                for e in self.nodes[n].outgoing_edges.keys():
                    print "\t \t to " + str(e)
                print "\t##########"       

    def get_random_action_transition(self, graph_walk, action_name, is_cycle=False):
        """ Get random start state based on edge from previous elementary action if possible
        """
        next_node = None
        if graph_walk.step_count > 0:
            prev_node_key = graph_walk.steps[-1].node_key
      
            if prev_node_key in self.nodes.keys():
                next_node = self.nodes[prev_node_key].generate_random_action_transition(action_name,is_cycle)
            write_message_to_log("Generate start from transition of last action " + str(prev_node_key) + str(next_node), mode=LOG_MODE_DEBUG)
        # if there is no previous elementary action or no action transition
        #  use transition to random start state
        if next_node is None or next_node not in self.node_groups[action_name].nodes:
            #print next_node, "not in", action_name
            next_node = self.node_groups[action_name].get_random_start_state()
            write_message_to_log("Generate random start" + str(next_node), mode=LOG_MODE_DEBUG)
        return next_node

    def get_start_nodes(self, action_name):
        """
        Get all start state based on edge from previous elementary action if possible
        :param action_name:
        :return:
        """
        return self.node_groups[action_name].get_start_states()

    def get_random_start_node(self):
        """ If there are start sates defined in the graph a node key tuple is returned  (action_name, state_name)
            else None is returned.
        """
        actions = self.node_groups.keys()
        if len(actions) > 0:
            random_index = random.randrange(0, len(actions), 1)
            action_name = actions[random_index]
            start_states = self.get_start_nodes(action_name)
            if len(start_states) > 0:
                random_index = random.randrange(0, len(start_states), 1)
                state_name = start_states[random_index]
                return action_name, state_name
        return None
