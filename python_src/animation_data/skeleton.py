# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:18:37 2015

@author: Erik Herrmann, Martin Manns
"""

import collections
import numpy as np

class Skeleton(object):
    """ Data structure that stores the skeleton hierarchy information 
        extracted from a BVH file with additional meta information.
    """
    def __init__(self, bvh_reader):
        self.frame_time = bvh_reader.frame_time
        self.root = bvh_reader.root
        self.node_names = bvh_reader.node_names
        self.max_level = max([node["level"] for node in
                                  self.node_names.values()
                                  if "level" in node.keys()])
        self.parent_dict = self._get_parent_dict()
        self._create_filtered_node_name_map()
        self._set_joint_weights()
        
    def _get_parent_dict(self):
        """Returns a dict of node names to their parent node's name"""

        parent_dict = {}

        for node_name in self.node_names:
            if "children" in self.node_names[node_name].keys():
                for child_node in self.node_names[node_name]["children"]:
                    parent_dict[child_node] = node_name

        return parent_dict
        
    def gen_all_parents(self, node_name):
        """Generator of all parents' node names of node with node_name"""

        while node_name in self.parent_dict:
            parent_name = self.parent_dict[node_name]
            yield parent_name
            node_name = parent_name

    
    
    def _set_joint_weights(self):
        """ Gives joints weights according to their distance in the joint hiearchty
           to the root joint. The further away the smaller the weight.
        """

        self.joint_weights = [np.exp(-self.node_names[node_name]["level"]) \
                for node_name in self.node_name_map.keys()]
     
    def _create_filtered_node_name_map(self):
        """
        creates dictionary that maps node names to indices in a frame vector
        without "Bip" joints
        """
        self.node_name_map = collections.OrderedDict()
        j = 0
        for node_name in self.node_names:
            if not node_name.startswith("Bip") and \
                "children" in self.node_names[node_name].keys():
                self.node_name_map[node_name] = j
                j += 1
