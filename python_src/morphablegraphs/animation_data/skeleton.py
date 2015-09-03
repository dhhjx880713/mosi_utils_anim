# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:18:37 2015

@author: Erik Herrmann, Martin Manns
"""

import collections
import numpy as np
from ..external.transformations import quaternion_matrix
from itertools import izip


class Skeleton(object):
    """ Data structure that stores the skeleton hierarchy information
        extracted from a BVH file with additional meta information.
    """
    def __init__(self, bvh_reader):
        self.frame_time = bvh_reader.frame_time
        self.root = bvh_reader.root
        self.node_names = bvh_reader.node_names
        self._create_filtered_node_name_map()
        self._add_tool_bones()
        self.max_level = max([node["level"] for node in
                      self.node_names.values()
                      if "level" in node.keys()])
        self._set_joint_weights()
        self.parent_dict = self._get_parent_dict()
        print "node name map keys", self.node_name_map.keys(), len(self.node_name_map)

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

        # self.joint_weights = [np.exp(-self.node_names[node_name]["level"])
        #                       for node_name in self.node_name_map.keys()]
        self.joint_weights = [1.0/(self.node_names[node_name]["level"] + 1.0) \
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

    def get_joint_weights(self):
        return self.joint_weights

    def _add_tool_bones(self):
        new_node_name = 'LeftToolEndSite'
        parent_node_name = 'LeftHand'
        new_node_offset = [9.55928, -0.145352, -0.186424]
        self._add_new_end_site(new_node_name, parent_node_name, new_node_offset)
        new_node_name = 'RightToolEndSite'
        parent_node_name = 'RightHand'
        new_node_offset = [9.559288, 0.145353, -0.186417]
        self._add_new_end_site(new_node_name, parent_node_name, new_node_offset)
        #Finger21 = 'Bip01_L_Finger21'
        #Finger21_offset = [3.801407, 0.0, 0.0]
        #Finger2 = 'Bip01_R_Finger2'
        #Finger21 = 'Bip01_R_Finger21'
        #Finger2_offset = [9.559288, 0.145353, -0.186417]
        #Finger21_offset = [3.801407, 0.0, 0.0]

    def _add_new_end_site(self, new_node_name, parent_node, offset):
        if parent_node in self.node_name_map.keys():
            level = self.node_names[parent_node]["level"] + 1
            node_desc = dict()
            node_desc["level"] = level
            node_desc["offset"] = offset
            self.node_names[parent_node]["children"].append(new_node_name)
            self.node_names[new_node_name] = node_desc
            self.node_name_map[new_node_name] = -1 #the nodes needs an entry but the index is only important if it has children

    def get_cartesian_coordinates_from_quaternion(self,
                                                  node_name,
                                                  quaternion_frame,
                                                  return_global_matrix=False):
        """Returns cartesian coordinates for one node at one frame. Modified to
         handle frames with omitted values for joints starting with "Bip"

        Parameters
        ----------

        * node_name: String
        \tName of node
         * skeleton: Skeleton
        \tBVH data structure read from a file
        * node_name_map: dict
        \tA map from node name to index in the euler frame

        """
        if self.node_names[node_name]["level"] == 0:
            root_frame_position = quaternion_frame[:3]
            root_node_offset = self.node_names[node_name]["offset"]

            return [t + o for t, o in
                    izip(root_frame_position, root_node_offset)]

        else:
            # Names are generated bottom to up --> reverse
            chain_names = list(self.gen_all_parents(node_name))
            chain_names.reverse()
            chain_names += [node_name]  # Node is not in its parent list

            offsets = [self.node_names[node_name]["offset"]
                       for node_name in chain_names]
            root_position = quaternion_frame[:3].flatten()
            offsets[0] = [r + o for r, o in izip(root_position, offsets[0])]

            j_matrices = []
            count = 0
            for node_name in chain_names:
                if "children" in self.node_names[node_name].keys():  # check if it is a joint or an end site
                    index = self.node_name_map[node_name] * 4 + 3
                    j_matrix = quaternion_matrix(quaternion_frame[index: index + 4])
                    j_matrix[:, 3] = offsets[count] + [1]
                else:
                    print node_name
                    j_matrix = np.identity(4)
                    j_matrix[:, 3] = offsets[count] + [1]
                    break # there should not be any nodes after an end site
                j_matrices.append(j_matrix)
                count += 1

            global_matrix = np.identity(4)
            for j_matrix in j_matrices:
                global_matrix = np.dot(global_matrix, j_matrix)
            if return_global_matrix:
                return global_matrix
            else:
                point = np.array([0, 0, 0, 1])
                point = np.dot(global_matrix, point)
                return point[:3].tolist()

    def convert_quaternion_frame_to_cartesian_frame(self, quat_frame):
        """
        Converts quaternion frames to cartesian frames by calling get_cartesian_coordinates_from_quaternion for each joint
        """
        cartesian_frame = []
        for node_name in self.node_name_map.keys():
            cartesian_frame.append(self.get_cartesian_coordinates_from_quaternion(node_name, quat_frame))

        return cartesian_frame
