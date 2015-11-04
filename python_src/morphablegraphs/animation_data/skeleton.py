# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:18:37 2015

@author: Erik Herrmann, Martin Manns
"""

import collections
from copy import deepcopy
import numpy as np
from ..external.transformations import quaternion_matrix
from quaternion_frame import QuaternionFrame
from itertools import izip
from skeleton_node import SkeletonRootNode, SkeletonJointNode, SkeletonEndSiteNode, ROTATION_TYPE_QUAT, ROTATION_TYPE_EULER


class Skeleton(object):
    """ Data structure that stores the skeleton hierarchy information
        extracted from a BVH file with additional meta information.
    """
    def __init__(self, bvh_reader):
        self.frame_time = deepcopy(bvh_reader.frame_time)
        self.root = deepcopy(bvh_reader.root)
        self.node_names = deepcopy(bvh_reader.node_names)
        self.reference_frame = self._extract_reference_frame(bvh_reader)
        self.joint_map = None
        self._create_filtered_node_name_frame_map()
        self.tool_bones = []
        self._add_tool_bones()
        self.max_level = self._get_max_level()
        self._set_joint_weights()
        self.parent_dict = self._get_parent_dict()
        self._chain_names = self._generate_chain_names()
        self.construct_hierarchy_iterative()
        #print "node name map keys", len(self.node_names), len(self.node_name_frame_map), self.node_name_frame_map.keys()
        print "shape of reference frame", (self.reference_frame.shape)
        #print "number of parameters", self.get_number_of_frame_parameters(ROTATION_TYPE_QUAT)
        #for joint in self.joint_map.keys():
        #    print joint, self.joint_map[joint].index

    def _extract_reference_frame(self, bvh_reader):
        quaternion_frame = np.array((QuaternionFrame(bvh_reader, bvh_reader.frames[0], False, False).values())).flatten()
        return np.array(bvh_reader.frames[0][:3].tolist() + quaternion_frame.tolist())

    def construct_hierarchy_iterative(self):
        index = 0
        self.joint_map = collections.OrderedDict()
        for node_name in self.node_names.keys():
            if "children" in self.node_names[node_name].keys():
                is_endsite = len(self.node_names[node_name]["children"]) <= 0
            else:
                is_endsite = True

            if node_name == self.root:
                node = SkeletonRootNode(node_name, None)
                node.index = index
                index += 1
            elif not is_endsite:
                node = SkeletonJointNode(node_name, None)
                node.index = index
                index += 1
            else:
                node = SkeletonEndSiteNode(node_name, None)

            if node_name in self.node_name_frame_map and self.node_name_frame_map[node_name] >= 0:
                node.quaternion_frame_index = self.node_name_frame_map[node_name] * 4 + 3

            node.offset = self.node_names[node_name]["offset"]

            if node_name in self.parent_dict.keys():
                parent_node_name = self.parent_dict[node_name]
                if parent_node_name in self.joint_map.keys():
                    node.parent = self.joint_map[parent_node_name]
                    self.joint_map[parent_node_name].children.append(node)

            self.joint_map[node_name] = node

    def construct_hierarchy(self):
        self.root_node = SkeletonRootNode(self.root, None)
        self.root_node.index = 0
        self.root_node.quaternion_frame_index = 3
        self.joint_map = collections.OrderedDict()
        self.joint_map[self.root] = self.root_node
        self.add_skeleton_node(self.root_node)
        # print "joints", self.joint_map.keys()

    def add_skeleton_node(self, parent_node):
        """ Currently assumes there are no EndSites
        :param parent_node:
        :return:
        """
        for child_name in self.node_names[parent_node.node_name]["children"]:
            child_node = SkeletonJointNode(child_name, parent_node)
            child_node.index = len(self.joint_map.keys())
            child_node.offset = self.node_names[child_name]["offset"]
            if child_name in self.node_name_frame_map:
                child_node.quaternion_frame_index = self.node_name_frame_map[child_name] * 4 + 3
            if "children" in self.node_names[child_name].keys() and len(self.node_names[child_name]["children"]):
                self.add_skeleton_node(child_node)
            self.joint_map[child_name] = child_node
            parent_node.children.append(child_node)

    def is_motion_vector_complete(self, frames, is_quaternion):
        if is_quaternion:
            rotation_type = ROTATION_TYPE_QUAT
        else:
            rotation_type = ROTATION_TYPE_EULER
        return len(frames[0]) == self.get_number_of_frame_parameters(rotation_type)

    def get_number_of_frame_parameters(self, rotation_type):
        n_parameters = 0
        for node_name in self.joint_map.keys():
            if node_name not in self.tool_bones:
                local_parameters = self.joint_map[node_name].get_number_of_frame_parameters(rotation_type)
                n_parameters += local_parameters

        return n_parameters

    def complete_motion_vector_from_reference(self, quat_frames):
        new_quat_frames = []
        for frame in quat_frames:
            new_quat_frames.append(self.complete_frame_vector_from_reference(frame))
        return new_quat_frames

    def complete_frame_vector_from_reference(self, frame):
        new_frame = []
        for joint_name in self.joint_map.keys():
            if joint_name not in self.tool_bones:
                if joint_name in self.node_name_frame_map.keys() and self.node_name_frame_map[joint_name] > -1:
                    if joint_name == self.root:
                        joint_parameters = frame[:3].tolist()
                    else:
                        joint_parameters = []
                    joint_parameters += frame[self.joint_map[joint_name].quaternion_frame_index:self.joint_map[joint_name].quaternion_frame_index + 4].tolist()
                    #joint_parameters = self.joint_map[joint_name].get_frame_parameters(frame, ROTATION_TYPE_QUAT)
                else:
                    joint_parameters = self.joint_map[joint_name].get_frame_parameters(self.reference_frame, ROTATION_TYPE_QUAT)
                    #print "from reference", joint_name,joint_parameters
                if joint_parameters is not None:
                    #print joint_name, joint_parameters
                    new_frame += joint_parameters
        #print "frame length", len(new_frame)
        return new_frame

    def _get_max_level(self):
        return max([node["level"] for node in
                      self.node_names.values()
                      if "level" in node.keys()])

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
        #                       for node_name in self.node_name_frame_map.keys()]
        self.joint_weights = [1.0/(self.node_names[node_name]["level"] + 1.0) for node_name in self.node_name_frame_map.keys()]
        self.joint_weight_map = collections.OrderedDict()
        weight_index = 0
        for node_name in self.node_name_frame_map.keys():
            self.joint_weight_map[node_name] = self.joint_weights[weight_index]
            weight_index += 1
        self.joint_weight_map["RightHand"] = 2.0
        self.joint_weight_map["LeftHand"] = 2.0
        self.joint_weights = self.joint_weight_map.values()

    def _create_filtered_node_name_frame_map(self):
        """
        creates dictionary that maps node names to indices in a frame vector
        without "Bip" joints
        """
        self.node_name_frame_map = collections.OrderedDict()
        j = 0
        for node_name in self.node_names:
            if not node_name.startswith("Bip") and \
                    "children" in self.node_names[node_name].keys():
                self.node_name_frame_map[node_name] = j
                j += 1

    def get_joint_weights(self):
        #return self.joint_weights#.values()
        return self.joint_weight_map.values()

    def _add_tool_bones(self):
        new_node_name = 'LeftToolEndSite'
        parent_node_name = 'LeftHand'
        new_node_offset = [9.55928, -0.145352, 5.186424]
        self._add_new_end_site(new_node_name, parent_node_name, new_node_offset)
        self.tool_bones.append(new_node_name)
        new_node_name = 'RightToolEndSite'
        parent_node_name = 'RightHand'
        new_node_offset = [9.559288, 0.145353, 5.186417]
        self._add_new_end_site(new_node_name, parent_node_name, new_node_offset)
        self.tool_bones.append(new_node_name)
        #Finger21 = 'Bip01_L_Finger21'
        #Finger21_offset = [3.801407, 0.0, 0.0]
        #Finger2 = 'Bip01_R_Finger2'
        #Finger21 = 'Bip01_R_Finger21'
        #Finger2_offset = [9.559288, 0.145353, -0.186417]
        #Finger21_offset = [3.801407, 0.0, 0.0]

    def _add_new_end_site(self, new_node_name, parent_node, offset):
        if parent_node in self.node_name_frame_map.keys():
            level = self.node_names[parent_node]["level"] + 1
            node_desc = dict()
            node_desc["level"] = level
            node_desc["offset"] = offset
            self.node_names[parent_node]["children"].append(new_node_name)
            self.node_names[new_node_name] = node_desc
            self.node_name_frame_map[new_node_name] = -1 #the node needs an entry but the index is only important if it has children

    def _generate_chain_names(self):
        chain_names = dict()
        for node_name in self.node_name_frame_map.keys():
            chain_names[node_name] = list(self.gen_all_parents(node_name))
            # Names are generated bottom to up --> reverse
            chain_names[node_name].reverse()
            chain_names[node_name] += [node_name]  # Node is not in its parent list
        return chain_names


    def get_cartesian_coordinates_from_quaternion(self,
                                                  target_node_name,
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

        """
        if self.node_names[target_node_name]["level"] == 0:
            root_frame_position = quaternion_frame[:3]
            root_node_offset = self.node_names[target_node_name]["offset"]
            return [t + o for t, o in
                    izip(root_frame_position, root_node_offset)]
        else:
            offsets = [self.node_names[node_name]["offset"]
                       for node_name in self._chain_names[target_node_name]]
            root_position = quaternion_frame[:3].flatten()
            offsets[0] = [r + o for r, o in izip(root_position, offsets[0])]
            j_matrices = []
            count = 0
            for node_name in self._chain_names[target_node_name]:
                if "children" in self.node_names[node_name].keys():  # check if it is a joint or an end site
                    index = self.node_name_frame_map[node_name] * 4 + 3
                    j_matrix = quaternion_matrix(quaternion_frame[index: index + 4])
                    j_matrix[:, 3] = offsets[count] + [1]
                else:
                    #print node_name, self._chain_names[target_node_name][count-1], offsets[count]
                    j_matrix = np.identity(4)
                    j_matrix[:, 3] = offsets[count] + [1]
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
                #print target_node_name, "position", point
                return point[:3].tolist()

    def convert_quaternion_frame_to_cartesian_frame(self, quat_frame):
        """
        Converts quaternion frames to cartesian frames by calling get_cartesian_coordinates_from_quaternion for each joint
        """
        cartesian_frame = []
        for node_name in self.node_name_frame_map.keys():
            position = self.joint_map[node_name].get_global_position(quat_frame)
            cartesian_frame.append(position)

        return cartesian_frame
