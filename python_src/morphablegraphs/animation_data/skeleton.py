# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:18:37 2015

@author: Erik Herrmann, Martin Manns
"""

import collections
from copy import copy
import json
import numpy as np
from ..external.transformations import quaternion_matrix

from .constants import ROTATION_TYPE_QUATERNION, ROTATION_TYPE_EULER
from .skeleton_models import ROCKETBOX_ANIMATED_JOINT_LIST, ROCKETBOX_FREE_JOINTS_MAP, ROCKETBOX_REDUCED_FREE_JOINTS_MAP, ROCKETBOX_SKELETON_MODEL, ROCKETBOX_BOUNDS, ROCKETBOX_TOOL_BONES, ROCKETBOX_ROOT_DIR
try:
    from mgrd import Skeleton as MGRDSkeleton
    from mgrd import SkeletonNode as MGRDSkeletonNode
    has_mgrd = True
except ImportError:
    has_mgrd = False
    pass


class Skeleton(object):
    """ Data structure that stores the skeleton hierarchy information
        extracted from a BVH file with additional meta information.
    """
    def __init__(self):
        self.animated_joints = ROCKETBOX_ANIMATED_JOINT_LIST
        self.free_joints_map = ROCKETBOX_FREE_JOINTS_MAP
        self.reduced_free_joints_map = ROCKETBOX_REDUCED_FREE_JOINTS_MAP
        self.skeleton_model = ROCKETBOX_SKELETON_MODEL
        self.bounds = ROCKETBOX_BOUNDS
        self.frame_time = None
        self.root = None
        self.aligning_root_node = None  # Node that defines body orientation. Can be different from the root node.
        self.aligning_root_dir = None
        self.node_names = None
        self.reference_frame = None
        self.reference_frame_length = None
        self.node_channels = collections.OrderedDict()
        self.nodes = collections.OrderedDict()
        self.node_name_frame_map = collections.OrderedDict()
        self.tool_nodes = []
        self.max_level = -1
        self.parent_dict = dict()
        self._chain_names = []
        self.identity_frame = None
        self.annotation = None

    def _get_node_desc(self, name):
        node_desc = dict()
        node = self.nodes[name]
        node_desc["name"] = node.node_name
        if node.parent is not None:
            node_desc["parent"] = node.parent.node_name
        else:
            node_desc["parent"] = None
        node_desc["quaternion_frame_index"] = node.quaternion_frame_index
        node_desc["index"] = node.index
        node_desc["offset"] = node.offset
        node_desc["channels"] = node.channels
        node_desc["rotation"] = node.rotation.tolist()
        node_desc["fixed"] = node.fixed
        node_desc["node_type"] = node.node_type
        node_desc["children"] = []
        for c in node.children:
            c_desc = self._get_node_desc(c.node_name)
            node_desc["children"].append(c_desc)
        return node_desc

    def save_to_json(self, file_name):
        data = dict()
        data["animated_joints"] = self.animated_joints
        data["node_names"] = self.node_names
        data["free_joints_map"] = self.free_joints_map
        data["reduced_free_joints_map"] = self.reduced_free_joints_map
        data["bounds"] = self.bounds
        data["skeleton_model"] = self.skeleton_model
        data["frame_time"] = self.frame_time
        data["root"] = self._get_node_desc(self.root)
        data["reference_frame"] = self.reference_frame.tolist()
        data["node_channels"] = self.node_channels
        data["tool_nodes"] = self.tool_nodes
        data["node_name_frame_map"] = self.node_name_frame_map
        with open(file_name, 'wb') as outfile:
            tmp = json.dumps(data, indent=4)
            outfile.write(tmp)
            outfile.close()

    def extract_channels(self):
        for node_idx, node_name in enumerate(self.node_names):
            if "channels" in list(self.node_names[node_name].keys()):
                channels = self.node_names[node_name]["channels"]
                self.node_channels[node_name] = channels

    def is_motion_vector_complete(self, frames, is_quaternion):
        if is_quaternion:
            rotation_type = ROTATION_TYPE_QUATERNION
        else:
            rotation_type = ROTATION_TYPE_EULER
        return len(frames[0]) == self.get_number_of_frame_parameters(rotation_type)

    def get_number_of_frame_parameters(self, rotation_type):
        n_parameters = 0
        for node_name in list(self.nodes.keys()):
            local_parameters = self.nodes[node_name].get_number_of_frame_parameters(rotation_type)
            n_parameters += local_parameters
        return n_parameters

    def complete_motion_vector_from_reference(self, reduced_quat_frames):
        new_quat_frames = np.zeros((len(reduced_quat_frames), self.reference_frame_length))
        for idx, reduced_frame in enumerate(reduced_quat_frames):
            new_quat_frames[idx] = self.generate_complete_frame_vector_from_reference(reduced_frame)
        return new_quat_frames

    def generate_complete_frame_vector_from_reference(self, reduced_frame):
        """
        Takes parameters from the reduced frame for each joint of the complete skeleton found in the reduced skeleton
        otherwise it takes parameters from the reference frame
        :param reduced_frame:
        :return:
        """
        new_frame = np.zeros(self.reference_frame_length)
        joint_index = 0
        for joint_name in list(self.nodes.keys()):
            if len(self.nodes[joint_name].children) > 0 and "EndSite" not in joint_name:
                if joint_name == self.root:
                    new_frame[:7] = reduced_frame[:7]
                else:
                    dest_start = joint_index * 4 + 3
                    if self.nodes[joint_name].fixed:
                        new_frame[dest_start: dest_start+4] = self.nodes[joint_name].rotation
                    else:
                        src_start = self.nodes[joint_name].quaternion_frame_index * 4 + 3
                        new_frame[dest_start: dest_start+4] = reduced_frame[src_start: src_start + 4]

                joint_index += 1
        return new_frame

    def _get_max_level(self):
        levels = [node["level"] for node in list(self.node_names.values()) if "level" in list(node.keys())]
        if len(levels)> 0:
            return max(levels)
        else:
            return 0

    def _get_parent_dict(self):
        """Returns a dict of node names to their parent node's name"""

        parent_dict = {}
        for node_name in list(self.nodes.keys()):
            for child_node in self.nodes[node_name].children:
                parent_dict[child_node.node_name] = node_name

        return parent_dict

    def gen_all_parents(self, node_name):
        """Generator of all parents' node names of node with node_name"""

        while node_name in self.parent_dict:
            parent_name = self.parent_dict[node_name]
            yield parent_name
            node_name = parent_name

    def _set_joint_weights(self):
        """ Gives joints weights according to their distance in the joint hierarchy
           to the root joint. The further away the smaller the weight.
        """

        # self.joint_weights = [np.exp(-self.node_names[node_name]["level"])
        #                       for node_name in self.node_name_frame_map.keys()]
        self.joint_weights = [1.0/(self.node_names[node_name]["level"] + 1.0) for node_name in list(self.node_name_frame_map.keys())]
        self.joint_weight_map = collections.OrderedDict()
        weight_index = 0
        for node_name in list(self.node_name_frame_map.keys()):
            self.joint_weight_map[node_name] = self.joint_weights[weight_index]
            weight_index += 1
        self.joint_weight_map["RightHand"] = 2.0
        self.joint_weight_map["LeftHand"] = 2.0
        self.joint_weights = list(self.joint_weight_map.values())

    def get_joint_weights(self):
        return list(self.joint_weight_map.values())

    def _generate_chain_names(self):
        chain_names = dict()
        for node_name in list(self.nodes.keys()):
            chain_names[node_name] = list(self.gen_all_parents(node_name))
            # Names are generated bottom to up --> reverse
            chain_names[node_name].reverse()
            chain_names[node_name] += [node_name]  # Node is not in its parent list
        return chain_names

    def get_cartesian_coordinates_from_quaternion(self, target_node_name, quaternion_frame, return_global_matrix=False):
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
                    zip(root_frame_position, root_node_offset)]
        else:
            offsets = [self.node_names[node_name]["offset"]
                       for node_name in self._chain_names[target_node_name]]
            root_position = quaternion_frame[:3].flatten()
            offsets[0] = [r + o for r, o in zip(root_position, offsets[0])]
            j_matrices = []
            count = 0
            for node_name in self._chain_names[target_node_name]:
                if "children" in list(self.node_names[node_name].keys()):  # check if it is a joint or an end site
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
                return point[:3].tolist()

    def convert_quaternion_frame_to_cartesian_frame(self, quat_frame, node_names=None):
        """
        Converts quaternion frames to cartesian frames by calling get_cartesian_coordinates_from_quaternion for each joint
        """
        if node_names is None:
            node_names = list(self.node_name_frame_map.keys())
        cartesian_frame = []
        for node_name in node_names:
            if node_name in list(self.node_name_frame_map.keys()):
                position = self.nodes[node_name].get_global_position(quat_frame)
                cartesian_frame.append(position)
        return cartesian_frame

    def clear_cached_global_matrices(self):
        for joint in list(self.nodes.values()):
            joint.clear_cached_global_matrix()

    def convert_to_mgrd_skeleton(self):
        if not has_mgrd:
            return None

        def create_mgrd_node(mg_node, parent):
            mgrd_node = MGRDSkeletonNode(mg_node.node_name, parent, mg_node.offset, mg_node.rotation)
            mgrd_node.fixed = mg_node.fixed
            return mgrd_node

        def populate(skeleton, mgrd_node):
            node = skeleton.nodes[mgrd_node.name]
            for child in node.children:
                child_node = create_mgrd_node(child, mgrd_node)
                mgrd_node.add_child(child_node)
                populate(skeleton, child_node)
        root_node = create_mgrd_node(self.nodes[self.root], None)
        populate(self, root_node)
        return MGRDSkeleton(root_node)

    def get_root_reference_orientation(self):
        # reference orientation from BVH: 179.477078182 3.34148613293 -87.6482840381 x y z euler angles
        return self.reference_frame[3:7]

    def get_joint_indices(self, joint_names):
        indices = []
        for name in joint_names:
            if name in list(self.nodes.keys()):
                node = self.nodes[name]
                indices.append(node.index)
        return indices

    def get_n_joints(self):
        return len([node for node in list(self.nodes.values()) if len(node.channels) > 0])

    def to_unity_format(self, joint_name_map=None, scale=1):
        """ Converts the skeleton into a custom json format and applies a coordinate transform to the left-handed coordinate system of Unity.
            src: http://answers.unity3d.com/questions/503407/need-to-convert-to-right-handed-coordinates.html
        """

        animated_joints = [j for j, n in list(self.nodes.items()) if "EndSite" not in j and len(n.children) > 0]#self.animated_joints
        joint_descs = []
        self.nodes[self.root].to_unity_format(joint_descs, animated_joints, joint_name_map=joint_name_map)

        data = dict()
        data["root"] = self.aligning_root_node
        data["jointDescs"] = joint_descs
        data["jointSequence"] = animated_joints
        default_pose = dict()
        default_pose["rotations"] = []
        for node in list(self.nodes.values()):
              if node.node_name in animated_joints:
                  q = node.rotation
                  if len(q) ==4:
                      r = {"x":-q[1], "y":q[2], "z":q[3], "w":-q[0]}
                  else:
                      r = {"x":0, "y":0, "z":0, "w":1}
                  default_pose["rotations"].append(r)

        default_pose["translations"] = [{"x":-scale*node.offset[0], "y":scale*node.offset[1], "z":scale*node.offset[2]} for node in list(self.nodes.values()) if node.node_name in animated_joints and len(node.children) > 0]

        data["referencePose"] = default_pose
        return data

    def get_joint_names(self):
        return [k for k, n in list(self.nodes.items()) if len(n.children) > 0]

    def scale(self, scale_factor):
        for node in list(self.nodes.values()):
            node.offset = np.array(node.offset) * scale_factor

    def get_channels(self, euler=False):
        channels = collections.OrderedDict()
        for node in list(self.nodes.values()):
            if node.node_name in self.animated_joints:
                node_channels = copy(node.channels)
                if not euler:
                    if np.all([ch in node_channels for ch in ["Xrotation", "Yrotation", "Zrotation"]]):
                        node_channels += ["Wrotation"]  # TODO fix order
                channels[node.node_name] = node_channels
        return channels
