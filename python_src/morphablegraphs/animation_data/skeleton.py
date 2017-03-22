# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:18:37 2015

@author: Erik Herrmann, Martin Manns
"""

import collections
from copy import deepcopy
import json
import numpy as np
from ..external.transformations import quaternion_matrix
from quaternion_frame import QuaternionFrame
from ..animation_data.motion_editing import euler_to_quaternion
from itertools import izip
from skeleton_node import SkeletonRootNode, SkeletonJointNode, SkeletonEndSiteNode, SKELETON_NODE_TYPE_JOINT
from . import ROTATION_TYPE_QUATERNION, ROTATION_TYPE_EULER
try:
    from mgrd import Skeleton as MGRDSkeleton
    from mgrd import SkeletonNode as MGRDSkeletonNode
    has_mgrd = True
except ImportError:
    has_mgrd = False
    pass

DEFAULT_TOOL_BONES = [{
            "new_node_name": 'LeftToolEndSite',
            "parent_node_name": 'LeftHand',
            "new_node_offset": [6.1522069, -0.09354633,  3.33790343]
        },{
            "new_node_name": 'RightToolEndSite',
            "parent_node_name": 'RightHand',
            "new_node_offset": [6.1522069, 0.09354633,  3.33790343]
            },{
            "new_node_name": 'RightScrewDriverEndSite',
            "parent_node_name": 'RightHand',
            "new_node_offset": [22.1522069, -9.19354633, 3.33790343]
            }, {
                "new_node_name": 'LeftScrewDriverEndSite',
                "parent_node_name": 'LeftHand',
                "new_node_offset": [22.1522069,  9.19354633,  3.33790343]
            }
]
DEFAULT_FREE_JOINTS_MAP = {"LeftHand":["Spine","LeftArm", "LeftForeArm"],
                           "RightHand":["Spine","RightArm","RightForeArm"],
                           "LeftToolEndSite":["Spine","LeftArm","LeftForeArm"],
                           "RightToolEndSite":["Spine","RightArm", "RightForeArm"],#, "RightHand"
                            "Head":[],
                           "RightScrewDriverEndSite":["Spine","RightArm","RightForeArm"],
                           "LeftScrewDriverEndSite": ["Spine","LeftArm", "LeftForeArm"]
                           }
DEFAULT_REDUCED_FREE_JOINTS_MAP = {"LeftHand":["LeftArm", "LeftForeArm"],
                                       "RightHand":["RightArm","RightForeArm"],
                                       "LeftToolEndSite":["LeftArm","LeftForeArm"],
                                       "RightToolEndSite":["RightArm", "RightForeArm"],
                                        "Head":[],
                                        "RightScrewDriverEndSite":["RightArm","RightForeArm"],
                                       "LeftScrewDriverEndSite": ["LeftArm", "LeftForeArm"]
                                       }
DEG2RAD = np.pi / 180
hand_bounds = [{"dim": 0, "min":30*DEG2RAD, "max": 180*DEG2RAD}, {"dim": 1, "min": -15*DEG2RAD, "max":120*DEG2RAD}, {"dim": 1, "min": -40*DEG2RAD, "max":40*DEG2RAD}]
DEFAULT_HEAD_JOINT = "Head"
DEFAULT_NECK_JOINT = "Neck"
DEFAULT_BOUNDS = {"LeftArm":[],#{"dim": 1, "min": 0, "max": 90}
                       "RightArm":[]#{"dim": 1, "min": 0, "max": 90},{"dim": 0, "min": 0, "max": 90}
                 ,"RightHand":hand_bounds,#[[-90, 90],[0, 0],[-90,90]]
                  "LeftHand": hand_bounds#[[-90, 90],[0, 0],[-90,90]]
                  }



UNITY_REFERENCE_POSE = \
{"rotations": [{"y": -0.02728448828483766, "x": 0.026979130045631322, "z": -0.71667433988565554, "w": -0.69622837227840906},
{"y": -0.031026367926539958, "x": -0.0045301396093549683, "z": 0.0022127969924659735, "w": 0.9995828340762698},
{"y": 4.0069260614371328e-10, "x": 7.0633929059553417e-10, "z": -2.5830424681454735e-10, "w": 0.99999999999999978},
 {"y": -0.12724628820312309, "x": -0.0063754473142348632, "z": -0.011225899963726169, "w": 0.99146422592244943},
 {"y": 0.081518955062957768, "x": 0.0013514090530289194, "z": 0.0027141144297866205, "w": 0.99589250700677878},
 {"y": 0.84560044204711748, "x": -0.50853248400628293, "z": 0.031967534372243207, "w": -0.15351234961591828},
 {"y": 0.022782181787896816, "x": 0.29483759105619262, "z": 0.29469947547647279, "w": 0.90968376410482854},
 {"y": 0.19700044664706723, "x": -2.8889416108166181e-07, "z": -1.0389068815203269e-07, "w": 0.97755976550301904},
 {"y": 0.075837344449012425, "x": 0.75489503335913755, "z": -0.089786695372349234, "w": -0.62392002567005167},
 {"y": 0.8337004170201876, "x": 0.54085171730808834, "z": -0.041947743977626427, "w": -0.097181067845937205},
 {"y": 0.1832977289492416, "x": -0.43585730993035865, "z": -0.23756399170419112, "w": 0.84721890613230821},
 {"y": 0.24200074221953535, "x": -2.4020236824195608e-07, "z": -8.866701269401663e-08, "w": 0.97050407612725609},
 {"y": -0.0085632014942411752, "x": 0.75095367933495549, "z": -0.081672985246118379, "w": 0.65473587571560399},
 {"y": -0.11849530941152131, "x": 0.044644456768014519, "z": 0.99017019086187563, "w": -0.038496497055440421},
 {"y": 0.24024059851647939, "x": -5.1412884906297746e-10, "z": -3.8293685631199153e-08, "w": 0.97113256791600611},
 {"y": -0.13550730393864649, "x": 0.018940402129228005, "z": -0.052239307325811626, "w": 0.98870093999803688},
 {"y": 0.12067316065876704, "x": -0.032437016121700801, "z": 0.99219446764746755, "w": 0.018042238935378908},
 {"y": 0.14026836383402469, "x": -2.6345270425333571e-10, "z": -2.412424358820542e-08, "w": 0.990130381100918},
 {"y": -0.1770436763916802, "x": -0.00078367233470393038, "z": 0.0040246213147693676, "w": 0.98403149607017226}],
 "translations": [{"y": 0.0, "x": 0.0, "z": 0.0}, {"y": 0.0, "x": 15.3169, "z": -0.012192}, {"y": 0.0, "x": 15.317600000000001, "z": -0.012192}, {"y": -9.9999999999999995e-07, "x": 19.914200000000001, "z": 1.563377}, {"y": 0.0, "x": 6.9265210000000002, "z": 0.0}, {"y": -7.28423, "x": -6.2016910000000003, "z": 0.70883200000000002}, {"y": 1.2999999999999999e-05, "x": 14.107998, "z": 0.0}, {"y": 0.0, "x": 28.580200000000001, "z": 0.0}, {"y": 0.0, "x": 27.091895999999998, "z": 0.0}, {"y": 7.2801390000000001, "x": -6.2054900000000002, "z": 0.70887}, {"y": -1.0000000000000001e-05, "x": 14.107998, "z": 0.0}, {"y": -1.9000000000000001e-05, "x": 28.580200000000001, "z": 0.0}, {"y": 0.0, "x": 27.092002999999998, "z": 0.0}, {"y": -8.8368500000000001, "x": -12.021299000000001, "z": -0.372498}, {"y": 0.0, "x": 39.903500000000001, "z": 0.0}, {"y": 0.0, "x": 39.668900000000001, "z": 0.0}, {"y": 8.8368490000000008, "x": -12.021299000000001, "z": -0.372444}, {"y": 0.0, "x": 39.903500000000001, "z": 0.0}, {"y": 0.0, "x": 39.668900000000001, "z": 0.0}]}

DEFAULT_ANIMATED_JOINT_LIST = ["Hips", "Spine", "Spine_1", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"]

class Skeleton(object):
    """ Data structure that stores the skeleton hierarchy information
        extracted from a BVH file with additional meta information.
    """
    def __init__(self):
        self.animated_joints = DEFAULT_ANIMATED_JOINT_LIST
        self.free_joints_map = DEFAULT_FREE_JOINTS_MAP
        self.reduced_free_joints_map = DEFAULT_REDUCED_FREE_JOINTS_MAP
        self.head_joint = DEFAULT_HEAD_JOINT
        self.neck_joint = DEFAULT_NECK_JOINT
        self.bounds = DEFAULT_BOUNDS
        self.frame_time = None
        self.root = None
        self.node_names = None
        self.reference_frame = None
        self.reference_frame_length = None
        self.node_channels = collections.OrderedDict()
        self.nodes = collections.OrderedDict()
        self.tool_nodes = []
        self.max_level = -1
        self.parent_dict = dict()
        self._chain_names = []

    def load_from_bvh(self, bvh_reader, animated_joints=DEFAULT_ANIMATED_JOINT_LIST, add_tool_joints=True):
        self.animated_joints = animated_joints
        self.frame_time = deepcopy(bvh_reader.frame_time)
        self.root = deepcopy(bvh_reader.root)
        self.node_names = deepcopy(bvh_reader.node_names)
        self.reference_frame = self._extract_reference_frame(bvh_reader)
        self.reference_frame_length = len(self.reference_frame)
        self.node_channels = collections.OrderedDict()
        self.extract_channels()
        self.nodes = collections.OrderedDict()
        self._create_filtered_node_name_frame_map()
        self.tool_nodes = []
        if add_tool_joints:
            self._add_tool_nodes(DEFAULT_TOOL_BONES)
        self.max_level = self._get_max_level()
        self._set_joint_weights()


        self.nodes = collections.OrderedDict()
        self.construct_hierarchy_from_bvh(bvh_reader.node_names, self.node_channels, self.root)

        self.parent_dict = self._get_parent_dict()
        self._chain_names = self._generate_chain_names()

    def construct_hierarchy_from_bvh(self, node_names, node_channels, node_name):
        joint_index = node_names.keys().index(node_name)

        if node_name == self.root:
            node = SkeletonRootNode(node_name, node_channels[node_name], None)
            if node_name in self.animated_joints:
                node.fixed = False
                node.quaternion_frame_index = self.animated_joints.index(node_name)
            else:
                node.fixed = True
            node.index = joint_index
        elif "children" in node_names[node_name].keys() and len(node_names[node_name]["children"]) > 0:
            node = SkeletonJointNode(node_name, node_channels[node_name], None)
            offset = joint_index * 4 + 3
            node.rotation = self.reference_frame[offset: offset + 4]
            if node_name in self.animated_joints:
                node.fixed = False
                node.quaternion_frame_index = self.animated_joints.index(node_name)
            else:
                node.fixed = True
            node.index = joint_index
        else:
            node = SkeletonEndSiteNode(node_name, [], None)

        node.index = joint_index

        print "node", node_name, node.quaternion_frame_index, node.index

        node.offset = self.node_names[node_name]["offset"]

        if "children" in node_names[node_name].keys():
            for c in node_names[node_name]["children"]:
                c_node = self.construct_hierarchy_from_bvh(node_names, node_channels, c)
                c_node.parent = node
                node.children.append(c_node)
        self.nodes[node_name] = node

        return node

    def load_from_json_file(self, filename):
        with open(filename) as infile:
            data = json.load(infile)
            self.load_from_json_data(data)

    def load_from_json_data(self, data):
        self.animated_joints = data["animated_joints"]
        if "free_joints_map" in data.keys():
            self.free_joints_map = data["free_joints_map"]
        self.reduced_free_joints_map = DEFAULT_REDUCED_FREE_JOINTS_MAP#data["reduced_free_joints_map"]
        self.bounds = DEFAULT_BOUNDS#data["bounds"]
        if "head_joint" in data.keys():
            self.head_joint = data["head_joint"]
        if "neck_joint" in data.keys():
            self.neck_joint = data["neck_joint"]

        self.frame_time = data["frame_time"]
        self.nodes = collections.OrderedDict()
        root = self._create_node_from_desc(data["root"], None)
        self.root = root.node_name
        self.reference_frame = np.array(data["reference_frame"])
        self.reference_frame_length = len(self.reference_frame)
        self.node_channels = data["node_channels"]
        if "tool_nodes" in data.keys():
            self.tool_nodes = data["tool_nodes"]
        self.node_name_frame_map = data["node_name_frame_map"]
        self.node_names = data["node_names"]
        self.max_level = self._get_max_level()
        self._set_joint_weights()
        self.parent_dict = self._get_parent_dict()
        self._chain_names = self._generate_chain_names()

    def load_from_fbx_data(self, data):
        self.animated_joints = data["animated_joints"]
        #self.inv_bind_poses = [self._create_node_from_desc(node, None) for node in data["nodes"].values()]
        self.root = data["root"]
        self._create_node_from_desc2(data, self.root, None)
        self.frame_time = data["frame_time"]
        #self.max_level = self._get_max_level()
        #self._set_joint_weights()
        self.parent_dict = self._get_parent_dict()
        self._chain_names = self._generate_chain_names()

    def _create_node_from_desc(self, data, parent):
        node_name = data["name"]
        channels = data["channels"]
        if parent is None:
            node = SkeletonRootNode( node_name, channels, parent)
        elif data["node_type"] == SKELETON_NODE_TYPE_JOINT:
            node = SkeletonJointNode(node_name, channels, parent)
        else:
            node = SkeletonEndSiteNode(node_name, channels, parent)
        node.fixed = data["fixed"]
        node.index = data["index"]
        node.offset = np.array(data["offset"])
        node.rotation = np.array(data["rotation"])
        node.quaternion_frame_index = data["quaternion_frame_index"]
        self.nodes[node_name] = node
        self.nodes[node_name].children = []
        for c_desc in data["children"]:
            self.nodes[node_name].children.append(self._create_node_from_desc(c_desc, node))
        return node

    def _create_node_from_desc2(self, data, node_name, parent):
        node_data = data["nodes"][node_name]

        channels = node_data["channels"]
        if parent is None:
            node = SkeletonRootNode(node_name, channels, parent)
        elif node_data["node_type"] == SKELETON_NODE_TYPE_JOINT:
            node = SkeletonJointNode(node_name, channels, parent)
        else:
            node = SkeletonEndSiteNode(node_name, channels, parent)
        node.fixed = node_data["fixed"]
        node.index = node_data["index"]
        node.offset = np.array(node_data["offset"])
        node.rotation = np.array(node_data["rotation"])
        node.quaternion_frame_index = node_data["quaternion_frame_index"]
        node.children = []

        for c_name in node_data["children"]:
            c_node = self._create_node_from_desc2(data, c_name, node)
            node.children.append(c_node)
        self.nodes[node_name] = node

        return node

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
        data["head_joint"] = self.head_joint
        data["neck_joint"] = self.neck_joint
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
            if "channels" in self.node_names[node_name].keys():
                channels = self.node_names[node_name]["channels"]
                self.node_channels[node_name] = channels

    def _extract_reference_frame(self, bvh_reader, frame_index=0):
        quaternion_frame = np.array((QuaternionFrame(bvh_reader, bvh_reader.frames[frame_index], False, False).values())).flatten()
        return np.array(bvh_reader.frames[0][:3].tolist() + quaternion_frame.tolist())

    def is_motion_vector_complete(self, frames, is_quaternion):
        if is_quaternion:
            rotation_type = ROTATION_TYPE_QUATERNION
        else:
            rotation_type = ROTATION_TYPE_EULER
        return len(frames[0]) == self.get_number_of_frame_parameters(rotation_type)

    def get_number_of_frame_parameters(self, rotation_type):
        n_parameters = 0
        for node_name in self.nodes.keys():
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
        for joint_name in self.nodes.keys():
            if len(self.nodes[joint_name].children) > 0:
                if joint_name == self.root:
                    new_frame[:7] = reduced_frame[:7]
                else:
                    dest_start = self.nodes[joint_name].index * 4 + 3
                    if self.nodes[joint_name].fixed:
                        new_frame[dest_start: dest_start+4] = self.nodes[joint_name].rotation
                    else:
                        src_start = self.nodes[joint_name].quaternion_frame_index * 4 + 3
                        new_frame[dest_start: dest_start+4] = reduced_frame[src_start: src_start + 4]
        return new_frame

    def _get_max_level(self):
        return max([node["level"] for node in self.node_names.values() if "level" in node.keys()])

    def _get_parent_dict(self):
        """Returns a dict of node names to their parent node's name"""

        parent_dict = {}
        for node_name in self.nodes.keys():
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
        return self.joint_weight_map.values()

    def _add_tool_nodes(self, new_tool_bones):
        for b in new_tool_bones:
            self._add_new_end_site(b["new_node_name"], b["parent_node_name"], b["new_node_offset"])
            self.tool_nodes.append(b["new_node_name"])

    def _add_new_end_site(self, new_node_name, parent_node_name, offset):
        if parent_node_name in self.node_names.keys():
            level = self.node_names[parent_node_name]["level"] + 1
            node_desc = dict()
            node_desc["level"] = level
            node_desc["offset"] = offset
            self.node_names[parent_node_name]["children"].append(new_node_name)
            self.node_names[new_node_name] = node_desc
            self.node_name_frame_map[new_node_name] = -1 #the node needs an entry but the index is only important if it has children

    def _generate_chain_names(self):
        chain_names = dict()
        for node_name in self.nodes.keys():
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
                return point[:3].tolist()

    def convert_quaternion_frame_to_cartesian_frame(self, quat_frame):
        """
        Converts quaternion frames to cartesian frames by calling get_cartesian_coordinates_from_quaternion for each joint
        """
        cartesian_frame = []
        for node_name in self.node_name_frame_map.keys():
            position = self.nodes[node_name].get_global_position(quat_frame)
            cartesian_frame.append(position)
        return cartesian_frame

    def clear_cached_global_matrices(self):
        for joint in self.nodes.values():
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
            if name in self.nodes.keys():
                node = self.nodes[name]
                indices.append(node.index)
        return indices

    def get_n_joints(self):
        return len([node for node in self.nodes.values() if len(node.channels) > 0])

    def to_unity_json(self, joint_name_map=None):
        joint_descs = []
        self.nodes[self.root].to_unity_json(joint_descs, self.animated_joints, joint_name_map=joint_name_map)

        data = dict()
        data["root"] = self.root
        data["jointDescs"] = joint_descs
        data["jointSequence"] = self.animated_joints
        data["referencePose"] = UNITY_REFERENCE_POSE
        return data