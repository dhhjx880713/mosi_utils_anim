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
from skeleton_node import SkeletonRootNode, SkeletonJointNode, SkeletonEndSiteNode, SKELETON_NODE_TYPE_JOINT, SKELETON_NODE_TYPE_END_SITE
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
DEFAULT_ROOT_DIR = [0,0,1]
DEFAULT_BOUNDS = {"LeftArm":[],#{"dim": 1, "min": 0, "max": 90}
                       "RightArm":[]#{"dim": 1, "min": 0, "max": 90},{"dim": 0, "min": 0, "max": 90}
                 ,"RightHand":hand_bounds,#[[-90, 90],[0, 0],[-90,90]]
                  "LeftHand": hand_bounds#[[-90, 90],[0, 0],[-90,90]]
                  }

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
        self.aligning_root_node = None  # Node that defines body orientation. Can be different from the root node.
        self.aligning_root_dir = DEFAULT_ROOT_DIR
        self.node_names = None
        self.reference_frame = None
        self.reference_frame_length = None
        self.node_channels = collections.OrderedDict()
        self.nodes = collections.OrderedDict()
        self.tool_nodes = []
        self.max_level = -1
        self.parent_dict = dict()
        self._chain_names = []

    def load_from_bvh(self, bvh_reader, animated_joints=None, add_tool_joints=True):
        if animated_joints is None:
            animated_joints = DEFAULT_ANIMATED_JOINT_LIST
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
        self.create_euler_frame_indice()

    def create_euler_frame_indice(self):
        nodes_without_endsite = [node for node in self.nodes.values() if node.node_type != SKELETON_NODE_TYPE_END_SITE]
        for node in nodes_without_endsite:
            node.euler_frame_index = nodes_without_endsite.index(node)

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
            if node_name in self.animated_joints:
                node.fixed = False
                node.quaternion_frame_index = self.animated_joints.index(node_name)
                offset = node.quaternion_frame_index * 4 + 3
                node.rotation = self.reference_frame[offset: offset + 4]
            else:
                node.fixed = True
            node.index = joint_index
        else:
            node = SkeletonEndSiteNode(node_name, [], None)

        node.index = joint_index

        print "node", node_name, node.quaternion_frame_index, node.index

        node.offset = self.node_names[node_name]["offset"]
        self.nodes[node_name] = node
        if "children" in node_names[node_name].keys():
            for c in node_names[node_name]["children"]:
                c_node = self.construct_hierarchy_from_bvh(node_names, node_channels, c)
                c_node.parent = node
                node.children.append(c_node)
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
        if "aligning_root_node" in data.keys():
            self.aligning_root_node = data["aligning_root_node"]

        else:
            self.aligning_root_node = self.root

        if "aligning_root_dir" in data.keys():
            self.aligning_root_dir = data["aligning_root_dir"]
        else:
            self.aligning_root_dir = DEFAULT_ROOT_DIR

    def load_from_fbx_data(self, data):
        self.nodes = collections.OrderedDict()

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
        #node.fixed = data["fixed"]
        node.index = data["index"]
        node.offset = np.array(data["offset"])
        node.rotation = np.array(data["rotation"])
        if node_name in self.animated_joints:
            node.quaternion_frame_index = self.animated_joints.index(node_name)
            node.fixed = False
        else:
            node.quaternion_frame_index = -1
            node.fixed = True
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
        #node.fixed = node_data["fixed"]
        node.index = node_data["index"]
        node.offset = np.array(node_data["offset"])
        node.rotation = np.array(node_data["rotation"])
        if node_name in self.animated_joints:
            node.quaternion_frame_index = self.animated_joints.index(node_name)
            node.fixed = False
        else:
            node.quaternion_frame_index = -1
            node.fixed = True
        node.children = []
        self.nodes[node_name] = node
        for c_name in node_data["children"]:
            c_node = self._create_node_from_desc2(data, c_name, node)
            node.children.append(c_node)

        return node

    def _get_node_desc(self, name):
        node_desc = dict()
        node = self.nodes[name]
        node_desc["name"] = node.node_name
        if node.parent is not None:
            node_desc["parent"] = node.parent.node_name
        else:
            node_desc["parent"] = None
        if name in self.animated_joints:
            node.quaternion_frame_index = self.animated_joints.index(name)
        else:
            node.quaternion_frame_index = -1
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
        joint_index = 0
        for joint_name in self.nodes.keys():
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

    def to_unity_format(self, joint_name_map=None, scale=1):
        """ Converts the skeleton into a custom json format and applies a coordinate transform to the left-handed coordinate system of Unity.
            src: http://answers.unity3d.com/questions/503407/need-to-convert-to-right-handed-coordinates.html
        """

        animated_joints = [j for j, n in self.nodes.items() if "EndSite" not in j and len(n.children) > 0]#self.animated_joints
        joint_descs = []
        self.nodes[self.root].to_unity_format(joint_descs, animated_joints, joint_name_map=joint_name_map)

        data = dict()
        data["root"] = self.aligning_root_node
        data["jointDescs"] = joint_descs
        data["jointSequence"] = animated_joints
        default_pose = dict()
        default_pose["rotations"] = []
        for node in self.nodes.values():
              if node.node_name in animated_joints:
                  q = node.rotation
                  if len(q) ==4:
                      r = {"x":-q[1], "y":q[2], "z":q[3], "w":-q[0]}
                  else:
                      r = {"x":0, "y":0, "z":0, "w":1}
                  default_pose["rotations"].append(r)

        default_pose["translations"] = [{"x":-scale*node.offset[0], "y":scale*node.offset[1], "z":scale*node.offset[2]} for node in self.nodes.values() if node.node_name in animated_joints and len(node.children) > 0]

        data["referencePose"] = default_pose
        return data
