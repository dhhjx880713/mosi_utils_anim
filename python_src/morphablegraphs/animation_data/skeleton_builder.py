import collections
from copy import deepcopy
import json
import numpy as np
from skeleton import Skeleton
from skeleton_node import SkeletonRootNode, SkeletonJointNode, SkeletonEndSiteNode, SKELETON_NODE_TYPE_JOINT, SKELETON_NODE_TYPE_END_SITE
from skeleton_models import ROCKETBOX_ANIMATED_JOINT_LIST, ROCKETBOX_FREE_JOINTS_MAP, ROCKETBOX_REDUCED_FREE_JOINTS_MAP, ROCKETBOX_SKELETON_MODEL, ROCKETBOX_BOUNDS, ROCKETBOX_TOOL_BONES, ROCKETBOX_ROOT_DIR
from quaternion_frame import QuaternionFrame


def create_identity_frame(skeleton):
    skeleton.identity_frame = np.zeros(skeleton.reference_frame_length)
    offset = 3
    for j in skeleton.nodes.keys():
        if len(skeleton.nodes[j].channels) > 0:
            skeleton.identity_frame[offset:offset + 4] = [1, 0, 0, 0]
            offset += 4


def create_euler_frame_indices(skeleton):
    nodes_without_endsite = [node for node in skeleton.nodes.values() if node.node_type != SKELETON_NODE_TYPE_END_SITE]
    for node in nodes_without_endsite:
        node.euler_frame_index = nodes_without_endsite.index(node)


def read_reference_frame_from_bvh_reader(bvh_reader, frame_index=0):
    quaternion_frame = np.array((QuaternionFrame(bvh_reader, bvh_reader.frames[frame_index], False, False).values())).flatten()
    return np.array(bvh_reader.frames[0][:3].tolist() + quaternion_frame.tolist())

def add_tool_nodes(skeleton, new_tool_bones):
    for b in new_tool_bones:
        add_new_end_site(skeleton, b["new_node_name"], b["parent_node_name"], b["new_node_offset"])
        skeleton.tool_nodes.append(b["new_node_name"])


def add_new_end_site(skeleton, new_node_name, parent_node_name, offset):
    if parent_node_name in skeleton.node_names.keys():
        level = skeleton.node_names[parent_node_name]["level"] + 1
        node_desc = dict()
        node_desc["level"] = level
        node_desc["offset"] = offset
        skeleton.node_names[parent_node_name]["children"].append(new_node_name)
        skeleton.node_names[new_node_name] = node_desc
        skeleton.node_name_frame_map[new_node_name] = -1  # the node needs an entry but the index is only important if it has children


def create_filtered_node_name_frame_map(skeleton):
    """
    creates dictionary that maps node names to indices in a frame vector
    without "Bip" joints
    """
    skeleton.node_name_frame_map = collections.OrderedDict()
    j = 0
    for node_name in skeleton.node_names:
        if not node_name.startswith("Bip") and \
                        "children" in skeleton.node_names[node_name].keys():
            skeleton.node_name_frame_map[node_name] = j
            j += 1


class SkeletonBuilder(object):

    def load_from_bvh(self, bvh_reader, animated_joints=None, add_tool_joints=True):
        skeleton = Skeleton()
        if animated_joints is None:
            animated_joints = ROCKETBOX_ANIMATED_JOINT_LIST
        skeleton.animated_joints = animated_joints
        skeleton.frame_time = deepcopy(bvh_reader.frame_time)
        skeleton.root = deepcopy(bvh_reader.root)
        skeleton.node_names = deepcopy(bvh_reader.node_names)
        skeleton.reference_frame = read_reference_frame_from_bvh_reader(bvh_reader)
        skeleton.reference_frame_length = len(skeleton.reference_frame)
        skeleton.node_channels = collections.OrderedDict()
        skeleton.extract_channels()
        skeleton.nodes = collections.OrderedDict()
        create_filtered_node_name_frame_map(skeleton)
        skeleton.tool_nodes = []
        if add_tool_joints:
            add_tool_nodes(skeleton, ROCKETBOX_TOOL_BONES)
        skeleton.max_level = skeleton._get_max_level()
        skeleton._set_joint_weights()
        skeleton.nodes = collections.OrderedDict()
        joint_list = [k for k in bvh_reader.node_names if
                      "children" in bvh_reader.node_names[k].keys() and len(bvh_reader.node_names[k]["children"]) > 0]
        self.construct_hierarchy_from_bvh(skeleton, joint_list, bvh_reader.node_names, skeleton.node_channels, skeleton.root)

        skeleton.parent_dict = skeleton._get_parent_dict()
        skeleton._chain_names = skeleton._generate_chain_names()
        create_euler_frame_indices(skeleton)
        create_identity_frame(skeleton)
        return skeleton

    def construct_hierarchy_from_bvh(self, skeleton, joints, node_info, node_channels, node_name):
        if node_name in joints:
            joint_index = joints.index(node_name)
        else:
            joint_index = -1
        if node_name == skeleton.root:
            node = SkeletonRootNode(node_name, node_channels[node_name], None)
            if node_name in skeleton.animated_joints:
                node.fixed = False
                node.quaternion_frame_index = skeleton.animated_joints.index(node_name)
            else:
                node.fixed = True
            node.index = joint_index
        elif "children" in node_info[node_name].keys() and len(node_info[node_name]["children"]) > 0:
            node = SkeletonJointNode(node_name, node_channels[node_name], None)
            if node_name in skeleton.animated_joints:
                node.fixed = False
                node.quaternion_frame_index = skeleton.animated_joints.index(node_name)
                offset = node.quaternion_frame_index * 4 + 3
                node.rotation = skeleton.reference_frame[offset: offset + 4]
            else:
                node.fixed = True
            node.index = joint_index
        else:
            node = SkeletonEndSiteNode(node_name, [], None)

        node.index = joint_index
        node.offset = skeleton.node_names[node_name]["offset"]
        skeleton.nodes[node_name] = node
        if "children" in node_info[node_name].keys():
            for c in node_info[node_name]["children"]:
                c_node = self.construct_hierarchy_from_bvh(skeleton, joints, node_info, node_channels, c)
                c_node.parent = node
                node.children.append(c_node)
        return node

    def construct_hierarchy_from_bvh2(self, skeleton, node_names, node_channels, node_name):
        joint_index = node_names.keys().index(node_name)
        if node_name == skeleton.root:
            node = SkeletonRootNode(node_name, node_channels[node_name], None)
            if node_name in skeleton.animated_joints:
                node.fixed = False
                node.quaternion_frame_index = skeleton.animated_joints.index(node_name)
                offset = node.quaternion_frame_index + 3
                node.rotation = skeleton.reference_frame[offset: offset + 4]
            else:
                node.fixed = True
            node.index = joint_index
        elif "children" in node_names[node_name].keys() and len(node_names[node_name]["children"]) > 0:
            node = SkeletonJointNode(node_name, node_channels[node_name], None)
            offset = skeleton.animated_joints.index(node_name) * 4 + 3  # TODO fix
            node.rotation = skeleton.reference_frame[offset: offset + 4]
            if node_name in skeleton.animated_joints:
                node.fixed = False
                node.quaternion_frame_index = skeleton.animated_joints.index(node_name)
            else:
                node.fixed = True
            node.index = joint_index
        else:
            node = SkeletonEndSiteNode(node_name, [], None)

        node.index = joint_index

        # print "node", node_name, node.quaternion_frame_index, node.index

        node.offset = skeleton.node_names[node_name]["offset"]
        skeleton.nodes[node_name] = node
        if "children" in node_names[node_name].keys():
            for c in node_names[node_name]["children"]:
                c_node = self.construct_hierarchy_from_bvh(skeleton, node_names, node_channels, c)
                c_node.parent = node
                node.children.append(c_node)
        return node

    def load_from_json_file(self, filename):
        with open(filename) as infile:
            data = json.load(infile)
            skeleton = self.load_from_json_data(data)
            return skeleton

    def load_from_json_data(self, data):
        skeleton = Skeleton()
        print "load from json"
        skeleton.animated_joints = data["animated_joints"]
        if "free_joints_map" in data.keys():
            skeleton.free_joints_map = data["free_joints_map"]
        skeleton.reduced_free_joints_map = ROCKETBOX_REDUCED_FREE_JOINTS_MAP  # data["reduced_free_joints_map"]
        skeleton.bounds = ROCKETBOX_BOUNDS  # data["bounds"]
        if "skeleton_model" in data.keys():
            skeleton.skeleton_model = data["skeleton_model"]
        else:
            skeleton.skeleton_model = collections.OrderedDict()
            if "head_joint" in data.keys():
                skeleton.skeleton_model["Head"] = data["head_joint"]
            if "neck_joint" in data.keys():
                skeleton.skeleton_model["Neck"] = data["neck_joint"]

        skeleton.frame_time = data["frame_time"]
        skeleton.nodes = collections.OrderedDict()
        root = self._create_node_from_desc(skeleton, data["root"], None)
        skeleton.root = root.node_name
        skeleton.reference_frame = np.array(data["reference_frame"])
        skeleton.reference_frame_length = len(skeleton.reference_frame)
        skeleton.node_channels = data["node_channels"]
        if "tool_nodes" in data.keys():
            skeleton.tool_nodes = data["tool_nodes"]
        skeleton.node_name_frame_map = data["node_name_frame_map"]
        skeleton.node_names = data["node_names"]
        skeleton.max_level = skeleton._get_max_level()
        skeleton._set_joint_weights()
        skeleton.parent_dict = skeleton._get_parent_dict()
        skeleton._chain_names = skeleton._generate_chain_names()
        create_euler_frame_indices(skeleton)
        if "aligning_root_node" in data.keys():
            skeleton.aligning_root_node = data["aligning_root_node"]
        else:
            skeleton.aligning_root_node = skeleton.root
        if "aligning_root_dir" in data.keys():
            skeleton.aligning_root_dir = data["aligning_root_dir"]
        else:
            skeleton.aligning_root_dir = ROCKETBOX_ROOT_DIR
        create_identity_frame(skeleton)
        return skeleton

    def load_from_fbx_data(self, data):
        skeleton = Skeleton()
        skeleton.nodes = collections.OrderedDict()

        skeleton.animated_joints = data["animated_joints"]
        # self.inv_bind_poses = [self._create_node_from_desc(node, None) for node in data["nodes"].values()]
        skeleton.root = data["root"]
        self._create_node_from_desc2(skeleton, data, skeleton.root, None)
        skeleton.frame_time = data["frame_time"]
        skeleton.parent_dict = skeleton._get_parent_dict()
        skeleton._chain_names = skeleton._generate_chain_names()

        n_params = len(skeleton.animated_joints) * 4 + 3
        skeleton.reference_frame = np.zeros(n_params)
        offset = 3
        for node_name in skeleton.animated_joints:
            skeleton.reference_frame[offset:offset + 4] = data["nodes"][node_name]["rotation"]
            offset += 4
        skeleton.reference_frame_length = len(skeleton.reference_frame)
        create_identity_frame(skeleton)
        return skeleton

    def _create_node_from_desc(self, skeleton, data, parent):
        node_name = data["name"]
        channels = data["channels"]
        if parent is None:
            node = SkeletonRootNode(node_name, channels, parent)
        elif data["node_type"] == SKELETON_NODE_TYPE_JOINT:
            node = SkeletonJointNode(node_name, channels, parent)
        else:
            node = SkeletonEndSiteNode(node_name, channels, parent)
        # node.fixed = data["fixed"]
        node.index = data["index"]
        node.offset = np.array(data["offset"])
        node.rotation = np.array(data["rotation"])
        if node_name in skeleton.animated_joints:
            node.quaternion_frame_index = skeleton.animated_joints.index(node_name)
            node.fixed = False
        else:
            node.quaternion_frame_index = -1
            node.fixed = True
        print "load from json", node_name, node.quaternion_frame_index, data["index"], data[
            "fixed"], node.fixed, node.rotation

        skeleton.nodes[node_name] = node
        skeleton.nodes[node_name].children = []
        for c_desc in data["children"]:
            skeleton.nodes[node_name].children.append(self._create_node_from_desc(skeleton, c_desc, node))
        return node

    def _create_node_from_desc2(self, skeleton, data, node_name, parent):
        node_data = data["nodes"][node_name]
        channels = node_data["channels"]
        if parent is None:
            node = SkeletonRootNode(node_name, channels, parent)
        elif node_data["node_type"] == SKELETON_NODE_TYPE_JOINT:
            node = SkeletonJointNode(node_name, channels, parent)
        else:
            node = SkeletonEndSiteNode(node_name, channels, parent)
        # node.fixed = node_data["fixed"]
        node.index = node_data["index"]
        node.offset = np.array(node_data["offset"])
        node.rotation = np.array(node_data["rotation"])
        if node_name in skeleton.animated_joints:
            node.quaternion_frame_index = skeleton.animated_joints.index(node_name)
        else:
            node.quaternion_frame_index = -1
        node.children = []
        skeleton.nodes[node_name] = node
        for c_name in node_data["children"]:
            c_node = self._create_node_from_desc2(skeleton, data, c_name, node)
            node.children.append(c_node)
        return node



