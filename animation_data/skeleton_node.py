__author__ = 'erhe01'

import numpy as np
from transformations import quaternion_matrix, euler_matrix, euler_from_matrix, quaternion_from_matrix, quaternion_about_axis
from .constants import ROTATION_TYPE_EULER, ROTATION_TYPE_QUATERNION

SKELETON_NODE_TYPE_ROOT = 0
SKELETON_NODE_TYPE_JOINT = 1
SKELETON_NODE_TYPE_END_SITE = 2
ORIGIN = point = np.array([0, 0, 0, 1])


def create_euler_matrix(angles, order):
    angles = np.radians(angles)
    if order[0] == 'Xrotation':
        if order[1] == 'Yrotation':
            local_matrix = euler_matrix(*angles, axes='rxyz')
        elif order[1] == 'Zrotation':
            local_matrix = euler_matrix(*angles, axes='rxzy')
    elif order[0] == 'Yrotation':
        if order[1] == 'Xrotation':
            local_matrix = euler_matrix(*angles, axes='ryxz')
        elif order[1] == 'Zrotation':
            local_matrix = euler_matrix(*angles, axes='ryzx')
    elif order[0] == 'Zrotation':
        if order[1] == 'Xrotation':
            local_matrix = euler_matrix(*angles, axes='rzxy')
        elif order[1] == 'Yrotation':
            local_matrix = euler_matrix(*angles, axes='rzyx')
    else:
        raise ValueError("Unknown rotation order")
    return local_matrix

class SkeletonNodeBase(object):
    def __init__(self, node_name, channels, parent=None, level=0, channel_indices=None):
        self.parent = parent
        self.node_name = node_name
        self.channels = channels
        self.level = level
        self.children = []
        self.index = -1
        self.quaternion_frame_index = -1
        self.euler_frame_index = -1
        self.node_type = None
        self.offset = [0.0, 0.0, 0.0]
        self.rotation = np.array([1.0, 0.0, 0.0, 0.0])
        self.euler_angles = np.array([0.0, 0.0, 0.0])
        self.translation = np.array([0.0, 0.0, 0.0])
        self.fixed = True
        self.cached_global_matrix = None
        self.joint_constraint = None
        self.stiffness = 0
        self.rotation_order = [c for c in self.channels if "rotation" in c]
        self.channel_indices = channel_indices
        self.translation_order = [c for c in self.channels if "position" in c]
        if self.channel_indices is not None:
            self.rotation_channel_indices = [self.channel_indices[self.channels.index(r)] for r in self.rotation_order]
            self.translation_channel_indices = [self.channel_indices[self.channels.index(t)] for t in self.translation_order]

    def clear_cached_global_matrix(self):
        self.cached_global_matrix = None

    def get_global_position(self, quaternion_frame, use_cache=False):
        global_matrix = self.get_global_matrix(quaternion_frame, use_cache)
        point = np.dot(global_matrix, ORIGIN)
        return point[:3]

    def get_global_position_from_euler(self, euler_frame, use_cache=False):
        global_matrix = self.get_global_matrix_from_euler_frame(euler_frame, use_cache)
        point = np.dot(global_matrix, ORIGIN)
        return point[:3]

    def get_local_position_from_euler(self, euler_frame):
        local_matrix = self.get_local_matrix_from_euler(euler_frame)
        point = np.dot(local_matrix, ORIGIN)
        return point[:3]

    def get_global_orientation_quaternion(self, quaternion_frame, use_cache=False):
        global_matrix = self.get_global_matrix(quaternion_frame, use_cache)
        return quaternion_from_matrix(global_matrix)

    def get_global_orientation_euler(self, euler_frame, use_cache=False):
        global_matrix = self.get_global_matrix_from_euler_frame(euler_frame, use_cache)
        return euler_from_matrix(global_matrix)

    def get_global_matrix(self, quaternion_frame, use_cache=False):
        if self.cached_global_matrix is not None and use_cache:
            return self.cached_global_matrix
        else:
            if self.parent is not None:
                parent_matrix = self.parent.get_global_matrix(quaternion_frame, use_cache)
                self.cached_global_matrix = np.dot(parent_matrix, self.get_local_matrix(quaternion_frame))
                return self.cached_global_matrix
            else:
                self.cached_global_matrix = self.get_local_matrix(quaternion_frame)
                return self.cached_global_matrix

    def get_global_matrix_from_euler_frame(self, euler_frame, use_cache=False):
        if self.parent is not None:
            parent_matrix = self.parent.get_global_matrix_from_euler_frame(euler_frame)
            self.cached_global_matrix = np.dot(parent_matrix, self.get_local_matrix_from_euler(euler_frame))
            return self.cached_global_matrix
        else:
            self.cached_global_matrix = self.get_local_matrix_from_euler(euler_frame)
            return self.cached_global_matrix

    def get_global_matrix_from_anim_frame(self, frame, use_cache=False):
        if self.cached_global_matrix is not None and use_cache:
            return self.cached_global_matrix
        else:
            if self.parent is not None:
                parent_matrix = self.parent.get_global_matrix2(frame, use_cache)
                self.cached_global_matrix = np.dot(parent_matrix, frame[self.node_name])
                return self.cached_global_matrix
            else:
                self.cached_global_matrix = frame[self.node_name]
                return self.cached_global_matrix

    def get_local_matrix(self, quaternion_frame):
        pass

    def get_local_matrix_from_euler(self, euler_frame):
        pass

    def get_frame_parameters(self, frame, rotation_type):
        pass

    def get_number_of_frame_parameters(self, rotation_type):
        pass
    
    def to_unity_format(self, joints, animated_joint_list, joint_name_map=None):
        joint_desc = dict()
        joint_desc["name"] = self.node_name
        if joint_name_map is not None:
            if self.node_name in joint_name_map:
                joint_desc["targetName"] = joint_name_map[self.node_name]
            else:
                joint_desc["targetName"] = "none"
        else:
            joint_desc["targetName"] = self.node_name
        joint_desc["children"] = []
        joint_desc["offset"] = {"x": -self.offset[0], "y": self.offset[1], "z": self.offset[2]}
        joints.append(joint_desc)
        for c in self.children:
            if c.node_name in animated_joint_list:
                joint_desc["children"].append(c.node_name)
                c.to_unity_format(joints, animated_joint_list, joint_name_map=joint_name_map)

    def get_fk_chain_list(self):
        pass

    def get_parent_name(self, animated_joint_list):
        '''
        :return: string, the name of parent node. If parent node is None, return None
        '''
        if self.parent is None:
            return None
        else:
            if animated_joint_list is None:
                    return self.parent.node_name
            else:
                if self.parent.node_name in animated_joint_list:
                    return self.parent.node_name
                else:
                    return self.parent.get_parent_name(animated_joint_list)


class SkeletonRootNode(SkeletonNodeBase):
    def __init__(self, node_name, channels, parent=None, level=0, channel_indices=None):
        super(SkeletonRootNode, self).__init__(node_name, channels, parent, level, channel_indices)
        self.node_type = SKELETON_NODE_TYPE_ROOT

    def get_local_matrix(self, quaternion_frame):
        local_matrix = quaternion_matrix(quaternion_frame[3:7])
        local_matrix[:3, 3] = [t + o for t, o in zip(quaternion_frame[:3], self.offset)]
        return local_matrix

    def get_local_matrix_from_euler(self, euler_frame):
        euler_frame = np.asarray(euler_frame)
        if not self.rotation_order is None and not self.rotation_channel_indices is None:
            local_matrix = create_euler_matrix(euler_frame[self.rotation_channel_indices], self.rotation_order)
        else:
            raise ValueError("no rotation order!")
        if len(self.translation_channel_indices)> 0:
            local_matrix[:3, 3] = [t + o for t, o in zip(euler_frame[self.translation_channel_indices], self.offset)]
        else:
            local_matrix[:3, 3] = self.offset
        return local_matrix

    def get_frame_parameters(self, frame, rotation_type):
        if not self.fixed:
            return frame[:7].tolist()
        else:
            return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    def get_euler_frame_parameters(self, euler_frame):
        if not self.fixed:
            return euler_frame[:6].tolist()
        else:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def get_number_of_frame_parameters(self, rotation_type):
        if rotation_type == ROTATION_TYPE_QUATERNION:
            return 7
        else:
            return 6

    def get_fk_chain_list(self):
        return [self.node_name]

class SkeletonJointNode(SkeletonNodeBase):
    def __init__(self, node_name, channels, parent=None, level=0, channel_indices=None):
        super(SkeletonJointNode, self).__init__(node_name, channels, parent, level, channel_indices)
        self.node_type = SKELETON_NODE_TYPE_JOINT

    def get_local_matrix(self, quaternion_frame):
        if not self.fixed:
            frame_index = self.quaternion_frame_index * 4 + 3
            m = quaternion_matrix(quaternion_frame[frame_index: frame_index + 4])
        else:
            m = quaternion_matrix(self.rotation)
        m[:3, 3] = self.offset
        return m

    def get_local_matrix_from_euler(self, euler_frame):
        if not self.fixed:
            if not self.rotation_order is None:
                local_matrix = create_euler_matrix(euler_frame[self.rotation_channel_indices], self.rotation_order)
            else:
                raise ValueError("no rotation order!")
        else:
            local_matrix = euler_matrix(*np.radians(self.euler_angles), axes='rxyz')
        # local_matrix[:3, 3] = self.offset
        if self.translation_channel_indices != []:
            local_matrix[:3, 3] = [t + o for t, o in zip(euler_frame[self.translation_channel_indices], self.offset)]
        else:
            local_matrix[:3, 3] = self.offset
        return local_matrix

    def get_frame_parameters(self, frame, rotation_type):
        if rotation_type == ROTATION_TYPE_QUATERNION:
            if not self.fixed:
                frame_index = self.quaternion_frame_index * 4 + 3
                return frame[frame_index:frame_index+4].tolist()
            else:
                return self.rotation
        else:
            if not self.fixed:
                frame_index = self.quaternion_frame_index * 3 + 3
                return frame[frame_index:frame_index+3].tolist()
            else:
                return [0.0, 0.0, 0.0]

    def get_number_of_frame_parameters(self, rotation_type):
        if rotation_type == ROTATION_TYPE_QUATERNION:
            return 4
        else:
            return 3

    def get_fk_chain_list(self):
        fk_chain = [self.node_name]
        if self.parent is not None:
            fk_chain += self.parent.get_fk_chain_list()
        return fk_chain

class SkeletonEndSiteNode(SkeletonNodeBase):
    def __init__(self, node_name, channels, parent=None, level=0):
        super(SkeletonEndSiteNode, self).__init__(node_name, channels, parent, level)
        self.node_type = SKELETON_NODE_TYPE_END_SITE

    def get_local_matrix(self, quaternion_frame):
        local_matrix = np.identity(4)
        local_matrix[:3, 3] = self.offset
        return local_matrix

    def get_local_matrix_from_euler(self, euler_frame):
        local_matrix = np.identity(4)
        local_matrix[:3, 3] = self.offset
        return local_matrix

    def get_frame_parameters(self, frame, rotation_type):
            return None

    def get_number_of_frame_parameters(self, rotation_type):
        return 0

    def get_fk_chain_list(self):
        fk_chain = []
        if self.parent is not None:
            fk_chain += self.parent.get_fk_chain_list()
        return fk_chain
