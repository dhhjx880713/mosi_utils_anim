__author__ = 'erhe01'

import numpy as np
from itertools import izip
from ..external.transformations import quaternion_matrix, euler_matrix, euler_from_matrix, quaternion_from_matrix
from . import ROTATION_TYPE_EULER, ROTATION_TYPE_QUATERNION

SKELETON_NODE_TYPE_ROOT = 0
SKELETON_NODE_TYPE_JOINT = 1
SKELETON_NODE_TYPE_END_SITE = 2


class SkeletonNodeBase(object):
    ORIGIN = point = np.array([0, 0, 0, 1])
    def __init__(self, node_name, channels, parent=None):
        self.parent = parent
        self.node_name = node_name
        self.channels = channels
        self.children = []
        self.index = -1
        self.quaternion_frame_index = -1
        self.euler_frame_index = -1
        self.node_type = None
        self.offset = [0.0, 0.0, 0.0]
        self.rotation = np.array([1.0, 0.0, 0.0, 0.0])
        self.euler_angles = np.array([0.0, 0.0, 0.0])
        self.fixed = True
        self.cached_global_matrix = None

    def clear_cached_global_matrix(self):
        self.cached_global_matrix = None

    def get_global_position(self, quaternion_frame, use_cache=False):
        global_matrix = self.get_global_matrix(quaternion_frame, use_cache)
        point = np.dot(global_matrix, self.ORIGIN)
        return point[:3]#.tolist()

    def get_global_position_from_euler(self, euler_frame, use_cache=False):
        global_matrix = self.get_global_matrix_from_euler_frame(euler_frame, use_cache)
        point = np.dot(global_matrix, self.ORIGIN)
        return point[:3]

    def get_global_orientation_quaternion(self, quaternion_frame, use_cache=False):
        global_matrix = self.get_global_matrix(quaternion_frame, use_cache)
        return quaternion_from_matrix(global_matrix)

    def get_global_orientation_euler(self, quaternion_frame, use_cache=False):
        global_matrix = self.get_global_matrix(quaternion_frame, use_cache)
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
        if joint_name_map is not None and self.node_name in joint_name_map:
            joint_desc["targetName"] = joint_name_map[self.node_name]
        else:
            joint_desc["targetName"] = "none"
        joint_desc["children"] = []
        joints.append(joint_desc)
        for c in self.children:
            if c.node_name in animated_joint_list:
                joint_desc["children"].append(c.node_name)
                c.to_unity_format(joints, animated_joint_list, joint_name_map=joint_name_map)


class SkeletonRootNode(SkeletonNodeBase):
    def __init__(self, node_name, channels, parent=None):
        super(SkeletonRootNode, self).__init__(node_name, channels, parent)
        self.node_type = SKELETON_NODE_TYPE_ROOT

    def get_local_matrix(self, quaternion_frame):
        local_matrix = quaternion_matrix(quaternion_frame[3:7])
        local_matrix[:3, 3] = [t + o for t, o in izip(quaternion_frame[:3], self.offset)]
        return local_matrix

    def get_local_matrix_from_euler(self, euler_frame):
        local_matrix = euler_matrix(*np.radians(euler_frame[3:6]), axes='rxyz')
        local_matrix[:3, 3] = [t + o for t, o in izip(euler_frame[:3], self.offset)]
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


class SkeletonJointNode(SkeletonNodeBase):
    def __init__(self, node_name, channels, parent=None):
        super(SkeletonJointNode, self).__init__(node_name, channels, parent)
        self.node_type = SKELETON_NODE_TYPE_JOINT

    def get_local_matrix(self, quaternion_frame):
        #self.node_name, self.quaternion_frame_index
        if not self.fixed:
            frame_index = self.quaternion_frame_index * 4 + 3
            local_matrix = quaternion_matrix(quaternion_frame[frame_index: frame_index + 4])
        else:
            local_matrix = quaternion_matrix(self.rotation)
        local_matrix[:3, 3] = self.offset
        return local_matrix

    def get_local_matrix_from_euler(self, euler_frame):
        if not self.fixed:
            frame_index = self.euler_frame_index * 3 + 3
            local_matrix = euler_matrix(*np.radians(euler_frame[frame_index: frame_index + 3]), axes='rxyz')
        else:
            local_matrix = euler_matrix(*np.radians(self.euler_angles), axes='rxyz')
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


class SkeletonEndSiteNode(SkeletonNodeBase):
    def __init__(self, node_name, channels, parent=None):
        super(SkeletonEndSiteNode, self).__init__(node_name, channels, parent)
        self.node_type = SKELETON_NODE_TYPE_END_SITE

    def get_local_matrix(self, quaternion_frame):
        local_matrix = np.identity(4)
        local_matrix[:3, 3] = self.offset
        return local_matrix

    def get_frame_parameters(self, frame, rotation_type):
            return None

    def get_number_of_frame_parameters(self, rotation_type):
        return 0
