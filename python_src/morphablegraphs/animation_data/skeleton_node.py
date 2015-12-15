__author__ = 'erhe01'

import numpy as np
from itertools import izip
from ..external.transformations import quaternion_matrix

SKELETON_NODE_TYPE_ROOT = 0
SKELETON_NODE_TYPE_JOINT = 1
SKELETON_NODE_TYPE_END_SITE = 2
ROTATION_TYPE_QUAT = 0
ROTATION_TYPE_EULER = 1


class SkeletonNodeBase(object):
    def __init__(self, node_name, parent=None):
        self.parent = parent
        self.node_name = node_name
        self.children = []
        self.index = -1
        self.quaternion_frame_index = -1
        self.node_type = None
        self.offset = [0.0, 0.0, 0.0]
        self.cached_global_matrix = None

    def clear_cached_global_matrix(self):
        self.cached_global_matrix = None

    def get_global_position(self, quaternion_frame, use_cache=False):
        global_matrix = self.get_global_matrix(quaternion_frame, use_cache)
        point = np.array([0, 0, 0, 1])
        point = np.dot(global_matrix, point)
        return point[:3]#.tolist()

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

    def get_local_matrix(self, quaternion_frame):
        pass

    def get_frame_parameters(self, frame, rotation_type):
        pass

    def get_number_of_frame_parameters(self, rotation_type):
        pass


class SkeletonRootNode(SkeletonNodeBase):
    def __init__(self, node_name, parent=None):
        super(SkeletonRootNode, self).__init__(node_name, parent)
        self.node_type = SKELETON_NODE_TYPE_ROOT

    def get_local_matrix(self, quaternion_frame):
        #print self.node_name, self.quaternion_frame_index
        local_matrix = quaternion_matrix(quaternion_frame[self.quaternion_frame_index: self.quaternion_frame_index + 4])
        local_translation = [t + o for t, o in izip(quaternion_frame[:3], self.offset)]
        local_matrix[:, 3] = local_translation + [1.0]
        return local_matrix

    def get_frame_parameters(self, frame, rotation_type):
        if rotation_type == ROTATION_TYPE_QUAT:
            if self.index >= 0:
                return frame[:7].tolist()
            else:
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        else:
            if self.index >= 0:
                return frame[:6].tolist()
            else:
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def get_number_of_frame_parameters(self, rotation_type):
        if rotation_type == ROTATION_TYPE_QUAT:
            return 7
        else:
            return 6


class SkeletonJointNode(SkeletonNodeBase):
    def __init__(self, node_name, parent=None):
        super(SkeletonJointNode, self).__init__(node_name, parent)
        self.node_type = SKELETON_NODE_TYPE_JOINT

    def get_local_matrix(self, quaternion_frame):
        self.node_name, self.quaternion_frame_index
        if self.quaternion_frame_index > -1:
            local_matrix = quaternion_matrix(quaternion_frame[self.quaternion_frame_index: self.quaternion_frame_index + 4])
        else:
            local_matrix = np.identity(4)
        local_matrix[:, 3] = self.offset + [1.0]
        return local_matrix

    def get_frame_parameters(self, frame, rotation_type):
        if rotation_type == ROTATION_TYPE_QUAT:
            if self.index >= 0:
                quaternion_frame_index = self.index * 4 + 3
                return frame[quaternion_frame_index:quaternion_frame_index+4].tolist()
            else:
                return [0.0, 0.0, 0.0, 1.0]
        else:
            if self.index >= 0:
                euler_frame_index = self.index * 3 + 3
                return frame[euler_frame_index:euler_frame_index+3].tolist()
            else:
                return [0.0, 0.0, 0.0]

    def get_number_of_frame_parameters(self, rotation_type):
        if rotation_type == ROTATION_TYPE_QUAT:
            return 4
        else:
            return 3


class SkeletonEndSiteNode(SkeletonNodeBase):
    def __init__(self, node_name, parent=None):
        super(SkeletonEndSiteNode, self).__init__(node_name, parent)
        self.node_type = SKELETON_NODE_TYPE_JOINT

    def get_local_matrix(self, quaternion_frame):
        local_matrix = np.identity(4)
        local_matrix[:, 3] = self.offset + [1.0]
        return local_matrix

    def get_frame_parameters(self, frame, rotation_type):
            return None

    def get_number_of_frame_parameters(self, rotation_type):
        return 0
