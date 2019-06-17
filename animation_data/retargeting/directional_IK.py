
import copy
import numpy as np
import os
from ..skeleton_builder import SkeletonBuilder
from ..retargeting.directional_constraints_retargeting import align_ref_frame, retarget_motion
from ..body_plane import BodyPlane
from ..bvh import BVHWriter


class DirectionalIK(object):

    def __init__(self, target_bvhreader):
        self.target_bvhreader = target_bvhreader
        self.skeleton = SkeletonBuilder().load_from_bvh(self.target_bvhreader)
        self.skeleton_scale_factor = 1.0

    def set_scale_factor(self, scale):
        self.skeleton_scale_factor = scale
    
    def __call__(self, points, points_info, body_plane_joints):
        """
        
        Arguments:
            points {numpy.array 3d} -- point cloud motion data
            points_info {[type]} -- the list of joint sequence. It is used to compute bone direction
            e.g. 
                h6M_SKELETON = [
                    {'name': 'Hips', 'parent': None, 'index': 0},
                    {'name': 'HipRight', 'parent': 'Hips', 'index': 1},
                    {'name': 'KneeRight', 'parent': 'HipRight', 'index': 2},
                    {'name': 'FootRight', 'parent': 'KneeRight', 'index': 3},
                    {'name': 'ToeBaseRight', 'parent': 'FootRight', 'index': 4},
                    {'name': 'Site1', 'parent': 'ToeBaseRight', 'index': 5},
                    {'name': 'HipLeft', 'parent': 'Hips', 'index': 6},
                    {'name': 'KneeLeft', 'parent': 'HipLeft', 'index': 7},
                    {'name': 'FootLeft', 'parent': 'KneeLeft', 'index': 8},
                    {'name': 'ToeBaseLeft', 'parent': 'FootLeft', 'index': 9},
                    {'name': 'Site2', 'parent': 'ToeBaseLeft', 'index': 10},
                    {'name': 'Spine1', 'parent': 'Hips', 'index': 11},
                    {'name': 'Spine2', 'parent': 'Spine1', 'index': 12},
                    {'name': 'Neck', 'parent': 'Spine2', 'index': 13},
                    {'name': 'Head', 'parent': 'Neck', 'index': 14},
                    {'name': 'Site3', 'parent': 'Head', 'index': 15},
                    {'name': 'ShoulderLeft', 'parent': 'Neck', 'index': 16},
                    {'name': 'ElbowLeft', 'parent': 'ShoulderLeft', 'index': 17},
                    {'name': 'WristLeft', 'parent': 'ElbowLeft', 'index': 18},
                    {'name': 'HandLeft', 'parent': 'WristLeft', 'index': 19},
                    {'name': 'HandThumbLeft', 'parent': 'HandLeft', 'index': 20},
                    {'name': 'Site4', 'parent': 'HandThumbLeft', 'index': 21},
                    {'name': 'WristEndLeft', 'parent': 'HandLeft', 'index': 22},
                    {'name': 'Site5', 'parent': 'WristEndLeft', 'index': 23},
                    {'name': 'ShoulderRight', 'parent': 'Neck', 'index': 24},
                    {'name': 'ElbowRight', 'parent': 'ShoulderRight', 'index': 25},
                    {'name': 'WristRight', 'parent': 'ElbowRight', 'index': 26},
                    {'name': 'HandRight', 'parent': 'WristRight', 'index': 27},
                    {'name': 'HandThumbRight', 'parent': 'HandRight', 'index': 28},
                    {'name': 'Site6', 'parent': 'HandThumbRight', 'index': 29},
                    {'name': 'WristEndRight', 'parent': 'HandRight', 'index': 30},
                    {'name': 'Site7', 'parent': 'WristEndRight', 'index': 31},
                ]
        """
        n_frames = len(points)
        directional_constraints = self._create_directional_constraints(points, points_info, body_plane_joints)  ## directional constraints can be sparse
        self.out_frames = []
        root_index = next(item['index'] for item in points_info if item['name'] == self.skeleton.root)
        for i in range(n_frames):
            pose_dir = directional_constraints[i]['pose_dir']
            if i == 0:
                new_frame = copy.deepcopy(self.target_bvhreader.frames[0])  ## take the first frame as reference frame
                ref_frame = align_ref_frame(new_frame, pose_dir, self.skeleton, body_plane_joints)
                ## set global position
                ref_frame[0] = points[0, root_index, 0] * self.skeleton_scale_factor
                ref_frame[2] = points[0, root_index, 2] * self.skeleton_scale_factor
            else:
                new_frame = copy.deepcopy(self.out_frames[i-1])
                ref_frame = align_ref_frame(new_frame, pose_dir, self.skeleton, body_plane_joints)
                ref_frame[:3] = (points[i, root_index] - points[i-1, root_index]) * self.skeleton_scale_factor + self.out_frames[i-1][:3]
            retarget_motion(self.skeleton.root, directional_constraints[i], self.skeleton, ref_frame)
            self.out_frames.append(ref_frame)
    
    def save_as_bvh(self, save_path):
        """save retarget motion as bvh file
        
        Arguments:
            save_path {str}
        """
        if not os.path.exists(os.path.split(save_path)[0]):
            os.makedirs(save_path)
        BVHWriter(save_path, self.skeleton, self.out_frames, self.skeleton.frame_time, is_quaternion=False)

    def point_cloud_retargeting(self, points, points_info, body_plane_joints, joint_mapping, root_joint):
        """
        
        Arguments:
            points {[type]} -- [description]
            points_info {[type]} -- [description]
            body_plane_joints {[type]} -- [description]
        
        Keyword Arguments:
            joint_mapping {[type]} -- [description] (default: {None})
        """
        n_frames = len(points)
        directional_constraints = self._create_directional_constraints(points, points_info, body_plane_joints, joint_mapping)
        self.set_scale_factor(self.estimate_scale_factor(directional_constraints[0], body_plane_joints,
                                                         joint_mapping, points[0], points_info))
        self.out_frames = []      
        root_index = next(item['index'] for item in points_info if item['name'] == root_joint)
        for i in range(n_frames):
            pose_dir = directional_constraints[i]['pose_dir']
            if i == 0:
                new_frame = copy.deepcopy(self.target_bvhreader.frames[0])  ## take the first frame as reference frame
                ref_frame = align_ref_frame(new_frame, pose_dir, self.skeleton, body_plane_joints)
                ## set global position
                ref_frame[0] = points[0, root_index, 0] * self.skeleton_scale_factor
                ref_frame[2] = points[0, root_index, 2] * self.skeleton_scale_factor
            else:
                new_frame = copy.deepcopy(self.out_frames[i-1])
                ref_frame = align_ref_frame(new_frame, pose_dir, self.skeleton, body_plane_joints)
                ref_frame[:3] = (points[i, root_index] - points[i-1, root_index]) * self.skeleton_scale_factor + self.out_frames[i-1][:3]
            retarget_motion(self.skeleton.root, directional_constraints[i], self.skeleton, ref_frame)
            self.out_frames.append(ref_frame)    

    def estimate_scale_factor(self, frame_constraints, body_plane_joints, joint_mapping, frame_positions, points_info):
        """estimate the scale factor between point cloud data and target skeleton
        
        Arguments:
            frame_constraints {[type]} -- [description]
            body_plane_joints {[type]} -- [description]
            joint_mapping {[type]} -- [description]
            frame_positions {numpy.array} -- n_joints * 3
        
        Returns:
            float -- scaling_factor
        """
        target_body_plane_joints = [joint_mapping[joint] for joint in body_plane_joints]
        ref_frame = copy.deepcopy(self.target_bvhreader.frames[0])
        ref_frame = align_ref_frame(ref_frame, frame_constraints['pose_dir'], self.skeleton, target_body_plane_joints)
        print(frame_constraints)
        input()
        retarget_motion(self.skeleton.root, frame_constraints, self.skeleton, ref_frame)
        input()
        n_mapping_joints = len(joint_mapping)
        src_joint_mat = np.zeros((n_mapping_joints, 3))
        target_joint_mat = np.zeros((n_mapping_joints, 3))
        i = 0
        for key, value in joint_mapping.items():
            target_joint_mat[i] = self.skeleton.nodes[key].get_global_position_from_euler(ref_frame)
            src_joint_index = next(item['index'] for item in points_info if item['name'] == value)
            src_joint_mat[i] = frame_positions[src_joint_index]
            i += 1

        src_maximum = np.amax(src_joint_mat, axis=0)  ## get the largest number of each axis
        src_minimum = np.amin(src_joint_mat, axis=0)  ## get the smallest number of each axis
        target_maximum = np.amax(target_joint_mat, axis=0)
        target_minimum = np.amin(target_joint_mat, axis=0)
        src_max_diff = np.max(src_maximum - src_minimum)
        target_max_diff = np.max(target_maximum - target_minimum)
        scale = target_max_diff/src_max_diff
        return scale        

    def _create_directional_constraints(self, points, points_info, body_plane_joints, joint_mapping=None):
        """compute directional constraints from global positions
        
        Arguments:
            points {numpy.array3d} -- n_frames * n_joints * 3
            body_plane_joints {list} -- a list of joints to define the torso  (default: {None})
            joint_mapping {dict} -- joint mapping from src skeleton to target skeleton (default: {None})
        """
        n_frames, n_joints, _ = points.shape
        constraints = []
        if isinstance(points_info, list):
            for i in range(n_frames):
                frame_constraints = {}
                if body_plane_joints is not None:
                    torso_points = []
                    for joint in body_plane_joints:
                        joint_index = next(item['index'] for item in points_info if item['name'] == joint)
                        torso_points.append(points[i, joint_index])
                    body_plane = BodyPlane(torso_points)
                    dir_vec = np.array([body_plane.normal_vector[0], body_plane.normal_vector[2]])
                    frame_constraints['pose_dir'] = dir_vec / np.linalg.norm(dir_vec)
                else:
                    frame_constraints['pose_dir'] = None
                if joint_mapping is not None:
                    for joint in joint_mapping.keys():  
                        parent = DirectionalIK.get_joint_parent_in_joint_mapping(joint, joint_mapping, points_info)
                        if parent == 'Neck':
                            print(joint)
                        if parent is not None:
                            parent_index = next(item['index'] for item in points_info if item['name'] == parent)
                            joint_index = next(item['index'] for item in points_info if item['name'] == joint)
                            joint_dir = points[i, joint_index] - points[i, parent_index]
                            assert np.linalg.norm(joint_dir) != 0
                            if joint_mapping[parent] in frame_constraints.keys():
                                frame_constraints[joint_mapping[parent]][joint_mapping[joint]] = joint_dir / np.linalg.norm(joint_dir)
                            else:
                                frame_constraints[joint_mapping[parent]] = {joint_mapping[joint]: joint_dir / np.linalg.norm(joint_dir)}
                else:
                    for joint_des in points_info:
                        if joint_des['parent'] is not None:
                            parent_index = next(item['index'] for item in points_info if item['name'] == joint_des['parent'])
                            joint_dir = points[i, joint_des['index']] - points[i, parent_index]
                            assert np.linalg.norm(joint_dir) != 0  ## avoid 0 zero directional constraint, if there are zero-length bone, skip the child bone, link to grandchild bone
                            if joint_des['parent'] in frame_constraints.keys():
                                frame_constraints[joint_des['parent']][joint_des['name']] = joint_dir / np.linalg.norm(joint_dir)
                            else:
                                frame_constraints[joint_des['parent']] = {joint_des['name']: joint_dir / np.linalg.norm(joint_dir)}
                constraints.append(frame_constraints)
        else:
            raise IOError("Unknown data type! ")
        return constraints
    
    @staticmethod
    def get_joint_parent_in_joint_mapping(joint, joint_mapping, skeleton):
        """find the up level joint in the kinematic chain, search a list of dictionary
           if parent is not in joint_mapping, go to up level
        
        Arguments:
            joint {str} -- joint name
            joint_mapping {dict} -- joint mapping from src to target
            skeleton {list} 
        """
        joint_parent = next(item['parent'] for item in skeleton if item['name'] == joint)
        
        while joint_parent is not None:
            if joint_parent in joint_mapping.keys():
                return joint_parent
            else:
                joint_parent = next(item['parent'] for item in skeleton if item['name'] == joint_parent)    
        return None
