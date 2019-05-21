
import numpy as np
import collections
import os
import glob
import sys
import scipy.interpolate as interpolate
import scipy.ndimage.filters as filters

from ..animation_data import BVHReader, Skeleton, SkeletonBuilder
from ..animation_data.utils import convert_euler_frames_to_cartesian_frames, \
    convert_quat_frames_to_cartesian_frames, rotate_cartesian_frames_to_ref_dir, get_rotation_angles_for_vectors, \
    rotation_cartesian_frames, cartesian_pose_orientation, pose_orientation_euler, rotate_around_y_axis
from ..utilities import write_to_json_file, load_json_file 
from ..animation_data.quaternion import Quaternion
from ..utilities.motion_plane import Plane
from .Learning import RBF

def get_rotation_to_ref_direction(dir_vecs, ref_dir):
    rotations = []
    for dir_vec in dir_vecs:
        rotations.append(Quaternion.between(dir_vec, ref_dir))
    return rotations



class Preprocessing_Handler():

    def __init__(self, bvh_file_path, to_meters = 1, forward_dir = np.array([0,0,1]), shoulder_joints = [10, 20], hip_joints = [2, 27], fid_l = [4, 5], fid_r = [29, 30]):#, phase_label_file, footstep_label_file):
        self.bvh_file_path = bvh_file_path
        self.__forwards = []
        self.__root_rotations = []
        self.__global_positions = []
        self.__local_positions, self.__local_velocities = [],[]
        self.__ref_dir = forward_dir
        self.n_frames = 0
        self.n_joints = 0
        
        self.shoulder_joints = shoulder_joints
        self.hip_joints = hip_joints
        self.foot_left = fid_l
        self.foot_right = fid_r
        self.head = 16

        self.window = 60
        self.to_meters = to_meters
        
        
    def set_holden_parameters(self):
        self.shoulder_joints = [18, 25]
        self.hip_joints = [2, 7]
        self.foot_left = [4,5]
        self.foot_right = [9, 10]
        self.to_meters = 5.6444
        self.head = 16 # check this!

    def set_makehuman_parameters(self):
        self.shoulder_joints = [10, 20]
        self.hip_joints = [2, 27]
        self.foot_left = [4, 5]
        self.foot_right = [29, 30]
        self.to_meters = 1
        self.head = 16 # check this!

        

    def heightmap_fitting(patches_file):
        patches_database = np.load(patches_file) #(r'E:\gits\PFNN\patches.npz')
        patches = patches_database['X'].astype(np.float32)
        patches_coord = patches_database['C'].astype(np.float32)

    def load_motion(self, scale = 10, frame_rate_divisor = 2):
        print('Processing Clip %s' % self.bvh_file_path)
        
        bvhreader = BVHReader(self.bvh_file_path)
        skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
        cartesian_frames = convert_euler_frames_to_cartesian_frames(skeleton, bvhreader.frames)
        global_positions = cartesian_frames * scale

        self.__global_positions = global_positions[::frame_rate_divisor]
        self.n_frames, self.n_joints, _ = self.__global_positions.shape

    def load_gait (self, gait_file, frame_rate_divisor = 2, adjust_crouch = False):
        # bvh_file.replace('.bvh', '.gait')
        gait = np.loadtxt(gait_file)[::frame_rate_divisor]
        """ Merge Jog / Run and Crouch / Crawl """
        gait = np.concatenate([
            gait[:,0:1],
            gait[:,1:2],
            gait[:,2:3] + gait[:,3:4],
            gait[:,4:5] + gait[:,6:7],
            gait[:,5:6],
            gait[:,7:8]
        ], axis=-1)

        global_positions = self.__global_positions

        if adjust_crouch:
            crouch_low, crouch_high = 80, 130
            head = self.head
            gait[:-1,3] = 1 - np.clip((global_positions[:-1,head,1] - 80) / (130 - 80), 0, 1)
            gait[-1,3] = gait[-2,3]
        return gait

    def load_phase(self, phase_file, frame_rate_divisor = 2):
        # phase_file = data.replace('.bvh', '.phase')
        phase = np.loadtxt(phase_file)[::frame_rate_divisor]
        dphase = phase[1:] - phase[:-1]
        dphase[dphase < 0] = (1.0-phase[:-1]+phase[1:])[dphase < 0]

        return phase, dphase


    def get_forward_directions(self):
        sdr_l, sdr_r = self.shoulder_joints[0], self.shoulder_joints[1]
        hip_l, hip_r = self.hip_joints[0], self.hip_joints[1]
        global_positions = self.__global_positions

        if len(self.__forwards) == 0:
            across = (
                (global_positions[:,sdr_l] - global_positions[:,sdr_r]) + 
                (global_positions[:,hip_l] - global_positions[:,hip_r]))
            across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]
    
            """ Smooth Forward Direction """
            direction_filterwidth = 20
            forward = filters.gaussian_filter1d(
                np.cross(across, np.array([[0,1,0]])), direction_filterwidth, axis=0, mode='nearest')    
            self.__forwards = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]
        return self.__forwards

    def get_root_rotations(self):
        ref_dir = self.__ref_dir
        forward = self.get_forward_directions()

        if len(self.__root_rotations) == 0:
            forwards = self.get_forward_directions()
            self.__root_rotations = get_rotation_to_ref_direction(forward, ref_dir=ref_dir)
        return self.__root_rotations

    def __root_local_transform(self):
        if len(self.__local_positions) == 0:
            local_positions = self.__global_positions.copy()
            local_velocities = np.zeros(local_positions.shape)

            local_positions[:,:,0] = local_positions[:,:,0] - local_positions[:,0:1,0]
            local_positions[:,:,2] = local_positions[:,:,2] - local_positions[:,0:1,2]

            root_rotations = self.get_root_rotations()

            for i in range(self.n_frames - 1):
                for j in range(self.n_joints):
                    local_positions[i, j] = root_rotations[i] * local_positions[i, j]

                    local_velocities[i, j] = root_rotations[i] *  (self.__global_positions[i+1, j] - self.__global_positions[i, j])
            self.__local_positions = local_positions
            self.__local_velocities = local_velocities
        return self.__local_positions, self.__local_velocities


    def get_root_local_joint_positions(self):
        lp, _ = self.__root_local_transform()
        return lp

    def get_root_local_joint_velocities(self):
        _, lv = self.__root_local_transform()
        return lv

    def get_root_velocity(self):
        global_positions = self.__global_positions
        root_rotations = self.get_root_rotations()
        root_velocity = (global_positions[1:, 0:1] - global_positions[:-1, 0:1]).copy()

        for i in range(self.n_frames - 1):
            root_velocity[i, 0] = root_rotations[i+1] * root_velocity[i, 0]
        return root_velocity

    def get_rotational_velocity(self):
        root_rvelocity = np.zeros(self.n_frames - 1)
        root_rotations = self.get_root_rotations()

        for i in range(self.n_frames - 1):
            q = root_rotations[i+1] * (-root_rotations[i])
            root_rvelocity[i] = Quaternion.get_angle_from_quaternion(q, self.__ref_dir)

        return root_rvelocity

    def get_foot_concats(self, velfactor = np.array([0.05, 0.05])):
        fid_l, fid_r = self.foot_left, self.foot_right
        velfactor = velfactor / self.to_meters

        global_positions = self.__global_positions

        feet_l_x = (global_positions[1:,fid_l,0] - global_positions[:-1,fid_l,0])**2
        feet_l_y = (global_positions[1:,fid_l,1] - global_positions[:-1,fid_l,1])**2
        feet_l_z = (global_positions[1:,fid_l,2] - global_positions[:-1,fid_l,2])**2
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor)).astype(np.float)
    
        feet_r_x = (global_positions[1:,fid_r,0] - global_positions[:-1,fid_r,0])**2
        feet_r_y = (global_positions[1:,fid_r,1] - global_positions[:-1,fid_r,1])**2
        feet_r_z = (global_positions[1:,fid_r,2] - global_positions[:-1,fid_r,2])**2
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float)

        return feet_l, feet_r

    def get_trajectory(self, frame, start_from = -1):
        """
        
        Args: 
            start_from (int): -1 if whole window should be considered, value if specific start frame should be considered (e.g. i+1)
        """
        window = self.window
        global_positions = self.__global_positions
        forward = self.get_forward_directions()
        root_rotations = self.get_root_rotations()
        
        if start_from < 0:
            start_from = frame - self.window

        rootposs = (global_positions[start_from:frame+self.window:10,0] - global_positions[frame:frame+1,0]) ### 12*3
        rootdirs = forward[start_from:frame+self.window:10]
        for j in range(len(rootposs)):
            rootposs[j] = root_rotations[frame] * rootposs[j]
            rootdirs[j] = root_rotations[frame] * rootdirs[j]

        return rootposs, rootdirs
