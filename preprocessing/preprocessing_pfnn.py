
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

""" Sampling Patch Heightmap """    
def patchfunc(P, Xp, to_meters, hscale=3.937007874, vscale=3.0, scale=False, ):  ##todo: figure out hscale
    if scale:
        hscale = hscale / to_meters
        vscale = vscale / to_meters
    Xp = Xp / hscale + np.array([P.shape[1]//2, P.shape[2]//2])
    
    A = np.fmod(Xp, 1.0)
    X0 = np.clip(np.floor(Xp).astype(np.int), 0, np.array([P.shape[1]-1, P.shape[2]-1]))
    X1 = np.clip(np.ceil (Xp).astype(np.int), 0, np.array([P.shape[1]-1, P.shape[2]-1]))
    
    H0 = P[:,X0[:,0],X0[:,1]]
    H1 = P[:,X0[:,0],X1[:,1]]
    H2 = P[:,X1[:,0],X0[:,1]]
    H3 = P[:,X1[:,0],X1[:,1]]
    
    HL = (1-A[:,0]) * H0 + (A[:,0]) * H2
    HR = (1-A[:,0]) * H1 + (A[:,0]) * H3
    
    return (vscale * ((1-A[:,1]) * HL + (A[:,1]) * HR))[...,np.newaxis]



def PREPROCESS_FOLDER(bvh_folder_path, output_file_name, base_handler, process_data_function, terrain_fitting = True, terrain_xslice = [], terrain_yslice = [], patches_path = "", split_files = False):
    P, X, Y = [], [], []
    bvhfiles = glob.glob(os.path.join(bvh_folder_path, '*.bvh'))
    data_folder = bvh_folder_path
    print(bvhfiles, os.path.join(bvh_folder_path, '*.bvh'))
    for data in bvhfiles:
        filename = os.path.split(data)[-1]
        data = os.path.join(data_folder, filename)

        handler = base_handler.copy()
        handler.bvh_file_path = data
        handler.load_motion()
        
        Pc, Xc, Yc = process_data_function(handler)
        Ptmp, Xtmp, Ytmp = handler.terrain_fitting(data.replace(".bvh", '_footsteps.txt'), patches_path, Pc, Xc, Yc, terrain_xslice, terrain_yslice)

        P.extend(Ptmp)
        X.extend(Xtmp)
        Y.extend(Ytmp)

    """ Clip Statistics """

    print('Total Clips: %i' % len(X))
    print('Shortest Clip: %i' % min(map(len,X)))
    print('Longest Clip: %i' % max(map(len,X)))
    print('Average Clip: %i' % np.mean(list(map(len,X))))

    """ Merge Clips """

    print('Merging Clips...')

    Xun = np.concatenate(X, axis=0)
    Yun = np.concatenate(Y, axis=0)
    Pun = np.concatenate(P, axis=0)

    print(Xun.shape, Yun.shape, Pun.shape)
        
    #np.savez_compressed(output_file_name, Xun=Xun, Yun=Yun, Pun=Pun)
    return Xun, Yun, Pun


class Preprocessing_Handler():
    """
    This class provides functionality to preprocess raw bvh data into a deep-learning favored format. 
    """

    def __init__(self, bvh_file_path, type = "flat", to_meters = 1, forward_dir = np.array([0,0,1]), shoulder_joints = [10, 20], hip_joints = [2, 27], fid_l = [4, 5], fid_r = [29, 30]):#, phase_label_file, footstep_label_file):
        self.bvh_file_path = bvh_file_path
        self.__global_positions = []
        
        self.__forwards = []
        self.__root_rotations = []
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
        self.type = type

    def reset_computations(self):
        self.__forwards = []
        self.__root_rotations = []
        self.__local_positions, self.__local_velocities = [],[]
       
    def copy(self):
        tmp = Preprocessing_Handler(self.bvh_file_path, self.type, self.to_meters, self.__ref_dir, self.shoulder_joints, self.hip_joints, self.foot_left, self.foot_right)
        tmp.__global_positions = np.array(self.__global_positions)
        return tmp
        
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

    def terrain_fitting(self, foot_step_path, patches_path, Pc, Xc, Yc, xslice, yslice):
        """
            xslice = slice((window*2)//10)*10+1, ((window*2)//10)*10+njoints*3+1, 3)
            yslice = slice(8+(window//10)*4+1, 8+(window//10)*4+njoints*3+1, 3)
        """
        P, X, Y = [], [], []
        with open(foot_step_path, 'r') as f:
            footsteps = f.readlines()
        
        patches_database = np.load(patches_path)
        patches = patches_database['X'].astype(np.float32)
        #patches_coord = patches_database['C'].astype(np.float32)
        
        """ For each Locomotion Cycle fit Terrains """
        for li in range(len(footsteps) - 1):
            curr, next = footsteps[li+0].split(' '), footsteps[li+1].split(' ')
            
            """ Ignore Cycles marked with '*' or not in range """
            if len(curr) == 3 and curr[2].strip().endswith('*'): continue
            if len(next) == 3 and next[2].strip().endswith('*'): continue
            if len(next) <  2: continue
            if int(curr[0])//2-self.window < 0: continue
            if int(next[0])//2-self.window >= len(Xc): continue


            """ Fit Heightmaps """
            slc = slice(int(curr[0])//2-self.window, int(next[0])//2+self.window+1)

            H, Hmean = self.process_heights(slc, patches)

            for h, hmean in zip(H, Hmean):

                slc2 = slice(int(curr[0])//2-self.window, int(next[0])//2-self.window+1)
                Xh, Yh = Xc[slc2].copy(), Yc[slc2].copy()

                """ Reduce Heights in Input/Output to Match"""

                Xh[:,xslice.start:xslice.stop:xslice.step] -= hmean[...,np.newaxis]
                Yh[:,yslice.start:yslice.stop:yslice.step] -= hmean[...,np.newaxis]
                # xo_s, xo_e = ((self.window * 2) // 10) * 10 + 1, ((self.window * 2) // 10) * 10 + self.n_joints * 3 + 1
                # yo_s, yo_e = 8 + (self.window // 10) * 4 + 1, 8 + (self.window // 10) * 4 + self.n_joints * 3 + 1

                # Xh[:,xo_s:xo_e:3] -= hmean[...,np.newaxis]
                # Yh[:,yo_s:yo_e:3] -= hmean[...,np.newaxis]

                #Xh[:,xslice[0]:xslice[1]:xslice[2]] -= hmean[...,np.newaxis]
                #Yh[:,yslice[0]:yslice[1]:yslice[2]] -= hmean[...,np.newaxis]
                Xh = np.concatenate([Xh, h - hmean[...,np.newaxis]], axis=-1)

                """ Append to Data """

                P.append(np.hstack([0.0, Pc[slc2][1:-1], 1.0]).astype(np.float32))
                X.append(Xh.astype(np.float32))
                Y.append(Yh.astype(np.float32))
        return P, X, Y

    def process_heights(self, slice, patches, nsamples = 10):
        tmp_handler = self.copy()
        tmp_handler.__global_positions = self.__global_positions[slice]
        tmp_handler.n_frames = len(tmp_handler.__global_positions)


        forward = tmp_handler.get_forward_directions()
        root_rotation = tmp_handler.get_root_rotations()
        global_positions = tmp_handler.__global_positions
        
        """ Toe and Heel Heights """
        feet_l, feet_r = tmp_handler.get_foot_concats()
        feet_l = np.concatenate([feet_l, feet_l[-1:]], axis=0)
        feet_r = np.concatenate([feet_r, feet_r[-1:]], axis=0)
        feet_l = feet_l.astype(np.bool)
        feet_r = feet_r.astype(np.bool)

    


        toe_h, heel_h = 4.0 / self.to_meters, 5.0 / self.to_meters
        
        """ Foot Down Positions """
        fid_l, fid_r = self.foot_left, self.foot_right
        feet_down = np.concatenate([
            global_positions[feet_l[:,0],fid_l[0]] - np.array([0, heel_h, 0]),
            global_positions[feet_l[:,1],fid_l[1]] - np.array([0,  toe_h, 0]),
            global_positions[feet_r[:,0],fid_r[0]] - np.array([0, heel_h, 0]),
            global_positions[feet_r[:,1],fid_r[1]] - np.array([0,  toe_h, 0])
        ], axis=0)

        """ Foot Up Positions """
        feet_up = np.concatenate([
            global_positions[~feet_l[:,0],fid_l[0]] - np.array([0, heel_h, 0]),
            global_positions[~feet_l[:,1],fid_l[1]] - np.array([0,  toe_h, 0]),
            global_positions[~feet_r[:,0],fid_r[0]] - np.array([0, heel_h, 0]),
            global_positions[~feet_r[:,1],fid_r[1]] - np.array([0,  toe_h, 0])
        ], axis=0)
        
        """ Down Locations """
        feet_down_xz = np.concatenate([feet_down[:,0:1], feet_down[:,2:3]], axis=-1)
        feet_down_xz_mean = feet_down_xz.mean(axis=0)
        feet_down_y = feet_down[:,1:2]
        feet_down_y_mean = feet_down_y.mean(axis=0)
        feet_down_y_std  = feet_down_y.std(axis=0)

        """ Up Locations """
        feet_up_xz = np.concatenate([feet_up[:,0:1], feet_up[:,2:3]], axis=-1)
        feet_up_y = feet_up[:,1:2]


        if len(feet_down_xz) == 0:
            """ No Contacts """
            terr_func = lambda Xp: np.zeros_like(Xp)[:,:1][np.newaxis].repeat(nsamples, axis=0)

        elif self.type == 'flat':
            """ Flat """
            terr_func = lambda Xp: np.zeros_like(Xp)[:,:1][np.newaxis].repeat(nsamples, axis=0) + feet_down_y_mean

        else:
            """ Terrain Heights """
            terr_down_y = patchfunc(patches, feet_down_xz - feet_down_xz_mean, self.to_meters)
            terr_down_y_mean = terr_down_y.mean(axis=1)
            terr_down_y_std  = terr_down_y.std(axis=1)
            terr_up_y = patchfunc(patches, feet_up_xz - feet_down_xz_mean, self.to_meters)


            """ Fitting Error """
            terr_down_err = 0.1 * ((
                (terr_down_y - terr_down_y_mean[:,np.newaxis]) -
                (feet_down_y - feet_down_y_mean)[np.newaxis])**2)[...,0].mean(axis=1)
        
            terr_up_err = (np.maximum(
                (terr_up_y - terr_down_y_mean[:,np.newaxis]) -
                (feet_up_y - feet_down_y_mean)[np.newaxis], 0.0)**2)[...,0].mean(axis=1)

            """ Jumping Error """
            if self.type == 'jumpy':
                terr_over_minh = 5.0
                if scale:
                    terr_over_minh = terr_over_minh / to_meters
                terr_over_err = (np.maximum(
                    ((feet_up_y - feet_down_y_mean)[np.newaxis] - terr_over_minh) -
                    (terr_up_y - terr_down_y_mean[:,np.newaxis]), 0.0)**2)[...,0].mean(axis=1)
            else:
                terr_over_err = 0.0

            """ Fitting Terrain to Walking on Beam """
            if type == 'beam':

                beam_samples = 1
                beam_min_height = 40.0 / self.to_meters

                beam_c = global_positions[:,0]
                beam_c_xz = np.concatenate([beam_c[:,0:1], beam_c[:,2:3]], axis=-1)
                beam_c_y = patchfunc(patches, beam_c_xz - feet_down_xz_mean)

                beam_o = (
                    beam_c.repeat(beam_samples, axis=0) + np.array([50, 0, 50]) / to_meters *
                    rng.normal(size=(len(beam_c)*beam_samples, 3)))

                beam_o_xz = np.concatenate([beam_o[:,0:1], beam_o[:,2:3]], axis=-1)
                beam_o_y = patchfunc(patches, beam_o_xz - feet_down_xz_mean)

                beam_pdist = np.sqrt(((beam_o[:,np.newaxis] - beam_c[np.newaxis,:])**2).sum(axis=-1))
                beam_far = (beam_pdist > 15 / to_meters).all(axis=1)
                terr_beam_err = (np.maximum(beam_o_y[:,beam_far] - 
                    (beam_c_y.repeat(beam_samples, axis=1)[:,beam_far] - 
                     beam_min_height), 0.0)**2)[...,0].mean(axis=1)
            else:
                terr_beam_err = 0.0


            """ Final Fitting Error """
        
            terr = terr_down_err + terr_up_err + terr_over_err + terr_beam_err
        
            """ Best Fitting Terrains """
        
            terr_ids = np.argsort(terr)[:nsamples]
            terr_patches = patches[terr_ids]
            terr_basic_func = lambda Xp: (
                (patchfunc(terr_patches, Xp - feet_down_xz_mean) - 
                terr_down_y_mean[terr_ids][:,np.newaxis]) + feet_down_y_mean)
        
            """ Terrain Fit Editing """
        
            terr_residuals = feet_down_y - terr_basic_func(feet_down_xz)
            terr_fine_func = [RBF(smooth=0.1, function='linear') for _ in range(nsamples)]
            for i in range(nsamples): terr_fine_func[i].fit(feet_down_xz, terr_residuals[i])
            terr_func = lambda Xp: (terr_basic_func(Xp) + np.array([ff(Xp) for ff in terr_fine_func]))

        """ Get Trajectory Terrain Heights """
    
        root_offsets_c = global_positions[:,0]
        root_offsets_r = np.zeros(root_offsets_c.shape)
        root_offsets_l = np.zeros(root_offsets_c.shape)
        for i in range(len(root_rotation)):
            root_offsets_r[i] = (-root_rotation[i]) * np.array([+25, 0, 0]) + root_offsets_c[i]
            root_offsets_l[i] = (-root_rotation[i]) * np.array([-25, 0, 0]) + root_offsets_c[i]

        root_heights_c = terr_func(root_offsets_c[:,np.array([0,2])])[...,0]
        root_heights_r = terr_func(root_offsets_r[:,np.array([0,2])])[...,0]
        root_heights_l = terr_func(root_offsets_l[:,np.array([0,2])])[...,0]
    
        """ Find Trajectory Heights at each Window """
    
        root_terrains = []
        root_averages = []
        for i in range(self.window, len(global_positions)-self.window, 1): 
            root_terrains.append(
                np.concatenate([
                    root_heights_r[:,i-self.window:i+self.window:10],
                    root_heights_c[:,i-self.window:i+self.window:10],
                    root_heights_l[:,i-self.window:i+self.window:10]], axis=1))
            root_averages.append(root_heights_c[:,i-self.window:i+ self.window:10].mean(axis=1))
     
        root_terrains = np.swapaxes(np.array(root_terrains), 0, 1)
        root_averages = np.swapaxes(np.array(root_averages), 0, 1)

        return root_terrains, root_averages