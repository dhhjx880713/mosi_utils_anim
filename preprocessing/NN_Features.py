
import numpy as np
import collections
import os
import glob
import sys
import scipy.interpolate as interpolate
import scipy.ndimage.filters as filters

from ..animation_data.utils import convert_euler_frames_to_cartesian_frames, quaternion_from_matrix, quaternion_inverse, quaternion_multiply, quaternion_matrix
#from ..animation_data.utils import convert_euler_frames_to_cartesian_frames, \
#    convert_quat_frames_to_cartesian_frames, rotate_cartesian_frames_to_ref_dir, get_rotation_angles_for_vectors, \
#    rotation_cartesian_frames, cartesian_pose_orientation, pose_orientation_euler, rotate_around_y_axis
from ..utilities import write_to_json_file, load_json_file
from ..utilities.motion_plane import Plane

from ..animation_data import BVHReader, Skeleton, SkeletonBuilder
from ..animation_data.quaternion import Quaternion
from .Learning import RBF
import json

def get_rotation_to_ref_direction(dir_vecs, ref_dir):
    rotations = []
    for dir_vec in dir_vecs:
        q = Quaternion.between(dir_vec, ref_dir)
        rotations.append(q)
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
    """
    This function processes a whole folder of BVH files. The "process_data_function" is called for each file and creates the actual training data (x,y,p). 
    process_data_function(handler : Preprocessing_handler) -> [P, X, Y]
    The handler starts as a copy from base_handler

    Pseudo-Code: 
        for bvh_file in folder:
            handler = base_handler.copy()
            handler.bvh_file = bvh_file
            handler.load_data()

            Pc, Xc, Yc = process_data_function(handler)
            merge (P,X,Y) (Pc, Xc, Yc)
        
        save file
        return X, Y, P

        :param bvh_folder_path: path to folder containing bvh files and labels
        :param output_file_name: output file to which the processed data should be written
        :param base_handler: base-handler containing configuration information (e.g. window size, # joints, etc.)
        :param process_data_function: process data function to create a single x-y pair (+p)
        :param terrain_fitting=True: If true, terrain fitting is applied
        :param terrain_xslice=[]: xslice definining joint positions in X that are affected by terrain fitting
        :param terrain_yslice=[]: yslice definining joint positions in Y that are affected by terrain fitting
        :param patches_path="": path to compressed patches file. 
        :param split_files=False: not yet implemented


    Example for a single file: 

        def process_data(handler):
            bvh_path = handler.bvh_file_path
            phase_path = bvh_path.replace('.bvh', '.phase')
            gait_path = bvh_path.replace(".bvh", ".gait")

            Pc, Xc, Yc = [], [], []


            gait = handler.load_gait(gait_path, adjust_crouch=True)
            phase, dphase = handler.load_phase(phase_path)

            local_positions = handler.get_root_local_joint_positions()
            local_velocities = handler.get_root_local_joint_velocities()

            root_velocity = handler.get_root_velocity()
            root_rvelocity = handler.get_rotational_velocity()

            feet_l, feet_r = handler.get_foot_concats()

            for i in range(handler.window, handler.n_frames - handler.window - 1, 1):
                rootposs,rootdirs = handler.get_trajectory(i)
                rootgait = gait[i - handler.window:i+handler.window:10]

                Pc.append(phase[i])

                Xc.append(np.hstack([
                        rootposs[:,0].ravel(), rootposs[:,2].ravel(), # Trajectory Pos
                        rootdirs[:,0].ravel(), rootdirs[:,2].ravel(), # Trajectory Dir
                        rootgait[:,0].ravel(), rootgait[:,1].ravel(), # Trajectory Gait
                        rootgait[:,2].ravel(), rootgait[:,3].ravel(), 
                        rootgait[:,4].ravel(), rootgait[:,5].ravel(), 
                        local_positions[i-1].ravel(),  # Joint Pos
                        local_velocities[i-1].ravel(), # Joint Vel
                    ]))

                rootposs_next, rootdirs_next = handler.get_trajectory(i + 1, i + 1)

                Yc.append(np.hstack([
                    root_velocity[i,0,0].ravel(), # Root Vel X
                    root_velocity[i,0,2].ravel(), # Root Vel Y
                    root_rvelocity[i].ravel(),    # Root Rot Vel
                    dphase[i],                    # Change in Phase
                    np.concatenate([feet_l[i], feet_r[i]], axis=-1), # Contacts
                    rootposs_next[:,0].ravel(), rootposs_next[:,2].ravel(), # Next Trajectory Pos
                    rootdirs_next[:,0].ravel(), rootdirs_next[:,2].ravel(), # Next Trajectory Dir
                    local_positions[i].ravel(),  # Joint Pos
                    local_velocities[i].ravel(), # Joint Vel
                    ]))

            return np.array(Pc), np.array(Xc), np.array(Yc)

        bvh_path = "./test_files/LocomotionFlat04_000.bvh"
        data_folder = "./test_files"
        patches_path = "./test_files/patches.npz"
        
        handler = FeatureExtractor
    (bvh_path)
        handler.load_motion()
        Pc, Xc, Yc = process_data(handler)
        
        xslice = slice(((handler.window*2)//10)*10+1, ((handler.window*2)//10)*10+handler.n_joints*3+1, 3)
        yslice = slice(8+(handler.window//10)*4+1, 8+(handler.window//10)*4+handler.n_joints*3+1, 3)
        P, X, Y = handler.terrain_fitting(bvh_path.replace(".bvh", '_footsteps.txt'), patches_path, Pc, Xc, Yc, xslice, yslice)

    """

    P, X, Y = [], [], []
    bvhfiles = glob.glob(os.path.join(bvh_folder_path, '*.bvh'))
    data_folder = bvh_folder_path
    print(bvhfiles, os.path.join(bvh_folder_path, '*.bvh'))
    config = {}
    for data in bvhfiles:
        filename = os.path.split(data)[-1]
        data = os.path.join(data_folder, filename)

        handler = base_handler.copy()
        handler.bvh_file_path = data
        handler.load_motion()
        
        Pc, Xc, Yc, config = process_data_function(handler)
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
        
    np.savez_compressed(output_file_name, Xun=Xun, Yun=Yun, Pun=Pun)
    with open(output_file_name + ".json", "w") as f:
        json.dump(config, f)
    return Xun, Yun, Pun, config


class FeatureExtractor():
    def __init__(self, bvh_file_path, type = "flat", to_meters = 1, forward_dir = np.array([0.0,0.0,1.0]), shoulder_joints = [10, 20], hip_joints = [2, 27], fid_l = [4, 5], fid_r = [29, 30]):#, phase_label_file, footstep_label_file):
        """

        This class provides functionality to preprocess raw bvh data into a deep-learning favored format. 
        It does not actually transfer the data, but provides the possibilitie to create these. An additional, lightweight process_data function is required. 

        Default configurations can be loaded using set_holden_parameters and set_makehuman_parameters. Other default configurations may be added later. 

            :param bvh_file_path: 
            :param type="flat": 
            :param to_meters=1: 
            :param forward_dir = [0,0,1]:
            :param shoulder_joints = [10, 20] (left, right):
            :param hip_joints = [2, 27] (left, right):
            :param fid_l = [4,5] (heel, toe):
            :param fid_r = [29, 30] (heel, toe):

        Example process_data:
            def process_data(handler):
                bvh_path = handler.bvh_file_path
                phase_path = bvh_path.replace('.bvh', '.phase')
                gait_path = bvh_path.replace(".bvh", ".gait")

                Pc, Xc, Yc = [], [], []


                gait = handler.load_gait(gait_path, adjust_crouch=True)
                phase, dphase = handler.load_phase(phase_path)

                local_positions = handler.get_root_local_joint_positions()
                local_velocities = handler.get_root_local_joint_velocities()

                root_velocity = handler.get_root_velocity()
                root_rvelocity = handler.get_rotational_velocity()

                feet_l, feet_r = handler.get_foot_concats()

                for i in range(handler.window, handler.n_frames - handler.window - 1, 1):
                    rootposs,rootdirs = handler.get_trajectory(i)
                    rootgait = gait[i - handler.window:i+handler.window:10]

                    Pc.append(phase[i])

                    Xc.append(np.hstack([
                            rootposs[:,0].ravel(), rootposs[:,2].ravel(), # Trajectory Pos
                            rootdirs[:,0].ravel(), rootdirs[:,2].ravel(), # Trajectory Dir
                            rootgait[:,0].ravel(), rootgait[:,1].ravel(), # Trajectory Gait
                            rootgait[:,2].ravel(), rootgait[:,3].ravel(), 
                            rootgait[:,4].ravel(), rootgait[:,5].ravel(), 
                            local_positions[i-1].ravel(),  # Joint Pos
                            local_velocities[i-1].ravel(), # Joint Vel
                        ]))

                    rootposs_next, rootdirs_next = handler.get_trajectory(i + 1, i + 1)

                    Yc.append(np.hstack([
                        root_velocity[i,0,0].ravel(), # Root Vel X
                        root_velocity[i,0,2].ravel(), # Root Vel Y
                        root_rvelocity[i].ravel(),    # Root Rot Vel
                        dphase[i],                    # Change in Phase
                        np.concatenate([feet_l[i], feet_r[i]], axis=-1), # Contacts
                        rootposs_next[:,0].ravel(), rootposs_next[:,2].ravel(), # Next Trajectory Pos
                        rootdirs_next[:,0].ravel(), rootdirs_next[:,2].ravel(), # Next Trajectory Dir
                        local_positions[i].ravel(),  # Joint Pos
                        local_velocities[i].ravel(), # Joint Vel
                        ]))

                return np.array(Pc), np.array(Xc), np.array(Yc)

        """   

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

        self.reference_skeleton = []

    def reset_computations(self):
        """
        Resets computation buffers (__forwards, __root_rotations, __local_positions, __local_velocities). Usefull, if global_rotations are changed. 
        """   
        self.__forwards = []
        self.__root_rotations = []
        self.__local_positions, self.__local_velocities = [],[]
       
    def copy(self):
        """
        Produces a copy of the current handler. 

        :return Preprocessing_handler:
        """
        tmp = FeatureExtractor(self.bvh_file_path, self.type, self.to_meters, self.__ref_dir, self.shoulder_joints, self.hip_joints, self.foot_left, self.foot_right)
        tmp.__global_positions = np.array(self.__global_positions)
        return tmp
        
    def set_holden_parameters(self):
        """
        Set parameters for holden-skeleton
        """
        self.shoulder_joints = [18, 25]
        self.hip_joints = [2, 7]
        self.foot_left = [4,5]
        self.foot_right = [9, 10]
        self.to_meters = 1#5.6444
        self.head = 16 # check this!

    def set_makehuman_parameters(self):
        """
        Set parameters for makehuman skeleton
        """
        self.shoulder_joints = [10, 20]
        self.hip_joints = [2, 27]
        self.foot_left = [4, 5]
        self.foot_right = [29, 30]
        self.to_meters = 1
        self.head = 16 # check this!

        

    def load_motion(self, scale = 10, frame_rate_divisor = 2):
        """
        loads the bvh-file, sets the global_coordinates, n_joints and n_frames. Has to be called before any of the other functions are used. 

            :param scale=10: spatial scale of skeleton. 
            :param frame_rate_divisor=2: frame-rate divisor (e.g. reducing framerat from 120 -> 60 fps)
        """   
        print('Processing Clip %s' % self.bvh_file_path)
        
        bvhreader = BVHReader(self.bvh_file_path)
        skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
        zero_rotations = np.zeros(bvhreader.frames.shape[1])
        zero_posture = convert_euler_frames_to_cartesian_frames(skeleton, np.array([zero_rotations]))[0]
        zero_posture[:,0] *= -1
        
        def rotation_to_target(vecA, vecB):
            vecA = vecA / np.linalg.norm(vecA)
            vecB = vecB / np.linalg.norm(vecB)
            dt = np.dot(vecA, vecB)
            cross = np.linalg.norm(np.cross(vecA, vecB))
            G = np.array([[dt, -cross, 0],[cross, dt, 0], [0,0,1]])

            v = (vecB - dt * vecA)
            v = v / np.linalg.norm(v)
            w = np.cross(vecB, vecA)
            #F = np.array([[vecA[0], vecA[1], vecA[2]], [v[0], v[1], v[2]], [w[0], w[1], w[2]]])
            F = np.array([vecA, v, w])

            #U = np.matmul(np.linalg.inv(F), np.matmul(G, F))
            U = np.matmul(np.matmul(np.linalg.inv(F), G), F)
            # U = np.zeros((4,4))
            # U[3,3] = 1
            # U[:3,:3] = b

            test = np.matmul(U, vecA)
            if np.linalg.norm(test - vecB) > 0.0001:
                print("error: ", test, vecB)
            
            #b = np.matmul(np.linalg.inv(F), np.matmul(G, F))
            b = np.matmul(np.matmul(np.linalg.inv(F), G), F)
            U = np.zeros((4,4))
            U[3,3] = 1
            U[:3,:3] = b
            q = quaternion_from_matrix(U)
            #q[3] = -q[3]
            return q

        self.reference_skeleton = []
        mapping = {}
        for b in skeleton.animated_joints:
            node_desc = skeleton._get_node_desc(b)
            self.reference_skeleton.append({"name" : b})
            mapping[b] = int(node_desc["index"])
            self.reference_skeleton[-1]["parent"] = "" if node_desc["parent"] is None else node_desc["parent"]
            children = []
            for c in node_desc["children"]:
                if "EndSite" in c["name"]:
                    continue
                else:
                    children.append(c["name"])
            self.reference_skeleton[-1]["children"] = children
            self.reference_skeleton[-1]["index"] = node_desc["index"]
            self.reference_skeleton[-1]["position"] = zero_posture[int(node_desc["index"])].tolist()
            child_id = 0

            forward = np.array([0.0, 1.0, 0.0])
            
            target_pos = np.array(zero_posture[int(node_desc["children"][child_id]["index"])])
            my_pos = np.array(self.reference_skeleton[-1]["position"] )
            target_dir = (target_pos - my_pos)
            
            if np.linalg.norm(target_dir) < 0.0001:
                rotation = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                rotation = rotation_to_target(forward, target_dir)# - (parent_dir))
            self.reference_skeleton[-1]["rotation"] = rotation.tolist()

            # local rotation:
            if node_desc["parent"] is not None:
                parent_rot = np.array(self.reference_skeleton[mapping[node_desc["parent"]]]["rotation"])
            else:
                parent_rot = np.array([1.0,0.0,0.0,0.0])
            #inv_parent = quaternion_inverse(parent_rot)
            #loc_rot = quaternion_multiply(inv_parent, rotation)
            inv_parent = np.linalg.inv(quaternion_matrix(parent_rot))
            loc_rot = quaternion_from_matrix(np.matmul(quaternion_matrix(rotation), inv_parent))



            self.reference_skeleton[-1]["local_rotation"] = (loc_rot).tolist()

            # local position: 
            loc_pos = np.array([0.0,0.0,0.0])
            if node_desc["parent"] is not None:
                loc_pos[1] = np.linalg.norm(my_pos - zero_posture[mapping[node_desc["parent"]]])
            self.reference_skeleton[-1]["local_position"] = loc_pos.tolist()

            lr = self.reference_skeleton[-1]["local_rotation"]
            print(b, "\n\tpos: ", self.reference_skeleton[-1]["local_position"], 
                "\n\tloc rot: ", lr[1], lr[2], lr[3], lr[0],
                "\n\tglob rot: ", self.reference_skeleton[-1]["rotation"])

        cartesian_frames = convert_euler_frames_to_cartesian_frames(skeleton, bvhreader.frames)
        global_positions = cartesian_frames * scale

        self.__global_positions = global_positions[::frame_rate_divisor]
        self.n_frames, self.n_joints, _ = self.__global_positions.shape

    def load_gait (self, gait_file, frame_rate_divisor = 2, adjust_crouch = False):
        """
        Loads gait information from a holden-style gait-file. 

            :param gait_file: 
            :param frame_rate_divisor=2: 
            :param adjust_crouch=False: 

            :return gait-vector (np.array(n_frames, 8))
        """   
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
        # Todo: adjust dynamically to file information

        global_positions = np.array(self.__global_positions)

        if adjust_crouch:
            crouch_low, crouch_high = 80, 130
            head = self.head
            gait[:-1,3] = 1 - np.clip((global_positions[:-1,head,1] - 80) / (130 - 80), 0, 1)
            gait[-1,3] = gait[-2,3]
        return gait


    def load_phase(self, phase_file, frame_rate_divisor = 2):
        """
        Load phase data from a holden-style phase file. 

            :param phase_file: 
            :param frame_rate_divisor=2: 

            :return phase (np.array(n_frames, 1)), dphase (np.array(n_frames - 1, 1))
        """   
        # phase_file = data.replace('.bvh', '.phase')
        phase = np.loadtxt(phase_file)[::frame_rate_divisor]
        dphase = phase[1:] - phase[:-1]
        dphase[dphase < 0] = (1.0-phase[:-1]+phase[1:])[dphase < 0]

        return phase, dphase


    def get_forward_directions(self):
        """
        Computes forward directions. Results are stored internally to reduce future computation time. 

            :return forward_dirs (np.array(n_frames, 3))

        """   
        sdr_l, sdr_r = self.shoulder_joints[0], self.shoulder_joints[1]
        hip_l, hip_r = self.hip_joints[0], self.hip_joints[1]
        global_positions = np.array(self.__global_positions)

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
        """
        Returns root rotations. Results are stored internally to reduce future computation time. 

            :return root_rotations (List(Quaternion), n_frames length)
        """   
        ref_dir = self.__ref_dir
        forward = self.get_forward_directions()

        if len(self.__root_rotations) == 0:
            forwards = self.get_forward_directions()
            self.__root_rotations = get_rotation_to_ref_direction(forward, ref_dir=ref_dir)
        return self.__root_rotations

    def __root_local_transform(self):
        """
        Helper function to compute and store local transformations. 
            
        """   
        if len(self.__local_positions) == 0:
            local_positions = np.array(self.__global_positions)
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
        """
        Computes and returns root_local joint positions in cartesian space. 
            
            :return joint positions (np.array(n_frames, n_joints, 3))
        """   
        lp, _ = self.__root_local_transform()
        return lp

    def get_root_local_joint_velocities(self):
        """
        Computes and returns root_local joint velocities in cartesian space. 
            
            :return joint velocities (np.array(n_frames, n_joints, 3))
        """   
        _, lv = self.__root_local_transform()
        return lv

    def get_root_velocity(self):
        """
        Returns root velocity in root local cartesian space. 
            
            : return np.array(n_frames, 1, 3)
        """   
        global_positions = np.array(self.__global_positions)
        root_rotations = self.get_root_rotations()
        root_velocity = (global_positions[1:, 0:1] - global_positions[:-1, 0:1]).copy()

        for i in range(self.n_frames - 1):
            root_velocity[i,0][1] = 0
            root_velocity[i,0] /= np.linalg.norm(root_velocity[i,0])
            root_velocity[i, 0] = root_rotations[i+1] * root_velocity[i, 0]
        return root_velocity

    def get_rotational_velocity(self):
        """
        Returns root rotational velocitie in root local space. 
            
            :return root_rvel (np.array(n_frames, 1, Quaternion))
        """   
        root_rvelocity = np.zeros(self.n_frames - 1)
        root_rotations = self.get_root_rotations()
        
        for i in range(self.n_frames - 1):
            q = root_rotations[i+1] * (-root_rotations[i])
            td = q * self.__ref_dir
            rvel = np.arctan2(td[0], td[2])
            root_rvelocity[i] = rvel #Quaternion.get_angle_from_quaternion(q, self.__ref_dir)

        return root_rvelocity

    def get_foot_concats(self, velfactor = np.array([0.05, 0.05])):
        """

        Performs a simple heuristical foot_step detection

            :param velfactor=np.array([0.05, 0.05])

            :return feet_l, feet_r  (np.array(n_frames, 1), dtype = np.float)
        """   
        fid_l, fid_r = self.foot_left, self.foot_right
        velfactor = velfactor / self.to_meters

        global_positions = np.array(self.__global_positions)

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
        Computes the trajectory string for the input frame (12 surrounding points with a distance of 10 frames each)

        :param start_from (int): -1 if whole window should be considered, value if specific start frame should be considered (e.g. i+1)

        :return rootposs, rootdirs (np.array(12, 3))
        """
        global_positions = np.array(self.__global_positions)
        forward = self.get_forward_directions()
        root_rotations = self.get_root_rotations()
        
        if start_from < 0:
            start_from = frame - self.window

        # Todo: expose frame-step
        rootposs = (global_positions[start_from:frame+self.window:10,0] - global_positions[frame:frame+1,0]) ### 12*3
        rootdirs = forward[start_from:frame+self.window:10]
        for j in range(len(rootposs)):
            rootposs[j] = root_rotations[frame] * rootposs[j]
            rootdirs[j] = root_rotations[frame] * rootdirs[j]

        return rootposs, rootdirs

    def terrain_fitting(self, foot_step_path, patches_path, Pc, Xc, Yc, xslice, yslice):
        """

        Performs terrain fitting algorithm. Footsteps are loaded from foot_step_path and iterated. Patches are loaded from patches_path. 
        The single steps are matched to the patches. 
        The best patches are considered and trajectory heights are sampled. 

        PC, Xc, Yc are the configuration as before, xslice denotes the slice of joint positions in Xc, yslice the slice of joint positions in Yc. 
        Joint positions are change to match each patch. 

        Joint heights are appended to Xc and results are returned.

            :param self: 
            :param foot_step_path: 
            :param patches_path: 
            :param Pc: 
            :param Xc: 
            :param Yc: 
            :param xslice = slice((window*2)//10)*10+1, ((window*2)//10)*10+njoints*3+1, 3):
            :param yslice = slice(8+(window//10)*4+1, 8+(window//10)*4+njoints*3+1, 3):

            :returns P, X, Y (adapted data):
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

            H, Hmean = self.__process_heights(slc, patches)

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

    def __process_heights(self, slice, patches, nsamples = 10):
        """
        Helper function to process a single step. 

            :param slice: 
            :param patches: 
            :param nsamples=10: 
        """   
        
        tmp_handler = self.copy()
        tmp_handler.__global_positions = np.array(self.__global_positions[slice])
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