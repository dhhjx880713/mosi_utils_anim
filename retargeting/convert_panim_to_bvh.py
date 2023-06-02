import copy
import numpy as np
import glob
import os
from mosi_utils_anim.utilities import load_json_file
from mosi_utils_anim.utilities.motion_plane import Plane
from mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder, BVHWriter
from directional_constraints_retargeting import align_ref_frame, retarget_motion
from skeleton_definition import GAME_ENGINE_JOINTS_DOFS


JOINTMAPPING= {
    'pelvis': 'pelvis',
    'spine_03': 'spine_03',
    'clavicle_l': 'clavicle_l',
    'upperarm_l': 'upperarm_l',
    'lowerarm_l': 'lowerarm_l',
    'hand_l': 'hand_l',
    'clavicle_r': 'clavicle_r',
    'upperarm_r': 'upperarm_r',
    'lowerarm_r': 'lowerarm_r',
    'hand_r': 'hand_r',
    'neck_01': 'neck_01',
    'head': 'head'
}


def convert_panim_to_euler_frames(panim_data, bvh_skeleton_file, skeleton_type='game_engine_skeleton'):
    '''

    :param panim_data:
    :param bvh_skeleton_file:
    :param skeleton_type:
    :return:
    '''
    target_bvhreader = BVHReader(bvh_skeleton_file)
    target_skeleton = SkeletonBuilder().load_from_bvh(target_bvhreader)
    motion_data = panim_data['motion_data']
    skeleton_data = panim_data['skeleton']
    if skeleton_type == 'game_engine_skeleton':
        body_plane_joints = ['thigh_r', 'Root', 'thigh_l']
        root_joint = 'pelvis'
        root_index = skeleton_data['pelvis']['index']
    elif skeleton_type == 'cmu_skeleton':
        body_plane_joints = ['RightUpLeg', 'Hips', 'LeftUpLeg']
        root_joint = 'Hips'
        root_index = skeleton_data['Hips']['index']
    elif skeleton_type == "Edin_skeleton":
        body_plane_joints = ['RightUpLeg', 'Hips', 'LeftUpLeg']
        root_joint = 'Hips'
        root_index = skeleton_data['Hips']['index']
    else:
        raise ValueError('unknown skeleton type')
    motion_data = np.asarray(motion_data)
    targets = create_direction_constraints_from_panim(skeleton_data, motion_data, body_plane_joints)
    n_frames = motion_data.shape[0]
    out_frames = []

    # root_index = skeleton_data['Hips']['index']
    for i in range(n_frames):
        pose_dir = targets[i]['pose_dir']
        print(i)
        if i == 0:
            new_frame = target_bvhreader.frames[0]
            ref_frame = align_ref_frame(new_frame, pose_dir, target_skeleton, body_plane_joints)
            ref_frame[0] = motion_data[0, root_index, 0]
            ref_frame[2] = motion_data[0, root_index, 2]
        else:
            # take the previous frame as initial guess
            new_frame = copy.deepcopy(out_frames[i-1])
            ref_frame = align_ref_frame(new_frame, pose_dir, target_skeleton, body_plane_joints)
            ref_frame[:3] = (motion_data[i, root_index] - motion_data[i-1, root_index]) + out_frames[i-1][:3]
        retarget_motion(root_joint, targets[i], target_skeleton, ref_frame, GAME_ENGINE_JOINTS_DOFS)
        out_frames.append(ref_frame)

    return np.asarray(out_frames)


def convert_panim_to_bvh(panim_data, bvh_skeleton_file, save_filename, 
                         body_plane_joints=['thigh_r', 'Root', 'thigh_l'], root_joint='pelvis'):
    """convert motion from panim format to bvh format 
    
    Arguments:
        panim_data {json} -- json data contrains skeleton definition and point cloud data
        bvh_skeleton_file {str} -- path to skeleton bvh file
        save_filename {[str} -- saving path
    """
    target_bvhreader = BVHReader(bvh_skeleton_file)
    target_skeleton = SkeletonBuilder().load_from_bvh(target_bvhreader)
    motion_data = np.asarray(panim_data['motion_data'])
    print(motion_data.shape)
    skeleton_data = panim_data['skeleton']
    # body_plane_joints = ['thigh_r', 'Root', 'thigh_l']
    # body_plane_joints = ['RightUpLeg', 'Hips', 'LeftUpLeg']
    motion_data = np.asarray(motion_data)
    # print(skeleton_data)
    targets = create_direction_constraints_from_panim(skeleton_data, motion_data, body_plane_joints)

    n_frames = motion_data.shape[0]
    out_frames = []
    # root_joint = 'pelvis'
    # root_joint = 'Hips'
    # root_index = skeleton_data['pelvis']['index']
    root_index = skeleton_data[root_joint]['index']
    for i in range(n_frames):
        pose_dir = targets[i]['pose_dir']
        # print(i)
        if i == 0:
            new_frame = target_bvhreader.frames[0]
            ref_frame = align_ref_frame(new_frame, pose_dir, target_skeleton, body_plane_joints)
            ref_frame[0] = motion_data[0, root_index, 0]
            ref_frame[2] = motion_data[0, root_index, 2]
        else:
            # take the previous frame as initial guess
            new_frame = copy.deepcopy(out_frames[i-1])
            ref_frame = align_ref_frame(new_frame, pose_dir, target_skeleton, body_plane_joints)
            ref_frame[:3] = (motion_data[i, root_index] - motion_data[i-1, root_index]) + out_frames[i-1][:3]
        # retarget_motion(root_joint, targets[i], target_skeleton, ref_frame, JOINTS_DOFS)
        retarget_motion(root_joint, targets[i], target_skeleton, ref_frame)
        out_frames.append(ref_frame)
    BVHWriter(save_filename, target_skeleton, out_frames, target_bvhreader.frame_time, is_quaternion=False)


def save_motion_data_to_bvh(motion_data, skeleton_data, bvh_skeleton_file, save_filename):
    '''
    
    Args:
        motion_data:
        skeleton_data:
        bvh_skeleton_file:
        save_filename:

    Returns:

    '''
    body_plane_joints = ['thigh_r', 'Root', 'thigh_l']
    target_bvhreader = BVHReader(bvh_skeleton_file)
    target_skeleton = SkeletonBuilder().load_from_bvh(target_bvhreader)
    targets = create_direction_constraints_from_panim(skeleton_data, motion_data, body_plane_joints)
    n_frames = motion_data.shape[0]
    out_frames = []
    root_joint = 'Hips'
    # root_index = skeleton_data['pelvis']['index']
    root_index = skeleton_data['Hips']['index']
    for i in range(n_frames):
        pose_dir = targets[i]['pose_dir']
        print(i)
        if i == 0:
            new_frame = target_bvhreader.frames[0]
            ## align target pose frame to pose_dir
            ref_frame = align_ref_frame(new_frame, pose_dir, target_skeleton, body_plane_joints)
            ## set reference pose position
            ref_frame[0] = motion_data[0, root_index, 0]
            ref_frame[2] = motion_data[0, root_index, 2]
        else:
            # take the previous frame as initial guess
            new_frame = copy.deepcopy(out_frames[i-1])
            ref_frame = align_ref_frame(new_frame, pose_dir, target_skeleton, body_plane_joints)
            ref_frame[:3] = (motion_data[i, root_index] - motion_data[i-1, root_index]) + out_frames[i-1][:3]
        retarget_motion(root_joint, targets[i], target_skeleton, ref_frame, GAME_ENGINE_JOINTS_DOFS)
        out_frames.append(ref_frame)
        out_frames = np.array(out_frames)
    BVHWriter(save_filename, target_skeleton, out_frames, target_bvhreader.frame_time, is_quaternion=False)


def create_direction_constraints_from_panim(skeleton_data, motion_data, body_plane_list=None):
    '''
    panim data is a costomized data format
    :param skeleton_data: OrderDict
    :param motion_data: n_frames * n_joints * 3
    :param body_plane_list : list<str>
    :return:
    '''
    ## find directional vector from parent joint to child joint

    n_frames = len(motion_data)
    targets = []
    for i in range(n_frames):
        frame_targets = {}  ## generate direcional constraints for one frame
        if isinstance(skeleton_data, dict):
            if body_plane_list is None:
                frame_targets['pose_dir'] = None
            else:
                points = []
                for joint in body_plane_list:
                    joint_index = skeleton_data[joint]['index']
                    points.append(motion_data[i, joint_index])
                body_plane = Plane(points)
                normal_vec = body_plane.normal_vector
                dir_vec = np.array([normal_vec[0], normal_vec[2]])
                frame_targets['pose_dir'] = dir_vec/np.linalg.norm(dir_vec)
            for joint, value in skeleton_data.items():  ## pairing parent and child
                if value['parent'] is not None:
                    parent_index = skeleton_data[value['parent']]['index']
                    joint_dir = motion_data[i, skeleton_data[joint]['index'], :] - motion_data[i, parent_index, :]
                    assert np.linalg.norm(joint_dir) != 0  ## avoid 0 zero directional constraint, if there are zero-length bone, skip the child bone, link to grandchild bone
                    if value['parent'] in frame_targets.keys():
                        frame_targets[value['parent']][joint] = joint_dir / np.linalg.norm(joint_dir)
                    else:
                        frame_targets[value['parent']] = {joint: joint_dir / np.linalg.norm(joint_dir)}
        elif isinstance(skeleton_data, list):
            if body_plane_list is None:
                frame_targets['pose_dir'] = None
            else:
                points = []
                for joint in body_plane_list:
                    joint_index = next((item["index"] for item in skeleton_data if item["name"] == joint), None)
                    assert joint_index is not None, ("skeleton mismatch!")
                    points.append(motion_data[i, joint_index])
                body_plane = Plane(points)
                normal_vec = body_plane.normal_vector
                dir_vec = np.array([normal_vec[0], normal_vec[2]])
                frame_targets['pose_dir'] = dir_vec/np.linalg.norm(dir_vec)          
            for joint in skeleton_data:
                if joint['parent'] is not None:
                    parent_index = next(item["index"] for item in skeleton_data if item["name"] == joint['parent'])
                    joint_dir = motion_data[i, joint["index"], :] - motion_data[i, parent_index, :]
                    assert np.linalg.norm(joint_dir) != 0 ## avoid 0 zero directional constraint, if there are zero-length bone, skip the child bone, link to grandchild bone
                    if joint['parent'] in frame_targets.keys():
                        frame_targets[joint['parent']][joint['name']] = joint_dir / np.linalg.norm(joint_dir)
                    else:
                        frame_targets[joint['parent']] = {joint['name']: joint_dir / np.linalg.norm(joint_dir)}
        else:
            raise KeyError("Unknown data type!")
        targets.append(frame_targets)
    return targets


def convert_style_transfer_data_to_bvh():
    data_dir = r'E:\workspace\mocap_data\original_processed\tmp'
    bvh_skeleton_file = r'E:\workspace\experiment data\cutted_holden_data_walking\tmp\ref.bvh'
    save_dir = r'E:\workspace\mocap_data\original_processed\sexy'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    panim_files = glob.glob(os.path.join(data_dir, '*.panim'))
    for panim_file in panim_files:
        if 'sexy' in panim_file:
            filename = os.path.split(panim_file)[-1]
            print(filename)
            panim_data = load_json_file(panim_file)
            output_filename = os.path.join(save_dir, filename.replace('panim', 'bvh'))
            convert_panim_to_bvh(panim_data, bvh_skeleton_file, output_filename)

