"""
Functions for retargeting based on the paper "Using an Intermediate Skeleton and Inverse Kinematics for Motion Retargeting"
by Monzani et al.
See: http://www.vis.uni-stuttgart.de/plain/vdl/vdl_upload/91_35_retargeting%20monzani00using.pdf
"""
import numpy as np
from copy import deepcopy
from math import degrees
import os
from . import Skeleton, BVHReader, MotionVector,  SkeletonEndSiteNode
from ..external.transformations import quaternion_from_matrix, euler_matrix, quaternion_matrix, quaternion_multiply, euler_from_quaternion, quaternion_from_euler, quaternion_inverse, euler_from_matrix
from ..utilities import load_json_file, write_to_json_file
from .motion_editing import quaternion_from_vector_to_vector
from scipy.optimize import minimize
import collections
import math
import time
GAME_ENGINE_REFERENCE_POSE_EULER = [0.202552772072, -0.3393745422363281, 10.18097736938018, 0.0, 0.0, 88.15288821532792, -3.3291626376861925, 172.40743933061506, 90.48198857145417, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.267224765943357, 5.461144951918523, -108.06852912064531, -15.717336936646204, 0.749500429122681, -31.810810127019366, 5.749795741186075, -0.64655017163842, -43.79621907038145, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 26.628020277759394, 18.180233818114445, 89.72419760530946, 18.24367060730651, 1.5799727651772104, 39.37862756278345, 45.669771502815834, 0.494263941559835, 19.71385918379141, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 31.738570778606658, -0.035796158863762605, -10.010293103153826, 0.0, 0.0, 0.0, 520.8293416407181, -6.305803932868075, -1.875562438841992, 23.726055821805346, 0.0010593260744296063, 3.267962297354599, -60.93853290197474, 0.0020840827755293063, -2.8705207369072694, 0.0, 0.0, 0.0, -158.31965133452601, -12.378967235699056, 6.357392524527775, 19.81125436520809, -0.03971871449276927, -11.895292807406602, -70.75282007667651, -1.2148004469780682, 20.150610072602195, 0.0, 0.0, 0.0]
ROCKETBOX_TO_GAME_ENGINE_MAP = dict()
ROCKETBOX_TO_GAME_ENGINE_MAP["Hips"] = "pelvis"
ROCKETBOX_TO_GAME_ENGINE_MAP["Spine"] = "head"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftShoulder"] = "clavicle_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightShoulder"] = "clavicle_r"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftArm"] = "upperarm_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightArm"] = "upperarm_r"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftForeArm"] = "lowerarm_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightForeArm"] = "lowerarm_r"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftHand"] = "hand_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightHand"] = "hand_r"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftUpLeg"] = "thigh_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightUpLeg"] = "thigh_r"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftLeg"] = "calf_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightLeg"] = "calf_r"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftFoot"] = "foot_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightFoot"] = "foot_r"
ROCKETBOX_TO_GAME_ENGINE_MAP["Bip01_L_Toe0"] = "ball_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["Bip01_R_Toe0"] = "ball_r"
#ROCKETBOX_TO_GAME_ENGINE_MAP["Neck"] = "neck_01"
#ROCKETBOX_TO_GAME_ENGINE_MAP["Head"] = "head"
GAME_ENGINE_TO_ROCKETBOX_MAP = {v:k for k,v in ROCKETBOX_TO_GAME_ENGINE_MAP.items()}
ADDITIONAL_ROTATION_MAP = dict()
ADDITIONAL_ROTATION_MAP["LeftShoulder"] = [0, 0, -20]
ADDITIONAL_ROTATION_MAP["LeftArm"] = [0, 0, 20]
ADDITIONAL_ROTATION_MAP["RightShoulder"] = [0, 0, 20]
ADDITIONAL_ROTATION_MAP["RightArm"] = [0, 0, -20]

OPENGL_UP_AXIS = np.array([0, 1, 0])
ROCKETBOX_ROOT_OFFSET = np.array([0, 100.949997, 0])
EXTRA_ROOT_NAME = "Root"
ROOT_JOINT = "Hips"
ROOT_CHILDREN = ["Spine", "LeftUpLeg","RightUpLeg"]
EXTREMITIES = ["RightUpLeg", "LeftUpLeg", "RightLeg", "LeftLeg", "RightArm", "LeftArm", "RightForeArm", "LeftForeArm"]
GAME_ENGINE_ROOT_JOINT = ROCKETBOX_TO_GAME_ENGINE_MAP[ROOT_JOINT]
GAME_ENGINE_ROOT_CHILDREN = ["spine_01", "clavicle_l", "clavicle_r"]#[ROCKETBOX_TO_GAME_ENGINE_MAP[k] for k in ROOT_CHILDREN]
GAME_ENGINE_EXTREMITIES = [ROCKETBOX_TO_GAME_ENGINE_MAP[k] for k in EXTREMITIES]



def normalize(v):
    return v/np.linalg.norm(v)


def filter_dofs(q, fixed_dims):
    e = list(euler_from_quaternion(q))
    for d in fixed_dims:
        e[d] = 0
    q = quaternion_from_euler(*e)
    return q


def apply_additional_rotation_on_frames(animated_joints, frames, additional_rotation_map):
    new_frames = []
    for frame in frames:
        new_frame = frame[:]
        for idx, name in enumerate(animated_joints):
            if name in additional_rotation_map:
                euler = np.radians(additional_rotation_map[name])
                additional_q = quaternion_from_euler(*euler)
                offset = idx *4+3
                q = new_frame[offset:offset + 4]
                new_frame[offset:offset + 4] = quaternion_multiply(q, additional_q)

        new_frames.append(new_frame)
    return new_frames


def get_dir_to_child(skeleton, name, child_name, frame, use_cache=False):

    child_pos = skeleton.nodes[child_name].get_global_position(frame, use_cache)
    global_target_dir = child_pos - skeleton.nodes[name].get_global_position(frame, True)
    global_target_dir /= np.linalg.norm(global_target_dir)
    return global_target_dir


def ik_dir_objective(q, new_skeleton, free_joint_name, targets, frame, offset):
    """ Get distance to multiple target directions similar to the Blender implementation based on FK methods
        of the Skeleton class
    """

    error = 0.0
    frame[offset: offset + 4] = q
    for target in targets:
        bone_dir = get_dir_to_child(new_skeleton, free_joint_name, target["dir_name"], frame)
        delta = bone_dir - target["dir_to_child"]
        error += np.linalg.norm(delta)
    return error


def find_rotation_using_optimization(new_skeleton, free_joint_name, targets, frame, offset, guess=None):
    if guess is None:
        guess = [1, 0, 0, 0]
    args = new_skeleton, free_joint_name, targets, frame, offset
    r = minimize(ik_dir_objective, guess, args)
    q = normalize(r.x)
    return q


def ik_dir_objective2(q, skeleton, parent_transform, targets, free_joint_offset):
    """ get distance based on precomputed parent matrices
    """
    local_m = quaternion_matrix(q)
    local_m[:3, 3] = free_joint_offset

    global_m = np.dot(parent_transform, local_m)
    free_joint_pos = global_m[:3,3]

    error = 0
    for target in targets:
        local_target_m = np.eye(4)
        local_target_m[:3, 3] = skeleton.nodes[target["dir_name"]].offset
        bone_dir = np.dot(global_m, local_target_m)[:3, 3] - free_joint_pos
        bone_dir = normalize(bone_dir)
        delta = target["dir_to_child"] - bone_dir
        error += np.linalg.norm(delta)#  np.dot(delta, delta) leads to instability. maybe try to normalize.
    return error


def find_rotation_using_optimization2(new_skeleton, free_joint_name, targets, frame, offset, guess=None):
    if guess is None:
        guess = [1, 0, 0, 0]

    parent = new_skeleton.nodes[free_joint_name].parent
    if parent is not None:
        parent_name = parent.node_name
        parent_transform = new_skeleton.nodes[parent_name].get_global_matrix(frame)

    else:
        parent_transform = np.eye(4)

    args = new_skeleton, parent_transform, targets, new_skeleton.nodes[free_joint_name].offset
    r = minimize(ik_dir_objective2, guess, args)
    q = normalize(r.x)

    return q

def quaternion_from_axis_angle(axis, angle):
    q = [1,0,0,0]
    q[1] = axis[0] * math.sin(angle / 2)
    q[2] = axis[1] * math.sin(angle / 2)
    q[3] = axis[2] * math.sin(angle / 2)
    q[0] = math.cos(angle / 2)
    return q


def find_rotation_between_vectors(a,b, guess=None):
    """http://math.stackexchange.com/questions/293116/rotating-one-3d-vector-to-another"""
    #if guess is not None:
    #    a = np.dot(quaternion_matrix(guess)[:3,:3], a)
    axis = normalize(np.cross(a, b))
    magnitude = np.linalg.norm(a) * np.linalg.norm(b)
    angle = math.acos(np.dot(a,b)/magnitude)
    q = quaternion_from_axis_angle(axis, angle)
    #if guess is not None:
    #    q = quaternion_multiply(q,guess )
    return q


def find_rotation_analytically_old(new_skeleton, free_joint_name, target, frame):
    bone_dir = get_dir_to_child(new_skeleton, free_joint_name, target["dir_name"], frame)
    target_dir = normalize(target["dir_to_child"])
    q = quaternion_from_vector_to_vector(bone_dir, target_dir)
    return q

def find_rotation_analytically(new_skeleton, free_joint_name, target, frame):
    #find global rotation
    offset = new_skeleton.nodes[target["dir_name"]].offset
    target_dir = normalize(target["dir_to_child"])

    q = find_rotation_between_vectors(offset, target_dir)

    # bring into parent coordinate system
    pm = new_skeleton.nodes[target["dir_name"]].parent.get_global_matrix(frame)
    pm[:3,3] = [0, 0, 0]
    inv_pm = np.linalg.inv(pm)
    r = quaternion_matrix(q)
    lr = np.dot(inv_pm, r)
    q = quaternion_from_matrix(lr)
    return q

def to_local_cos(skeleton, node_name, frame, q):
    # bring into parent coordinate system
    pm = skeleton.nodes[node_name].get_global_matrix(frame)[:3,:3]
    #pm[:3, 3] = [0, 0, 0]
    inv_pm = np.linalg.inv(pm)
    r = quaternion_matrix(q)[:3,:3]
    lr = np.dot(inv_pm, r)[:3,:3]
    q = quaternion_from_matrix(lr)
    return q

def find_angle_x(v):
    # angle around y axis to rotate v to match 1,0,0
    if v[0] == 0:
        if v[1] > 0:
            return -0.5 * math.pi
        else:
            return 0.5 * math.pi
    if v[0] == 1:
        return 0
    if v[0] == -1:
        return math.pi

    alpha = math.acos(v[0])
    if v[1] > 0:
        alpha = - alpha
    return alpha

def find_rotation_analytically_from_axis(new_skeleton, free_joint_name, target, frame, offset):
    #find global orientation
    global_src_up_vec = normalize(target["global_src_up_vec"])
    global_src_x_vec = normalize(target["global_src_x_vec"])
    local_target_up_vec = [0,1,0]
    local_target_x_vec = [1,0,0]
    #find rotation between up vector of target skeleton and global src up axis
    qy = find_rotation_between_vectors(local_target_up_vec, global_src_up_vec)

    #find rotation around y axis after aligning y axis
    m = quaternion_matrix(qy)[:3,:3]
    local_target_x_vec = np.dot(m, local_target_x_vec)
    local_target_x_vec = normalize(local_target_x_vec)
    #q = q1
    qx = find_rotation_between_vectors(local_target_x_vec, global_src_x_vec)
    q = quaternion_multiply(qx,qy)
    q = to_local_cos(new_skeleton, free_joint_name, frame, q)

    #frame[offset:offset+4] = lq
    #axes = get_coordinate_system_axes(new_skeleton,free_joint_name, frame, AXES)
    #v = change_of_basis(global_src_x_vec, *axes)
    #normalize(v)
    #a = find_angle_x(v)
    #xq = quaternion_from_axis_angle(global_src_up_vec, a)
    #q = quaternion_multiply(q1,xq)
    #bring into local coordinate system of target skeleton
    #q = to_local_cos(new_skeleton, free_joint_name,frame, q)


    #m = quaternion_matrix(q1)[:3,:3]
    #x_target = np.dot(m, x_target)
    #q2 = to_local_cos(new_skeleton, free_joint_name,frame, q2)
    #
    return q


def create_local_cos_map(skeleton, up_vector, x_vector):
    joint_cos_map = dict()
    for j in skeleton.nodes.keys():
        joint_cos_map[j] = dict()
        joint_cos_map[j]["y"] = up_vector
        joint_cos_map[j]["x"] = x_vector

        if j == skeleton.root:
            joint_cos_map[j]["x"] = (-np.array(x_vector)).tolist()
        else:
            if len(skeleton.nodes[j].children) >0:
                node = skeleton.nodes[j].children[0]
                joint_cos_map[j]["y"] = node.offset
    return joint_cos_map

def find_rotation_analytically2(new_skeleton, free_joint_name, target, frame, joint_cos_map):
    #find global orientation
    global_src_up_vec = target["global_src_up_vec"]
    global_src_x_vec = target["global_src_x_vec"]
    #local_target_up_vec = [0,1,0] #TODO get from target skeleton
    #if free_joint_name == "pelvis":
    #    local_target_x_vec = [-1, 0, 0]  # TODO get from target skeleton
    #else:
    #    local_target_x_vec = [1,0,0]  # TODO get from target skeleton

    local_target_x_vec = joint_cos_map[free_joint_name]["x"]
    local_target_up_vec = joint_cos_map[free_joint_name]["y"]
    #find rotation between up vector of target skeleton and global src up axis
    qy = find_rotation_between_vectors(local_target_up_vec, global_src_up_vec)

    #apply the rotation on the  local_target_x_vec
    m = quaternion_matrix(qy)[:3,:3]
    aligned_target_x_vec = np.dot(m, local_target_x_vec)
    aligned_target_x_vec = normalize(aligned_target_x_vec)

    # find rotation around y axis as rotation between aligned_target_x_vec and global_src_x_vec
    qx = find_rotation_between_vectors(aligned_target_x_vec, global_src_x_vec)
    #print "r", aligned_target_x_vec, global_src_x_vec, qx
    if not np.isnan(qx).any():
        q = quaternion_multiply(qx, qy)
    else:
        q = qy
    q = to_local_cos(new_skeleton, free_joint_name, frame, q)
    return q

def get_targets_from_motion(src_skeleton, src_frames, src_to_target_joint_map, additional_rotation_map=None):
    if additional_rotation_map is not None:
        src_frames = apply_additional_rotation_on_frames(src_skeleton.animated_joints, src_frames, additional_rotation_map)

    targets = []
    for idx in range(0, len(src_frames)):
        frame_targets = dict()
        for src_name in src_skeleton.animated_joints:
            if src_name not in src_to_target_joint_map.keys():
                #print "skip1", src_name
                continue
            target_name = src_to_target_joint_map[src_name]
            frame_targets[target_name] = dict()
            frame_targets[target_name]["pos"] = src_skeleton.nodes[src_name].get_global_position(src_frames[idx])

            if len(src_skeleton.nodes[src_name].children) > -1:
                frame_targets[target_name]["targets"] = []
                for child_node in src_skeleton.nodes[src_name].children:
                    child_name = child_node.node_name

                    if child_name not in src_to_target_joint_map.keys():
                        #print "skip2", src_name
                        continue
                    target = {"dir_to_child": get_dir_to_child(src_skeleton, src_name, child_name,
                                                               src_frames[idx]),
                              "dir_name": src_to_target_joint_map[child_name],
                              "src_name": src_name

                              }
                    frame_targets[target_name]["targets"].append(target)
        targets.append(frame_targets)
    return targets


def find_orientation_of_extra_root(target_skeleton, new_frame, target_root, use_optimization=True):
    extra_root_target = {"dir_name": target_root, "dir_to_child": OPENGL_UP_AXIS}
    if use_optimization:
        q = find_rotation_using_optimization(target_skeleton, EXTRA_ROOT_NAME, [extra_root_target],
                                                          new_frame, 3)
    else:
        q = find_rotation_analytically(target_skeleton, EXTRA_ROOT_NAME, extra_root_target, new_frame)
    return q


def find_rotation_of_joint(target_skeleton, free_joint_name, targets, new_frame, offset, target_root, use_optimization=True):
    if free_joint_name == target_root:
        q = find_rotation_using_optimization(target_skeleton, free_joint_name,
                                             targets,
                                             new_frame, offset)
    else:
        if use_optimization:
            q = find_rotation_using_optimization2(target_skeleton, free_joint_name,
                                                  targets,
                                                  new_frame, offset)
        else:
            q = find_rotation_analytically(target_skeleton, free_joint_name, targets[0], new_frame)

            #if free_joint_name ==  target_root:
            #    q = quaternion_multiply(q, quaternion_from_euler(*np.radians([0,180,0])))
    return q


def get_new_frames_from_direction_constraints(target_skeleton,
                                              targets, frame_range=None,
                                              target_root=GAME_ENGINE_ROOT_JOINT,
                                              src_root_offset=ROCKETBOX_ROOT_OFFSET,
                                              extra_root=True, scale_factor=1.0,
                                              use_optimization=True):

    n_params = len(target_skeleton.animated_joints) * 4 + 3

    if frame_range is None:
        frame_range = (0, len(targets))

    if extra_root:
        animated_joints = target_skeleton.animated_joints[1:]
    else:
        animated_joints = target_skeleton.animated_joints

    new_frames = []
    for frame_idx, frame_targets in enumerate(targets[frame_range[0]:frame_range[1]]):
        start = time.clock()
        target_skeleton.clear_cached_global_matrices()

        new_frame = np.zeros(n_params)
        new_frame[:3] = np.array(frame_targets[target_root]["pos"]) * scale_factor

        if extra_root:
            new_frame[:3] -= src_root_offset*scale_factor - target_skeleton.nodes[EXTRA_ROOT_NAME].offset
            new_frame[3:7] = find_orientation_of_extra_root(target_skeleton, new_frame, target_root, use_optimization)
            offset = 7
        else:
            offset = 3

        for free_joint_name in animated_joints:
            q = [1, 0, 0, 0]
            if free_joint_name in frame_targets.keys() and len(frame_targets[free_joint_name]["targets"]) > 0:
                q = find_rotation_of_joint(target_skeleton, free_joint_name,
                                           frame_targets[free_joint_name]["targets"],
                                           new_frame, offset, target_root, use_optimization)
            new_frame[offset:offset + 4] = q
            offset += 4

        # apply_ik_constraints(target_skeleton, new_frame, constraints[frame_idx])#TODO
        duration = time.clock()-start
        print "processed frame", frame_range[0] + frame_idx, use_optimization, "in", duration, "seconds"
        new_frames.append(new_frame)
    return new_frames

def change_of_basis(v, x,y,z):
    m = np.array([x,y,z])
    m = np.linalg.inv(m)
    return np.dot(m, v)


def rotation_from_axes(target_axes, src_axes):
    target_bone_y = target_axes[1]
    local_target_bone_y = change_of_basis(target_bone_y, *src_axes)
    local_target_bone_y = normalize(local_target_bone_y)
    return quaternion_from_vector_to_vector([0, 1, 0], local_target_bone_y)

def get_bone_rotation_from_axes(src_skeleton, target_skeleton,
                                src_parent_name, target_parent_name,
                                src_frame, target_frame):
    src_axes = get_coordinate_system_axes(src_skeleton, src_parent_name, src_frame, AXES)
    target_axes = get_coordinate_system_axes(target_skeleton, target_parent_name, target_frame, AXES)
    return rotation_from_axes(target_axes, src_axes)


def get_bone_rotation_from_axes2(src_skeleton, target_skeleton,
                                    src_parent_name, target_parent_name,
                                    src_frame, target_frame):
    target_m = target_skeleton.nodes[target_parent_name].get_global_matrix(target_frame)[:3, :3]
    direction = normalize(target_skeleton.nodes[target_parent_name].offset)
    target_y = np.dot(target_m, direction)

    src_m = src_skeleton.nodes[src_parent_name].get_global_matrix(src_frame)[:3, :3]
    l_target_y = np.dot(np.linalg.inv(src_m), target_y)
    return quaternion_from_vector_to_vector( l_target_y, [0, 1, 0])

def rotate_bone_change_of_basis(src_skeleton,target_skeleton, src_parent_name, target_parent_name, src_frame,target_frame):
    src_global_axes = get_coordinate_system_axes(src_skeleton, src_parent_name, src_frame, AXES)

    # Bring global target y axis into local source coordinate system
    target_global_matrix = target_skeleton.nodes[target_parent_name].get_global_matrix(target_frame)
    global_target_y = np.dot(target_global_matrix[:3, :3], [0, 1, 0])
    local_target_y = change_of_basis(global_target_y, *src_global_axes)

    # find rotation difference between target axis and local y-axis
    q = quaternion_from_vector_to_vector(local_target_y, [0, 1, 0])
    return q


def rotate_bone2(src_skeleton,target_skeleton, src_name,target_name, src_to_target_joint_map, src_frame,target_frame, src_cos_map, target_cos_map):
    q = [1,0,0,0]
    src_child_name = src_skeleton.nodes[src_name].children[0].node_name
    rocketbox_x_axis = src_cos_map[src_name]["x"]#[0, 1, 0]
    rocketbox_up_axis = src_cos_map[src_name]["y"]#[1, 0, 0]
    if src_child_name in src_to_target_joint_map: # This prevents the spine from being rotated by 180 degrees. TODO Find out how to fix this without this condition.
        global_m = src_skeleton.nodes[src_name].get_global_matrix(src_frame)[:3, :3]
        global_src_up_vec = np.dot(global_m, rocketbox_up_axis)
        global_src_up_vec = normalize(global_src_up_vec)
        global_src_x_vec = np.dot(global_m, rocketbox_x_axis)
        global_src_x_vec = normalize(global_src_x_vec)
        target = {"global_src_up_vec": global_src_up_vec, "global_src_x_vec":global_src_x_vec}
        q = find_rotation_analytically2(target_skeleton, target_name, target, target_frame, target_cos_map)

    else:
        print "ignore", src_name, src_child_name
    return q


def retarget_from_src_to_target(src_skeleton, target_skeleton, src_frames, target_to_src_joint_map, additional_rotation_map=None, scale_factor=1.0,extra_root=False, src_root_offset=ROCKETBOX_ROOT_OFFSET):

    src_cos_map = create_local_cos_map(src_skeleton, [1,0,0], [0,1,0]) # TODO get up axes and cross vector from src skeleton
    #src_cos_map["LeftFoot"]["x"] = [0,1,0]
    #src_cos_map["LeftFoot"]["y"] = [0,0,1]#src_skeleton.nodes["LeftFoot"].children[0].offset
    #src_cos_map["RightFoot"]["x"] = [0,1,0]
    #src_cos_map["RightFoot"]["y"] = [0,0,1]#src_skeleton.nodes["RightFoot"].children[0].offset
    target_cos_map = create_local_cos_map(target_skeleton, [0,1,0], [1,0,0])# TODO get up axes and cross vector from target skeleton
    target_cos_map["Root"]["y"] = [1,0,0]
    target_cos_map["Root"]["x"] = [0,1,0]
    src_to_target_joint_map = {v:k for k, v in target_to_src_joint_map.items()}
    #if additional_rotation_map is not None:
    #    src_frames = apply_additional_rotation_on_frames(src_skeleton.animated_joints, src_frames, additional_rotation_map)
    n_params = len(target_skeleton.animated_joints) * 4 + 3
    target_frames = []
    print "n_params", n_params
    for idx, src_frame in enumerate(src_frames):

        target_frame = np.zeros(n_params)
        target_frame[:3] = src_frame[:3]*scale_factor
        if extra_root:
            target_frame[:3] -= src_root_offset * scale_factor + target_skeleton.nodes[EXTRA_ROOT_NAME].offset
            animated_joints = target_skeleton.animated_joints[1:]
            target_offset = 7
            #target = {"dir_name": "pelvis", "dir_to_child": [0,1,0]}
            #target_frame[3:7] = find_rotation_analytically_old(target_skeleton, "Root",
            #                                               target, target_frame)
            # target_frame[3:7] = quaternion_from_euler(*np.radians([0,0,90]))
            target = {"global_src_up_vec": [0,1,0], "global_src_x_vec": [1, 0, 0]}
            target_frame[3:7] = find_rotation_analytically2(target_skeleton, "Root",
                                                               target, target_frame, target_cos_map)
        else:
            animated_joints = target_skeleton.animated_joints
            target_offset = 3

        for target_name in animated_joints:
            q = [1, 0, 0, 0]
            if target_name in target_to_src_joint_map.keys():
                src_name = target_to_src_joint_map[target_name]
                q = rotate_bone2(src_skeleton,target_skeleton, src_name,target_name,
                                  src_to_target_joint_map, src_frame,target_frame,
                                  src_cos_map, target_cos_map)
            target_frame[target_offset:target_offset+4] = q
            target_offset += 4

        target_frames.append(target_frame)

    return target_frames






AXES = [[1,0,0], [0,1,0], [0,0,1]]

def get_coordinate_system_axes(skeleton, joint_name, frame, axes):
    global_m = skeleton.nodes[joint_name].get_global_matrix(frame)[:3,:3]
    #global_m[:3, 3] = [0, 0, 0]
    dirs = []
    for axis in axes:
        direction = np.dot(global_m, axis)
        direction /= np.linalg.norm(direction)
        dirs.append(direction)
    return np.array(dirs)
