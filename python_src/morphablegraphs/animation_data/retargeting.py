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


def find_rotation_between_vectors(a,b, axis=None):
    """http://math.stackexchange.com/questions/293116/rotating-one-3d-vector-to-another"""
    if axis is None:
        axis = normalize(np.cross(a, b))
    magnitude = np.linalg.norm(a) * np.linalg.norm(b)
    angle = math.acos(np.dot(a,b)/magnitude)
    return quaternion_from_axis_angle(axis, angle)


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

