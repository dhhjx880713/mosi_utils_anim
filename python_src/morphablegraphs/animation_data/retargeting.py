import numpy as np
from copy import deepcopy
from math import degrees
import os
from . import Skeleton, BVHReader, MotionVector,  SkeletonEndSiteNode
from ..external.transformations import quaternion_from_matrix, euler_matrix, quaternion_matrix, quaternion_multiply, euler_from_quaternion, quaternion_from_euler, quaternion_inverse, euler_from_matrix
from ..utilities import load_json_file, write_to_json_file
from scipy.optimize import minimize

ROCKETBOX_TO_GAME_ENGINE_MAP = dict()
ROCKETBOX_TO_GAME_ENGINE_MAP["Hips"] = "pelvis"
#ROCKETBOX_TO_GAME_ENGINE_MAP["Spine"] = "spine_01"
#ROCKETBOX_TO_GAME_ENGINE_MAP["Spine_1"] = "spine_02"
#ROCKETBOX_TO_GAME_ENGINE_MAP["Neck"] = "neck_01"
#ROCKETBOX_TO_GAME_ENGINE_MAP["Head"] = "head"
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

ROOT_JOINT = "Hips"
ROOT_CHILDREN = ["RightUpLeg", "LeftUpLeg", "Spine"]
EXTREMITIES = ["RightUpLeg", "LeftUpLeg", "RightLeg", "LeftLeg", "RightArm", "LeftArm", "RightForeArm", "LeftForeArm"]
GAME_ENGINE_ROOT_JOINT = ROCKETBOX_TO_GAME_ENGINE_MAP[ROOT_JOINT]
GAME_ENGINE_ROOT_CHILDREN = ["pelvis", "clavicle_l", "clavicle_r"]#[ROCKETBOX_TO_GAME_ENGINE_MAP[k] for k in ROOT_CHILDREN]
GAME_ENGINE_EXTREMITIES = [ROCKETBOX_TO_GAME_ENGINE_MAP[k] for k in EXTREMITIES]


def normalize(v):
    return v/np.linalg.norm(v)


def filter_dofs(q, fixed_dims):
    e = list(euler_from_quaternion(q))
    for d in fixed_dims:
        e[d] = 0
    q = quaternion_from_euler(*e)
    return q


def ik_dir_objective(q, new_skeleton, free_joint_name, child_name, target_bone_dir, frame, offset):
    """ get distance based on fk methods of skeleton
    """
    frame[offset: offset + 4] = q
    bone_dir = get_dir_to_child(new_skeleton, free_joint_name, child_name, frame)
    delta = bone_dir - target_bone_dir
    e = np.linalg.norm(delta)
    return e


def find_rotation_using_optimization(new_skeleton, free_joint_name, child_name, target_bone_dir, frame, offset, guess=None):
    if guess is None:
        guess = [1, 0, 0, 0]
    args = new_skeleton, free_joint_name, child_name, target_bone_dir, frame, offset
    r = minimize(ik_dir_objective, guess, args)
    return r.x


def get_dir_to_child(skeleton, name, child_name, frame, use_cache=False):
    child_pos = skeleton.nodes[child_name].get_global_position(frame, use_cache)
    global_target_dir = child_pos - skeleton.nodes[name].get_global_position(frame, use_cache)
    global_target_dir /= np.linalg.norm(global_target_dir)
    return global_target_dir


def get_2d_root_rotation(target_skeleton, src_skeleton, src_frame, target_frame, src_root="Hips", target_root="pelvis"):
    global_src = src_skeleton.nodes[src_root].get_global_matrix(src_frame)
    global_src[:3, 3] = [0, 0, 0]

    global_target = target_skeleton.nodes[target_root].get_global_matrix(target_frame)
    global_target[:3, 3] = [0, 0, 0]

    ref_offset = [0, 0, 1, 1]
    rotated_point = np.dot(global_src, ref_offset)
    src_dir = np.array([rotated_point[0], rotated_point[2]])
    src_dir /= np.linalg.norm(src_dir)
    src_dir = [src_dir[0], src_dir[1]]

    rotated_point = np.dot(global_target, ref_offset)
    target_dir = np.array([rotated_point[0], rotated_point[2]])
    target_dir /= np.linalg.norm(src_dir)
    target_dir = [target_dir[0], target_dir[1]]

    angle = np.arccos(np.dot(src_dir, target_dir))
    angle = np.degrees(angle)
    #print "root quaternion", angle, src_dir, target_dir
    return quaternion_from_euler(*np.radians([0, angle, 0]))
