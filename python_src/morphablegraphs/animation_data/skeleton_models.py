""" This module contains mappings from a standard set of human joints to different skeletons

"""
import numpy as np
import collections



ROCKETBOX_TOOL_BONES = [{
    "new_node_name": 'LeftToolEndSite',
    "parent_node_name": 'LeftHand',
    "new_node_offset": [6.1522069, -0.09354633, 3.33790343]
}, {
    "new_node_name": 'RightToolEndSite',
    "parent_node_name": 'RightHand',
    "new_node_offset": [6.1522069, 0.09354633, 3.33790343]
}, {
    "new_node_name": 'RightScrewDriverEndSite',
    "parent_node_name": 'RightHand',
    "new_node_offset": [22.1522069, -9.19354633, 3.33790343]
}, {
    "new_node_name": 'LeftScrewDriverEndSite',
    "parent_node_name": 'LeftHand',
    "new_node_offset": [22.1522069, 9.19354633, 3.33790343]
}
]
ROCKETBOX_FREE_JOINTS_MAP = {"LeftHand": ["Spine", "LeftArm", "LeftForeArm"],
                           "RightHand": ["Spine", "RightArm", "RightForeArm"],
                           "LeftToolEndSite": ["Spine", "LeftArm", "LeftForeArm"],
                           "RightToolEndSite": ["Spine", "RightArm", "RightForeArm"],  # , "RightHand"
                           "Head": [],
                           "RightScrewDriverEndSite": ["Spine", "RightArm", "RightForeArm"],
                           "LeftScrewDriverEndSite": ["Spine", "LeftArm", "LeftForeArm"]
                             }
ROCKETBOX_REDUCED_FREE_JOINTS_MAP = {"LeftHand": ["LeftArm", "LeftForeArm"],
                                   "RightHand": ["RightArm", "RightForeArm"],
                                   "LeftToolEndSite": ["LeftArm", "LeftForeArm"],
                                   "RightToolEndSite": ["RightArm", "RightForeArm"],
                                   "Head": [],
                                   "RightScrewDriverEndSite": ["RightArm", "RightForeArm"],
                                   "LeftScrewDriverEndSite": ["LeftArm", "LeftForeArm"]
                                     }
DEG2RAD = np.pi / 180
hand_bounds = [{"dim": 0, "min": 30 * DEG2RAD, "max": 180 * DEG2RAD},
               {"dim": 1, "min": -15 * DEG2RAD, "max": 120 * DEG2RAD},
               {"dim": 1, "min": -40 * DEG2RAD, "max": 40 * DEG2RAD}]

ROCKETBOX_ROOT_DIR = [0, 0, 1]
ROCKETBOX_BOUNDS = {"LeftArm": [],  # {"dim": 1, "min": 0, "max": 90}
                  "RightArm": []  # {"dim": 1, "min": 0, "max": 90},{"dim": 0, "min": 0, "max": 90}
                    , "RightHand": hand_bounds,  # [[-90, 90],[0, 0],[-90,90]]
                  "LeftHand": hand_bounds  # [[-90, 90],[0, 0],[-90,90]]
                    }

ROCKETBOX_ANIMATED_JOINT_LIST = ["Hips", "Spine", "Spine_1", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm",
                               "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftUpLeg",
                               "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"]


IK_CHAINS_RAW_SKELETON = dict()
IK_CHAINS_RAW_SKELETON["RightFoot"] = {"root": "RightUpLeg", "joint": "RightLeg", "joint_axis": [1, 0, 0], "end_effector_dir": [0,0,1]}
IK_CHAINS_RAW_SKELETON["LeftFoot"] = {"root": "LeftUpLeg", "joint": "LeftLeg", "joint_axis": [1, 0, 0], "end_effector_dir": [0,0,1]}

#IK_CHAINS_RAW_SKELETON["RightToeBase"] = {"root": "RightUpLeg", "joint": "RightLeg", "joint_axis": [1, 0, 0], "end_effector_dir": [0,0,1]}
#IK_CHAINS_RAW_SKELETON["LeftToeBase"] = {"root": "LeftUpLeg", "joint": "LeftLeg", "joint_axis": [1, 0, 0], "end_effector_dir": [0,0,1]}

IK_CHAINS_RAW_SKELETON["RightHand"] = {"root": "RightArm", "joint": "RightForeArm", "joint_axis": [0, 1, 0], "end_effector_dir": [1,0,0]}
IK_CHAINS_RAW_SKELETON["LeftHand"] = {"root": "LeftArm", "joint": "LeftForeArm", "joint_axis": [0, 1, 0], "end_effector_dir": [1,0,0]}


IK_CHAINS_ROCKETBOX_SKELETON = dict()
IK_CHAINS_ROCKETBOX_SKELETON["RightFoot"] = {"root": "RightUpLeg", "joint": "RightLeg", "joint_axis": [0, 1, 0], "end_effector_dir": [0,0,1]}
IK_CHAINS_ROCKETBOX_SKELETON["LeftFoot"] = {"root": "LeftUpLeg", "joint": "LeftLeg", "joint_axis": [0, 1, 0], "end_effector_dir": [0,0,1]}
IK_CHAINS_ROCKETBOX_SKELETON["RightHand"] = {"root": "RightArm", "joint": "RightForeArm", "joint_axis": [0, 1, 0], "end_effector_dir": [1,0,0]}
IK_CHAINS_ROCKETBOX_SKELETON["LeftHand"] = {"root": "LeftArm", "joint": "LeftForeArm", "joint_axis": [0, 1, 0], "end_effector_dir": [1,0,0]}


IK_CHAINS_GAME_ENGINE_SKELETON = dict()
IK_CHAINS_GAME_ENGINE_SKELETON["foot_l"] = {"root": "thigh_l", "joint": "calf_l", "joint_axis": [1, 0, 0], "end_effector_dir": [0,0,1]}
IK_CHAINS_GAME_ENGINE_SKELETON["foot_r"] = {"root": "thigh_r", "joint": "calf_r", "joint_axis": [1, 0, 0], "end_effector_dir": [0,0,1]}
IK_CHAINS_GAME_ENGINE_SKELETON["hand_r"] = {"root": "upperarm_r", "joint": "lowerarm_r", "joint_axis": [1, 0, 0], "end_effector_dir": [1,0,0]}
IK_CHAINS_GAME_ENGINE_SKELETON["hand_l"] = {"root": "upperarm_l", "joint": "lowerarm_l", "joint_axis": [1, 0, 0], "end_effector_dir": [1,0,0]}


RIGHT_SHOULDER = "RightShoulder"
RIGHT_ELBOW = "RightElbow"
RIGHT_WRIST = "RightHand"
ELBOW_AXIS = [0,1,0]

LOCOMOTION_ACTIONS = ["walk", "carryRight", "carryLeft", "carryBoth"]
DEFAULT_WINDOW_SIZE = 20
LEFT_FOOT = "LeftFoot"
RIGHT_FOOT = "RightFoot"
RIGHT_TOE = "RightToeBase"
LEFT_TOE = "LeftToeBase"
LEFT_HEEL = "LeftHeel"
RIGHT_HEEL = "RightHeel"

OFFSET = 0
RAW_SKELETON_FOOT_JOINTS = [RIGHT_TOE, LEFT_TOE, RIGHT_HEEL,LEFT_HEEL]
HEEL_OFFSET = [0, -6.480602, 0]

RAW_SKELETON_MODEL = collections.OrderedDict()
RAW_SKELETON_MODEL["root"] = "Hips"
RAW_SKELETON_MODEL["pelvis"] = "Hips"
RAW_SKELETON_MODEL["spine"] = "Spine"
RAW_SKELETON_MODEL["spine_1"] = "Spine_1"
RAW_SKELETON_MODEL["left_clavicle"] = None
RAW_SKELETON_MODEL["right_clavicle"] = None
RAW_SKELETON_MODEL["left_shoulder"] = "LeftShoulder"
RAW_SKELETON_MODEL["right_shoulder"] = "RightShoulder"
RAW_SKELETON_MODEL["left_elbow"] = "LeftForeArm"
RAW_SKELETON_MODEL["right_elbow"] = "RightForeArm"
RAW_SKELETON_MODEL["left_wrist"] = "LeftHand"
RAW_SKELETON_MODEL["right_wrist"] = "RightHand"
RAW_SKELETON_MODEL["left_hip"] = "LeftUpLeg"
RAW_SKELETON_MODEL["right_hip"] = "RightUpLeg"
RAW_SKELETON_MODEL["left_knee"] = "LeftLeg"
RAW_SKELETON_MODEL["right_knee"] = "RightLeg"
RAW_SKELETON_MODEL["left_ankle"] = "LeftFoot"
RAW_SKELETON_MODEL["right_ankle"] = "RightFoot"
RAW_SKELETON_MODEL["left_toe"] = "LeftToeBase"
RAW_SKELETON_MODEL["right_toe"] = "RightToeBase"
RAW_SKELETON_MODEL["left_heel"] = "LeftHeel"
RAW_SKELETON_MODEL["right_heel"] = "RightHeel"
RAW_SKELETON_MODEL["neck"] = "Neck"
RAW_SKELETON_MODEL["head"] = "Head"
RAW_SKELETON_MODEL["foot_joints"] = RAW_SKELETON_FOOT_JOINTS
RAW_SKELETON_MODEL["heel_offset"] = [0, -6.480602, 0]
RAW_SKELETON_MODEL["ik_chains"] = IK_CHAINS_RAW_SKELETON



GAME_ENGINE_SKELETON_MODEL = collections.OrderedDict()
GAME_ENGINE_SKELETON_MODEL["root"] = "Game_engine"
GAME_ENGINE_SKELETON_MODEL["pelvis"] = "pelvis"
GAME_ENGINE_SKELETON_MODEL["spine"] = "spine_01"
GAME_ENGINE_SKELETON_MODEL["spine_1"] = "spine_02"
GAME_ENGINE_SKELETON_MODEL["left_clavicle"] = "clavicle_l"
GAME_ENGINE_SKELETON_MODEL["right_clavicle"] = "clavicle_r"
GAME_ENGINE_SKELETON_MODEL["left_shoulder"] = "upperarm_l"
GAME_ENGINE_SKELETON_MODEL["right_shoulder"] = "upperarm_r"
GAME_ENGINE_SKELETON_MODEL["left_elbow"] = "lowerarm_l"
GAME_ENGINE_SKELETON_MODEL["right_elbow"] = "lowerarm_r"
GAME_ENGINE_SKELETON_MODEL["left_wrist"] = "hand_l"
GAME_ENGINE_SKELETON_MODEL["right_wrist"] = "hand_r"
GAME_ENGINE_SKELETON_MODEL["left_hip"] = "thigh_l"
GAME_ENGINE_SKELETON_MODEL["right_hip"] = "thigh_r"
GAME_ENGINE_SKELETON_MODEL["left_knee"] = "calf_l"
GAME_ENGINE_SKELETON_MODEL["right_knee"] = "calf_r"
GAME_ENGINE_SKELETON_MODEL["left_ankle"] = "foot_l"
GAME_ENGINE_SKELETON_MODEL["right_ankle"] = "foot_r"
GAME_ENGINE_SKELETON_MODEL["left_toe"] = "ball_l"
GAME_ENGINE_SKELETON_MODEL["right_toe"] = "ball_r"
GAME_ENGINE_SKELETON_MODEL["left_heel"] = "heel_l"
GAME_ENGINE_SKELETON_MODEL["right_heel"] = "heel_r"
GAME_ENGINE_SKELETON_MODEL["neck"] = "neck_01"
GAME_ENGINE_SKELETON_MODEL["head"] = "head"
GAME_ENGINE_SKELETON_MODEL["foot_joints"] = ["foot_l", "foot_r", "ball_r", "ball_l", "heel_r", "heel_l"]
GAME_ENGINE_SKELETON_MODEL["heel_offset"] = (np.array([0, 2.45, 3.480602]) * 1.75).tolist()
GAME_ENGINE_SKELETON_MODEL["ik_chains"] = IK_CHAINS_GAME_ENGINE_SKELETON

ROCKETBOX_SKELETON_MODEL = collections.OrderedDict()
ROCKETBOX_SKELETON_MODEL["root"] = "Hips"
ROCKETBOX_SKELETON_MODEL["pelvis"] = "Hips"
ROCKETBOX_SKELETON_MODEL["spine"] = "Spine"
ROCKETBOX_SKELETON_MODEL["spine_1"] = "Spine_1"
ROCKETBOX_SKELETON_MODEL["left_clavicle"] = "LeftShoulder"
ROCKETBOX_SKELETON_MODEL["right_clavicle"] = "RightShoulder"
ROCKETBOX_SKELETON_MODEL["left_shoulder"] = "LeftArm"
ROCKETBOX_SKELETON_MODEL["right_shoulder"] = "RightArm"
ROCKETBOX_SKELETON_MODEL["left_elbow"] = "LeftForeArm"
ROCKETBOX_SKELETON_MODEL["right_elbow"] = "lowerarm_r"
ROCKETBOX_SKELETON_MODEL["left_wrist"] = "LeftHand"
ROCKETBOX_SKELETON_MODEL["right_wrist"] = "RightHand"
ROCKETBOX_SKELETON_MODEL["left_hip"] = "LeftUpLeg"
ROCKETBOX_SKELETON_MODEL["right_hip"] = "RightUpLeg"
ROCKETBOX_SKELETON_MODEL["left_knee"] = "LeftLeg"
ROCKETBOX_SKELETON_MODEL["right_knee"] = "RightLeg"
ROCKETBOX_SKELETON_MODEL["left_ankle"] = "LeftFoot"
ROCKETBOX_SKELETON_MODEL["right_ankle"] = "RightFoot"
ROCKETBOX_SKELETON_MODEL["left_toe"] = "LeftToeBase"
ROCKETBOX_SKELETON_MODEL["right_toe"] = "RightToeBase"
ROCKETBOX_SKELETON_MODEL["left_heel"] = "LeftHeel"
ROCKETBOX_SKELETON_MODEL["right_heel"] = "RightHeel"
ROCKETBOX_SKELETON_MODEL["neck"] = "Neck"
ROCKETBOX_SKELETON_MODEL["head"] = "Head"
ROCKETBOX_SKELETON_MODEL["heel_offset"] = [0, -6.480602, 0]
ROCKETBOX_SKELETON_MODEL["foot_joints"] = RAW_SKELETON_FOOT_JOINTS
ROCKETBOX_SKELETON_MODEL["ik_chains"] = IK_CHAINS_RAW_SKELETON

