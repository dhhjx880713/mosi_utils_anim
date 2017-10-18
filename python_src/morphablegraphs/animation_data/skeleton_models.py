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
RAW_SKELETON_JOINTS = collections.OrderedDict()
RAW_SKELETON_JOINTS["root"] = "Hips"
RAW_SKELETON_JOINTS["pelvis"] = "Hips"
RAW_SKELETON_JOINTS["spine"] = "Spine"
RAW_SKELETON_JOINTS["spine_1"] = "Spine1"
RAW_SKELETON_JOINTS["left_clavicle"] = "LeftShoulder"
RAW_SKELETON_JOINTS["right_clavicle"] = "RightShoulder"
RAW_SKELETON_JOINTS["left_shoulder"] = "LeftArm"
RAW_SKELETON_JOINTS["right_shoulder"] = "RightArm"
RAW_SKELETON_JOINTS["left_elbow"] = "LeftForeArm"
RAW_SKELETON_JOINTS["right_elbow"] = "RightForeArm"
RAW_SKELETON_JOINTS["left_wrist"] = "LeftHand"
RAW_SKELETON_JOINTS["right_wrist"] = "RightHand"
RAW_SKELETON_JOINTS["left_hip"] = "LeftUpLeg"
RAW_SKELETON_JOINTS["right_hip"] = "RightUpLeg"
RAW_SKELETON_JOINTS["left_knee"] = "LeftLeg"
RAW_SKELETON_JOINTS["right_knee"] = "RightLeg"
RAW_SKELETON_JOINTS["left_ankle"] = "LeftFoot"
RAW_SKELETON_JOINTS["right_ankle"] = "RightFoot"
RAW_SKELETON_JOINTS["left_toe"] = "LeftToeBase"
RAW_SKELETON_JOINTS["right_toe"] = "RightToeBase"
RAW_SKELETON_JOINTS["left_heel"] = "LeftHeel"
RAW_SKELETON_JOINTS["right_heel"] = "RightHeel"
RAW_SKELETON_JOINTS["neck"] = "Neck"
RAW_SKELETON_JOINTS["head"] = "Head"
RAW_SKELETON_MODEL = collections.OrderedDict()
RAW_SKELETON_MODEL["joints"] = RAW_SKELETON_JOINTS
RAW_SKELETON_MODEL["foot_joints"] = RAW_SKELETON_FOOT_JOINTS
RAW_SKELETON_MODEL["heel_offset"] = [0, -6.480602, 0]
RAW_SKELETON_MODEL["ik_chains"] = IK_CHAINS_RAW_SKELETON




GAME_ENGINE_JOINTS = collections.OrderedDict()
GAME_ENGINE_JOINTS["root"] = "Game_engine"
GAME_ENGINE_JOINTS["pelvis"] = "pelvis"
GAME_ENGINE_JOINTS["spine"] = "spine_01"
GAME_ENGINE_JOINTS["spine_1"] = "spine_02"
GAME_ENGINE_JOINTS["left_clavicle"] = "clavicle_l"
GAME_ENGINE_JOINTS["right_clavicle"] = "clavicle_r"
GAME_ENGINE_JOINTS["left_shoulder"] = "upperarm_l"
GAME_ENGINE_JOINTS["right_shoulder"] = "upperarm_r"
GAME_ENGINE_JOINTS["left_elbow"] = "lowerarm_l"
GAME_ENGINE_JOINTS["right_elbow"] = "lowerarm_r"
GAME_ENGINE_JOINTS["left_wrist"] = "hand_l"
GAME_ENGINE_JOINTS["right_wrist"] = "hand_r"
GAME_ENGINE_JOINTS["left_finger"] = "middle_03_l"
GAME_ENGINE_JOINTS["right_finger"] = "middle_03_r"
GAME_ENGINE_JOINTS["left_hip"] = "thigh_l"
GAME_ENGINE_JOINTS["right_hip"] = "thigh_r"
GAME_ENGINE_JOINTS["left_knee"] = "calf_l"
GAME_ENGINE_JOINTS["right_knee"] = "calf_r"
GAME_ENGINE_JOINTS["left_ankle"] = "foot_l"
GAME_ENGINE_JOINTS["right_ankle"] = "foot_r"
GAME_ENGINE_JOINTS["left_toe"] = "ball_l_EndSite"
GAME_ENGINE_JOINTS["right_toe"] = "ball_r_EndSite"
GAME_ENGINE_JOINTS["left_heel"] = "heel_l"
GAME_ENGINE_JOINTS["right_heel"] = "heel_r"
GAME_ENGINE_JOINTS["neck"] = "neck_01"
GAME_ENGINE_JOINTS["head"] = "head"
GAME_ENGINE_SKELETON_MODEL = collections.OrderedDict()
GAME_ENGINE_SKELETON_MODEL["joints"] = GAME_ENGINE_JOINTS
GAME_ENGINE_SKELETON_MODEL["foot_joints"] = ["foot_l", "foot_r", "ball_r", "ball_l", "heel_r", "heel_l"]
GAME_ENGINE_SKELETON_MODEL["heel_offset"] = (np.array([0, 2.45, 3.480602]) * 2.5).tolist()
GAME_ENGINE_SKELETON_MODEL["ik_chains"] = IK_CHAINS_GAME_ENGINE_SKELETON


ROCKETBOX_JOINTS = collections.OrderedDict()
ROCKETBOX_JOINTS["root"] = "Hips"
ROCKETBOX_JOINTS["pelvis"] = "Hips"
ROCKETBOX_JOINTS["spine"] = "Spine"
ROCKETBOX_JOINTS["spine_1"] = "Spine_1"
ROCKETBOX_JOINTS["left_clavicle"] = "LeftShoulder"
ROCKETBOX_JOINTS["right_clavicle"] = "RightShoulder"
ROCKETBOX_JOINTS["left_shoulder"] = "LeftArm"
ROCKETBOX_JOINTS["right_shoulder"] = "RightArm"
ROCKETBOX_JOINTS["left_elbow"] = "LeftForeArm"
ROCKETBOX_JOINTS["right_elbow"] = "RightForeArm"
ROCKETBOX_JOINTS["left_wrist"] = "LeftHand"
ROCKETBOX_JOINTS["right_wrist"] = "RightHand"
ROCKETBOX_JOINTS["left_hip"] = "LeftUpLeg"
ROCKETBOX_JOINTS["right_hip"] = "RightUpLeg"
ROCKETBOX_JOINTS["left_knee"] = "LeftLeg"
ROCKETBOX_JOINTS["right_knee"] = "RightLeg"
ROCKETBOX_JOINTS["left_ankle"] = "LeftFoot"
ROCKETBOX_JOINTS["right_ankle"] = "RightFoot"
ROCKETBOX_JOINTS["left_toe"] = "LeftToeBase"
ROCKETBOX_JOINTS["right_toe"] = "RightToeBase"
ROCKETBOX_JOINTS["left_heel"] = "LeftHeel"
ROCKETBOX_JOINTS["right_heel"] = "RightHeel"
ROCKETBOX_JOINTS["neck"] = "Neck"
ROCKETBOX_JOINTS["head"] = "Head"

ROCKETBOX_SKELETON_MODEL = collections.OrderedDict()
ROCKETBOX_SKELETON_MODEL["joints"] = ROCKETBOX_JOINTS
ROCKETBOX_SKELETON_MODEL["heel_offset"] = [0, -6.480602, 0]
ROCKETBOX_SKELETON_MODEL["foot_joints"] = RAW_SKELETON_FOOT_JOINTS
ROCKETBOX_SKELETON_MODEL["ik_chains"] = IK_CHAINS_RAW_SKELETON



CMU_SKELETON_JOINTS = collections.OrderedDict()
CMU_SKELETON_JOINTS["root"] = "hip"
CMU_SKELETON_JOINTS["pelvis"] = "hip"
CMU_SKELETON_JOINTS["spine"] = "abdomen"
CMU_SKELETON_JOINTS["spine_1"] = "chest"
CMU_SKELETON_JOINTS["left_clavicle"] = "lCollar"
CMU_SKELETON_JOINTS["right_clavicle"] = "rCollar"
CMU_SKELETON_JOINTS["left_shoulder"] = "lShldr"
CMU_SKELETON_JOINTS["right_shoulder"] = "rShldr"
CMU_SKELETON_JOINTS["left_elbow"] = "lForeArm"
CMU_SKELETON_JOINTS["right_elbow"] = "rForeArm"
CMU_SKELETON_JOINTS["left_wrist"] = "lHand"
CMU_SKELETON_JOINTS["right_wrist"] = "rHand"
CMU_SKELETON_JOINTS["left_hip"] = "lThigh"
CMU_SKELETON_JOINTS["right_hip"] = "rThigh"
CMU_SKELETON_JOINTS["left_knee"] = "lShin"
CMU_SKELETON_JOINTS["right_knee"] = "rShin"
CMU_SKELETON_JOINTS["left_ankle"] = "lFoot"
CMU_SKELETON_JOINTS["right_ankle"] = "rFoot"
CMU_SKELETON_JOINTS["left_toe"] = "lFoot_EndSite"
CMU_SKELETON_JOINTS["right_toe"] = "rFoot_EndSite"
CMU_SKELETON_JOINTS["left_heel"] = None
CMU_SKELETON_JOINTS["right_heel"] = None
CMU_SKELETON_JOINTS["neck"] = "neck"
CMU_SKELETON_JOINTS["head"] = "head"
CMU_SKELETON_MODEL = collections.OrderedDict()
CMU_SKELETON_MODEL["joints"] = CMU_SKELETON_JOINTS
CMU_SKELETON_MODEL["foot_joints"] = []

MOVIEMATION_SKELETON_JOINTS = collections.OrderedDict()
MOVIEMATION_SKELETON_JOINTS["root"] = "Hips"
MOVIEMATION_SKELETON_JOINTS["pelvis"] = "Hips"
MOVIEMATION_SKELETON_JOINTS["spine"] = "Ab"
MOVIEMATION_SKELETON_JOINTS["spine_1"] = "Chest"
MOVIEMATION_SKELETON_JOINTS["left_clavicle"] = "LeftCollar"
MOVIEMATION_SKELETON_JOINTS["right_clavicle"] = "RightCollar"
MOVIEMATION_SKELETON_JOINTS["left_shoulder"] = "LeftShoulder"
MOVIEMATION_SKELETON_JOINTS["right_shoulder"] = "RightShoulder"
MOVIEMATION_SKELETON_JOINTS["left_elbow"] = "LeftElbow"
MOVIEMATION_SKELETON_JOINTS["right_elbow"] = "RightElbow"
MOVIEMATION_SKELETON_JOINTS["left_wrist"] = "LeftWrist"
MOVIEMATION_SKELETON_JOINTS["right_wrist"] = "RightWrist"
MOVIEMATION_SKELETON_JOINTS["left_hip"] = "LeftHip"
MOVIEMATION_SKELETON_JOINTS["right_hip"] = "RightHip"
MOVIEMATION_SKELETON_JOINTS["left_knee"] = "LeftKnee"
MOVIEMATION_SKELETON_JOINTS["right_knee"] = "RightKnee"
MOVIEMATION_SKELETON_JOINTS["left_ankle"] = "LeftAnkle"
MOVIEMATION_SKELETON_JOINTS["right_ankle"] = "RightAnkle"
MOVIEMATION_SKELETON_JOINTS["left_toe"] = "LeftAnkle_EndSite"
MOVIEMATION_SKELETON_JOINTS["right_toe"] = "RightAnkle_EndSite"
MOVIEMATION_SKELETON_JOINTS["left_heel"] = None
MOVIEMATION_SKELETON_JOINTS["right_heel"] = None
MOVIEMATION_SKELETON_JOINTS["neck"] = "Neck"
MOVIEMATION_SKELETON_JOINTS["head"] = "Head"
MOVIEMATION_SKELETON_MODEL = collections.OrderedDict()
MOVIEMATION_SKELETON_MODEL["joints"] = MOVIEMATION_SKELETON_JOINTS
MOVIEMATION_SKELETON_MODEL["foot_joints"] = []


MCS_SKELETON_JOINTS = collections.OrderedDict()
MCS_SKELETON_JOINTS["root"] = "Hips"
MCS_SKELETON_JOINTS["pelvis"] = "Hips"
MCS_SKELETON_JOINTS["spine"] = None
MCS_SKELETON_JOINTS["spine_1"] = "Chest"
MCS_SKELETON_JOINTS["left_clavicle"] = "LeftCollar"
MCS_SKELETON_JOINTS["right_clavicle"] = "RightCollar"
MCS_SKELETON_JOINTS["left_shoulder"] = "LeftShoulder"
MCS_SKELETON_JOINTS["right_shoulder"] = "RightShoulder"
MCS_SKELETON_JOINTS["left_elbow"] = "LeftElbow"
MCS_SKELETON_JOINTS["right_elbow"] = "RightElbow"
MCS_SKELETON_JOINTS["left_wrist"] = "LeftWrist"
MCS_SKELETON_JOINTS["right_wrist"] = "RightWrist"
MCS_SKELETON_JOINTS["left_hip"] = "LeftHip"
MCS_SKELETON_JOINTS["right_hip"] = "RightHip"
MCS_SKELETON_JOINTS["left_knee"] = "LeftKnee"
MCS_SKELETON_JOINTS["right_knee"] = "RightKnee"
MCS_SKELETON_JOINTS["left_ankle"] = "LeftAnkle"
MCS_SKELETON_JOINTS["right_ankle"] = "RightAnkle"
MCS_SKELETON_JOINTS["left_toe"] = "LeftAnkle_EndSite"
MCS_SKELETON_JOINTS["right_toe"] = "RightAnkle_EndSite"
MCS_SKELETON_JOINTS["left_heel"] = None
MCS_SKELETON_JOINTS["right_heel"] = None
MCS_SKELETON_JOINTS["neck"] = "Neck"
MCS_SKELETON_JOINTS["head"] = "Head"
MCS_SKELETON_MODEL = collections.OrderedDict()
MCS_SKELETON_MODEL["joints"] = MCS_SKELETON_JOINTS
MCS_SKELETON_MODEL["foot_joints"] = []

SKELETON_MODELS = dict()
SKELETON_MODELS["rocketbox"] = ROCKETBOX_SKELETON_MODEL
SKELETON_MODELS["game_engine"] = GAME_ENGINE_SKELETON_MODEL
SKELETON_MODELS["cmu"] = CMU_SKELETON_MODEL
SKELETON_MODELS["mcs"] = MCS_SKELETON_MODEL
SKELETON_MODELS["moviemation"] = MOVIEMATION_SKELETON_MODEL


