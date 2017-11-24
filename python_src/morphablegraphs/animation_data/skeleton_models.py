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
GAME_ENGINE_JOINTS["spine_1"] = "spine_03"
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
GAME_ENGINE_JOINTS["left_toe"] = "ball_l"
GAME_ENGINE_JOINTS["right_toe"] = "ball_r"
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



MH_CMU_SKELETON_JOINTS = collections.OrderedDict()
MH_CMU_SKELETON_JOINTS["root"] = "CMU compliant skeleton"
MH_CMU_SKELETON_JOINTS["pelvis"] = "Hips"
MH_CMU_SKELETON_JOINTS["spine"] = "Spine"
MH_CMU_SKELETON_JOINTS["spine_1"] = "Spine1"
MH_CMU_SKELETON_JOINTS["left_clavicle"] = "LeftShoulder"
MH_CMU_SKELETON_JOINTS["right_clavicle"] = "RightShoulder"
MH_CMU_SKELETON_JOINTS["left_shoulder"] = "LeftArm"
MH_CMU_SKELETON_JOINTS["right_shoulder"] = "RightArm"
MH_CMU_SKELETON_JOINTS["left_elbow"] = "LeftForeArm"
MH_CMU_SKELETON_JOINTS["right_elbow"] = "RightForeArm"
MH_CMU_SKELETON_JOINTS["left_wrist"] = "LeftHand"
MH_CMU_SKELETON_JOINTS["right_wrist"] = "RightHand"
MH_CMU_SKELETON_JOINTS["left_hip"] = "LeftUpLeg"
MH_CMU_SKELETON_JOINTS["right_hip"] = "RightUpLeg"
MH_CMU_SKELETON_JOINTS["left_knee"] = "LeftLeg"
MH_CMU_SKELETON_JOINTS["right_knee"] = "RightLeg"
MH_CMU_SKELETON_JOINTS["left_ankle"] = "LeftFoot"
MH_CMU_SKELETON_JOINTS["right_ankle"] = "RightFoot"
MH_CMU_SKELETON_JOINTS["left_toe"] = "LeftToeBase"
MH_CMU_SKELETON_JOINTS["right_toe"] = "RightToeBase"
MH_CMU_SKELETON_JOINTS["left_heel"] = None
MH_CMU_SKELETON_JOINTS["right_heel"] = None
MH_CMU_SKELETON_JOINTS["neck"] = "Neck"
MH_CMU_SKELETON_JOINTS["head"] = "Head"

MH_CMU_SKELETON_MODEL = collections.OrderedDict()
MH_CMU_SKELETON_MODEL["joints"] = MH_CMU_SKELETON_JOINTS
MH_CMU_SKELETON_MODEL["foot_joints"] = []



ICLONE_SKELETON_JOINTS = collections.OrderedDict()
ICLONE_SKELETON_JOINTS["root"] = "CC_Base_BoneRoot"
ICLONE_SKELETON_JOINTS["pelvis"] = "CC_Base_Pelvis"
ICLONE_SKELETON_JOINTS["spine"] = "CC_Base_Waist"
ICLONE_SKELETON_JOINTS["spine_1"] = "CC_Base_Spine02"
ICLONE_SKELETON_JOINTS["left_clavicle"] = "CC_Base_L_Clavicle"
ICLONE_SKELETON_JOINTS["right_clavicle"] = "CC_Base_R_Clavicle"
ICLONE_SKELETON_JOINTS["left_shoulder"] = "CC_Base_L_Upperarm"
ICLONE_SKELETON_JOINTS["right_shoulder"] = "CC_Base_R_Upperarm"
ICLONE_SKELETON_JOINTS["left_elbow"] = "CC_Base_L_Forearm"
ICLONE_SKELETON_JOINTS["right_elbow"] = "CC_Base_R_Forearm"
ICLONE_SKELETON_JOINTS["left_wrist"] = "CC_Base_L_Hand"
ICLONE_SKELETON_JOINTS["right_wrist"] = "CC_Base_R_Hand"
ICLONE_SKELETON_JOINTS["left_hip"] = "CC_Base_L_Thigh"
ICLONE_SKELETON_JOINTS["right_hip"] = "CC_Base_R_Thigh"
ICLONE_SKELETON_JOINTS["left_knee"] = "CC_Base_L_Calf"
ICLONE_SKELETON_JOINTS["right_knee"] = "CC_Base_R_Calf"
ICLONE_SKELETON_JOINTS["left_ankle"] = "CC_Base_L_Foot"
ICLONE_SKELETON_JOINTS["right_ankle"] = "CC_Base_R_Foot"
ICLONE_SKELETON_JOINTS["left_toe"] = "CC_Base_L_ToeBase"
ICLONE_SKELETON_JOINTS["right_toe"] = "CC_Base_R_ToeBase"
ICLONE_SKELETON_JOINTS["left_heel"] = None
ICLONE_SKELETON_JOINTS["right_heel"] = None
ICLONE_SKELETON_JOINTS["neck"] = "CC_Base_NeckTwist01"
ICLONE_SKELETON_JOINTS["head"] = "CC_Base_Head"

ICLONE_SKELETON_MODEL = collections.OrderedDict()
ICLONE_SKELETON_MODEL["joints"] = ICLONE_SKELETON_JOINTS
ICLONE_SKELETON_MODEL["foot_joints"] = []

CUSTOM_SKELETON_JOINTS = collections.OrderedDict()
CUSTOM_SKELETON_JOINTS["root"] = "FK_back1_jnt"
CUSTOM_SKELETON_JOINTS["pelvis"] = "FK_back1_jnt"
CUSTOM_SKELETON_JOINTS["spine"] = "FK_back2_jnt"
CUSTOM_SKELETON_JOINTS["spine_1"] = "FK_back3_jnt"
CUSTOM_SKELETON_JOINTS["left_clavicle"] = "L_shoulder_jnt"
CUSTOM_SKELETON_JOINTS["right_clavicle"] = "R_shoulder_jnt"
CUSTOM_SKELETON_JOINTS["left_shoulder"] = "L_upArm_jnt"
CUSTOM_SKELETON_JOINTS["right_shoulder"] = "R_upArm_jnt"
CUSTOM_SKELETON_JOINTS["left_elbow"] = "L_lowArm_jnt"
CUSTOM_SKELETON_JOINTS["right_elbow"] = "R_lowArm_jnt"
CUSTOM_SKELETON_JOINTS["left_wrist"] = "L_hand_jnt"
CUSTOM_SKELETON_JOINTS["right_wrist"] = "R_hand_jnt"
CUSTOM_SKELETON_JOINTS["left_hip"] = "L_upLeg_jnt"
CUSTOM_SKELETON_JOINTS["right_hip"] = "R_upLeg_jnt"
CUSTOM_SKELETON_JOINTS["left_knee"] = "L_lowLeg_jnt"
CUSTOM_SKELETON_JOINTS["right_knee"] = "R_lowLeg_jnt"
CUSTOM_SKELETON_JOINTS["left_ankle"] = "L_foot_jnt"
CUSTOM_SKELETON_JOINTS["right_ankle"] = "R_foot_jnt"
CUSTOM_SKELETON_JOINTS["left_toe"] = "L_toe_jnt"
CUSTOM_SKELETON_JOINTS["right_toe"] = "R_toe_jnt"
CUSTOM_SKELETON_JOINTS["left_heel"] = None
CUSTOM_SKELETON_JOINTS["right_heel"] = None
CUSTOM_SKELETON_JOINTS["neck"] = "FK_back4_jnt"
CUSTOM_SKELETON_JOINTS["head"] = "head_jnt"

CUSTOM_SKELETON_MODEL = collections.OrderedDict()
CUSTOM_SKELETON_MODEL["joints"] = CUSTOM_SKELETON_JOINTS
CUSTOM_SKELETON_MODEL["foot_joints"] = []


SKELETON_MODELS = dict()
SKELETON_MODELS["rocketbox"] = ROCKETBOX_SKELETON_MODEL
SKELETON_MODELS["game_engine"] = GAME_ENGINE_SKELETON_MODEL
SKELETON_MODELS["raw"] = RAW_SKELETON_MODEL
SKELETON_MODELS["cmu"] = CMU_SKELETON_MODEL
SKELETON_MODELS["mcs"] = MCS_SKELETON_MODEL
SKELETON_MODELS["mh_cmu"] = MH_CMU_SKELETON_MODEL
SKELETON_MODELS["iclone"] = ICLONE_SKELETON_MODEL
SKELETON_MODELS["moviemation"] = MOVIEMATION_SKELETON_MODEL
SKELETON_MODELS["custom"] = CUSTOM_SKELETON_MODEL

JOINT_CHILD_MAP = dict()
JOINT_CHILD_MAP["root"] = "pelvis"
JOINT_CHILD_MAP["pelvis"] = "spine"
JOINT_CHILD_MAP["spine"] = "spine_1"
JOINT_CHILD_MAP["spine_1"] = "neck"
JOINT_CHILD_MAP["neck"] = "head"
JOINT_CHILD_MAP["left_clavicle"] = "left_shoulder"
JOINT_CHILD_MAP["left_shoulder"] = "left_elbow"
JOINT_CHILD_MAP["left_elbow"] = "left_wrist"
JOINT_CHILD_MAP["right_clavicle"] = "right_shoulder"
JOINT_CHILD_MAP["right_shoulder"] = "right_elbow"
JOINT_CHILD_MAP["right_elbow"] = "right_wrist"
JOINT_CHILD_MAP["left_hip"] = "left_knee"
JOINT_CHILD_MAP["left_knee"] = "left_ankle"
JOINT_CHILD_MAP["right_elbow"] = "right_wrist"
JOINT_CHILD_MAP["right_hip"] = "right_knee"
JOINT_CHILD_MAP["right_knee"] = "right_ankle"

JOINT_PARENT_MAP = {v: k for k, v in JOINT_CHILD_MAP.items()}
