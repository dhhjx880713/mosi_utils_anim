""" This module contains mappings from a standard set of human joints to different skeletons

"""
import numpy as np
import collections
GAME_ENGINE_SKELETON_MODEL = collections.OrderedDict()
GAME_ENGINE_SKELETON_MODEL["Root"] = "Game_engine"
GAME_ENGINE_SKELETON_MODEL["Hips"] = "pelvis"
GAME_ENGINE_SKELETON_MODEL["Spine"] = "spine_01"
GAME_ENGINE_SKELETON_MODEL["Spine_1"] = "spine_02"
GAME_ENGINE_SKELETON_MODEL["LeftShoulder"] = "clavicle_l"
GAME_ENGINE_SKELETON_MODEL["RightShoulder"] = "clavicle_r"
GAME_ENGINE_SKELETON_MODEL["LeftArm"] = "upperarm_l"
GAME_ENGINE_SKELETON_MODEL["RightArm"] = "upperarm_r"
GAME_ENGINE_SKELETON_MODEL["LeftForeArm"] = "lowerarm_l"
GAME_ENGINE_SKELETON_MODEL["RightForeArm"] = "lowerarm_r"
GAME_ENGINE_SKELETON_MODEL["LeftHand"] = "hand_l"
GAME_ENGINE_SKELETON_MODEL["RightHand"] = "hand_r"
GAME_ENGINE_SKELETON_MODEL["LeftHip"] = "thigh_l"
GAME_ENGINE_SKELETON_MODEL["RightHip"] = "thigh_r"
GAME_ENGINE_SKELETON_MODEL["LeftKnee"] = "calf_l"
GAME_ENGINE_SKELETON_MODEL["RightKnee"] = "calf_r"
GAME_ENGINE_SKELETON_MODEL["LeftFoot"] = "foot_l"
GAME_ENGINE_SKELETON_MODEL["RightFoot"] = "foot_r"
GAME_ENGINE_SKELETON_MODEL["Neck"] = "neck_01"
GAME_ENGINE_SKELETON_MODEL["Head"] = "head"

ROCKETBOX_SKELETON_MODEL = collections.OrderedDict()
ROCKETBOX_SKELETON_MODEL["Root"] = "Hips"
ROCKETBOX_SKELETON_MODEL["Hips"] = "Hips"
ROCKETBOX_SKELETON_MODEL["Spine"] = "Spine"
ROCKETBOX_SKELETON_MODEL["Spine_1"] = "Spine_1"
ROCKETBOX_SKELETON_MODEL["LeftShoulder"] = "LeftShoulder"
ROCKETBOX_SKELETON_MODEL["RightShoulder"] = "RightShoulder"
ROCKETBOX_SKELETON_MODEL["LeftArm"] = "LeftArm"
ROCKETBOX_SKELETON_MODEL["RightArm"] = "RightArm"
ROCKETBOX_SKELETON_MODEL["LeftForeArm"] = "LeftForeArm"
ROCKETBOX_SKELETON_MODEL["RightForeArm"] = "lowerarm_r"
ROCKETBOX_SKELETON_MODEL["LeftHand"] = "LeftHand"
ROCKETBOX_SKELETON_MODEL["RightHand"] = "RightHand"
ROCKETBOX_SKELETON_MODEL["LeftHip"] = "LeftUpLeg"
ROCKETBOX_SKELETON_MODEL["RightHip"] = "RightUpLeg"
ROCKETBOX_SKELETON_MODEL["LeftKnee"] = "LeftLeg"
ROCKETBOX_SKELETON_MODEL["RightKnee"] = "RightLeg"
ROCKETBOX_SKELETON_MODEL["LeftFoot"] = "LeftFoot"
ROCKETBOX_SKELETON_MODEL["RightFoot"] = "RightFoot"
ROCKETBOX_SKELETON_MODEL["Neck"] = "Neck"
ROCKETBOX_SKELETON_MODEL["Head"] = "Head"

#print json.dumps(GAME_ENGINE_SKELETON_MODEL)


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
