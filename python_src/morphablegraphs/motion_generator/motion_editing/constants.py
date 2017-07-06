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
IK_CHAINS_GAME_ENGINE_SKELETON["foot_l"] = {"root": "thigh_l", "joint": "calf_l", "joint_axis": [0, 1, 0], "end_effector_dir": [0,0,1]}
IK_CHAINS_GAME_ENGINE_SKELETON["foot_r"] = {"root": "thigh_r", "joint": "calf_r", "joint_axis": [0, 1, 0], "end_effector_dir": [0,0,1]}
IK_CHAINS_GAME_ENGINE_SKELETON["hand_r"] = {"root": "upperarm_r", "joint": "lowerarm_r", "joint_axis": [0, 1, 0], "end_effector_dir": [1,0,0]}
IK_CHAINS_GAME_ENGINE_SKELETON["hand_l"] = {"root": "upperarm_l", "joint": "lowerarm_l", "joint_axis": [0, 1, 0], "end_effector_dir": [1,0,0]}


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

SKELETON_DEFINITIONS = dict()
SKELETON_DEFINITIONS["raw"] = dict()
SKELETON_DEFINITIONS["raw"]["foot_joints"] = RAW_SKELETON_FOOT_JOINTS
SKELETON_DEFINITIONS["raw"]["left_foot"] = "LeftFoot"
SKELETON_DEFINITIONS["raw"]["right_foot"] = "RightFoot"
SKELETON_DEFINITIONS["raw"]["left_toe"] = "LeftToeBase"
SKELETON_DEFINITIONS["raw"]["right_toe"] = "RightToeBase"
SKELETON_DEFINITIONS["raw"]["left_heel"] = "LeftHeel"
SKELETON_DEFINITIONS["raw"]["right_heel"] = "RightHeel"
SKELETON_DEFINITIONS["raw"]["heel_offset"] = [0, -6.480602, 0]
SKELETON_DEFINITIONS["raw"]["ik_chains"] = IK_CHAINS_RAW_SKELETON

SKELETON_DEFINITIONS["game_engine"] = dict()
SKELETON_DEFINITIONS["game_engine"]["foot_joints"] = ["foot_l", "foot_r", "ball_r", "ball_l", "heel_r", "heel_l"]
SKELETON_DEFINITIONS["game_engine"]["left_foot"] = "foot_l"
SKELETON_DEFINITIONS["game_engine"]["right_foot"] = "foot_r"
SKELETON_DEFINITIONS["game_engine"]["left_toe"] = "ball_l"
SKELETON_DEFINITIONS["game_engine"]["right_toe"] = "ball_r"
SKELETON_DEFINITIONS["game_engine"]["left_heel"] = "heel_l"
SKELETON_DEFINITIONS["game_engine"]["right_heel"] = "heel_r"
SKELETON_DEFINITIONS["game_engine"]["heel_offset"] = HEEL_OFFSET
SKELETON_DEFINITIONS["game_engine"]["ik_chains"] = IK_CHAINS_GAME_ENGINE_SKELETON


