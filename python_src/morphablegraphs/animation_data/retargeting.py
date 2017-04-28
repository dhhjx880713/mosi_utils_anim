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
GAME_ENGINE_T_POSE_QUAT = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, -0.4034116551104901, 0.5807400765583645, 0.40341165511049004, 0.5807400765583645, 0.9718569538669156, 0.23557177509311245, -1.3791773619254053e-32, -9.140441879201479e-33, 0.9842093673640157, -0.17700825176506327, -5.454487879647522e-34, 3.0328292674444404e-33, 0.9988215725658287, 0.04853314513943063, -5.451814187658488e-11, -6.046242580699848e-10, 0.7247803814040166, 0.012592769755452584, 0.06644647837285525, -0.6856527447575631, 0.9983567056586047, 0.03270386259217816, -0.025876101002818737, -0.03930360078850377, 0.9965682409037466, 0.05203248433861332, -0.057026576671593956, 0.029871915718325807, 0.9297560338521583, 0.0938946447178257, 0.3530751373573903, -0.045557223234931374, 0.983733142437575, -0.0773187561935722, 0.14664502687446831, -0.06918200997050479, 0.9992691595380858, -0.015669716666339383, -0.028769659982579642, -0.019695518275262152, 0.9975311344163887, 0.04375497165440451, 0.034215414983264525, -0.04297026533543685, 0.9536776598189488, -0.04860775743712403, 0.2797821416180022, -0.09928826874727571, 0.9987647232093224, 0.005804170391239798, -0.03763753948846192, 0.03191794009533443, 0.9966582124862393, 0.051895568999548565, 0.046770662050906374, -0.042329216544453346, 0.9329213679196267, -0.08913572545954243, 0.33917282714358277, -0.08169661592258975, 0.9995615741071782, 0.009141850996134456, -0.017560293698507947, 0.022016407835221498, 0.9998819715691583, -0.0037191151629644803, 0.010434642587798942, -0.010645625742176174, 0.9322424364818923, -0.04990228518632508, 0.342604219994313, -0.1051482286944295, 0.9980915335828159, 0.005996155911584098, -0.04281223678247725, 0.044095907817706795, 0.9963697981737442, 0.045136674083357205, 0.05368016564496531, -0.048252935208484254, 0.2349924837443164, 0.31132145224589375, 0.8367206208319933, 0.38439054180573773, 0.7909589011799936, 0.46413884444997205, 0.3410569180890096, -0.20649292564252283, 0.9802741605131092, 0.09191251147798253, 0.14965372781082692, 0.09065551398812742, 0.7247803805906335, 0.012592769950269537, -0.06644647823624146, 0.6856527456270242, 0.9983567056586047, 0.03270386259217816, 0.02587610100281874, 0.039303600788503784, 0.9965682518976952, 0.05203243726125053, 0.05702650678920791, -0.029871764354427382, 0.9297559857385495, 0.09389467503083726, -0.3530751148595373, 0.045558317035600863, 0.9837333505577588, -0.0773206378561847, -0.14664363717997042, 0.06917989329673528, 0.9992693467177038, -0.015665213954855695, 0.028764041362077865, 0.019697809691502192, 0.9975314823914712, 0.04375194186430184, -0.03421120658988061, 0.042968623024728494, 0.9536775069845468, -0.04860738335400616, -0.2797828498945051, 0.09928792403976064, 0.9987646449635478, 0.005801503439721487, 0.03763947021552947, -0.03191859662597178, 0.9966579624324041, 0.051897847903993356, -0.046772431757685674, 0.04233035471733049, 0.9329213062387378, -0.0891352715094854, -0.3391729808717024, 0.08169717734010994, 0.9995615367687105, 0.009137951896895397, 0.01756132082424807, -0.022018902302604802, 0.9998819504532211, -0.0037172268673331425, -0.010435123687251087, 0.010647796763214817, 0.9322427205484806, -0.049904462569957994, -0.34260361428243297, 0.10514665035361338, 0.9980915983630915, 0.005999462058971182, 0.04281045823362956, -0.04409571858853277, 0.9963701280372368, 0.04513257610822263, -0.0536778650865031, 0.048252516293463533, -0.23499072279120342, -0.3113794298572072, 0.8366855880933993, 0.3844209119450591, 0.7910190999603561, 0.4640967281746974, -0.34091221681567874, 0.206595911918094, 0.9802667852924755, 0.09194228474474263, -0.14970309938761828, -0.09062355081331422, 0.9778768812145672, 0.20918127350714558, 2.5771780613672396e-10, 5.811305966263625e-10, 0.9909380491927413, -0.1343197031789609, -5.066775135883771e-11, -6.8679137736776675e-12, -0.1336980695398881, 0.966385580820868, 0.04173833379190176, -0.2155960270392212, 0.9725347805290746, 0.061775087466743615, 0.22395896183352518, -0.014224016457752058, 0.8338287133786118, -0.5510358060089089, 0.022043010546568757, -0.02456263274824042, 0.9633980641899633, -0.2673236757692123, 0.018056317099312928, 0.00872878577337811, -0.1336980711986006, 0.9663855803373587, -0.04173831577523998, 0.21559603166581406, 0.9725347764612302, 0.06177510013852306, -0.2239589750418668, 0.014224031586612895, 0.8338286703052938, -0.551035871807389, -0.022043023480577864, 0.0245626072356164, 0.963398245692903, -0.26732302190877494, -0.018056289743723242, -0.008728834635171634]
GAME_ENGINE_T_POSE_QUAT2 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.49999999999999994, 0.4999999999999999, 0.5000000000000001, 0.5, -0.403411655110279, 0.580740076558511, 0.4034116551102787, 0.580740076558511, 0.9718569538668924, 0.23557177509320731, -1.1102230246251565e-16, -5.968195498419857e-17, 0.9842093673640194, -0.1770082517650419, -1.1102230246251565e-16, 2.0412255914177302e-17, 0.9988215725658266, 0.048533145139473176, -5.451801732357127e-11, -6.046241931335687e-10, 0.7247803814041406, 0.01259276975548319, 0.06644647837284066, -0.6856527447574329, 0.9983567056586048, 0.03270386259213981, -0.025876101002807867, -0.03930360078853271, 0.9965682409037495, 0.05203248433858597, -0.05702657667158798, 0.02987191571830869, 0.9297560338521894, 0.0938946447181523, 0.35307513735725493, -0.04555722323467876, 0.9837331424375664, -0.07731875619354524, 0.14664502687453732, -0.06918200997051585, 0.9992691595380855, -0.015669716666298614, -0.028769659982590092, -0.01969551827528555, 0.9975311344163895, 0.043754971654412625, 0.034215414983260944, -0.04297026533541807, 0.9536776598189318, -0.04860775743717038, 0.27978214161803483, -0.09928826874732462, 0.9987647232093212, 0.005804170391240985, -0.037637539488489344, 0.03191794009533878, 0.9966582124862364, 0.05189556899956851, 0.04677066205094588, -0.042329216544453055, 0.9329213679197007, -0.08913572545953655, 0.33917282714338165, -0.08169661592258473, 0.9995615741071782, 0.009141850996163941, -0.017560293698508003, 0.02201640783520972, 0.9998819715691587, -0.0037191151629663473, 0.010434642587769397, -0.010645625742181223, 0.9322424364820183, -0.049902285186377245, 0.34260421999390694, -0.10514822869460992, 0.998091533582815, 0.005996155911585122, -0.042812236782500714, 0.044095907817706205, 0.9963697981737438, 0.04513667408334925, 0.05368016564497273, -0.04825293520849179, 0.23499248374450388, 0.3113214522469106, 0.8367206208326696, 0.38439054180332677, 0.7909589011799447, 0.4641388444503785, 0.3410569180887902, -0.20649292564215949, 0.9802741605131498, 0.0919125114780102, 0.1496537278105361, 0.09065551398814238, 0.7247803805906643, 0.012592769950276477, -0.06644647823619135, 0.6856527456269966, 0.9983567056586048, 0.03270386259213981, 0.025876101002807867, 0.03930360078853271, 0.9965682518976974, 0.05203243726121293, 0.0570265067892121, -0.029871764354401035, 0.9297559857383895, 0.09389467503126457, -0.35307511485991344, 0.04555831703506896, 0.9837333505578089, -0.0773206378561401, -0.14664363717968507, 0.06917989329668171, 0.9992693467177034, -0.01566521395483443, 0.02876404136210034, 0.019697809691507178, 0.9975314823914718, 0.04375194186426231, -0.034211206589911884, 0.04296862302472713, 0.9536775069844243, -0.04860738335408094, -0.2797828498948747, 0.09928792403985989, 0.998764644963548, 0.005801503439722635, 0.037639470215515725, -0.031918596625989504, 0.9966579624324049, 0.05189784790399392, -0.04677243175768359, 0.04233035471732105, 0.9329213062385932, -0.08913527150951468, -0.33917298087208786, 0.08169717734013165, 0.9995615367687092, 0.009137951896926988, 0.017561320824289475, -0.022018902302621688, 0.9998819504532211, -0.003717226867335589, -0.010435123687285549, 0.010647796763219678, 0.9322427205484667, -0.04990446256988014, -0.34260361428257413, 0.10514665035331486, 0.9980915983630924, 0.005999462058972744, 0.04281045823362141, -0.04409571858851882, 0.9963701280372361, 0.0451325761082384, -0.05367786508650819, 0.04825251629346674, -0.2349907227904058, -0.3113794298585417, 0.8366855880920271, 0.38442091194745265, 0.7910190999601722, 0.46409672817503295, -0.340912216815663, 0.20659591191807034, 0.9802667852924343, 0.09194228474479453, -0.149703099387827, -0.09062355081336093, 0.9778768812145765, 0.2091812735071013, 2.5771778407872157e-10, 5.811305966265294e-10, 0.9909380491926936, -0.13431970317931294, -5.0667640825130154e-11, -6.867799960398851e-12, -0.13369806954332625, 0.9663855808203412, 0.04173833379266804, -0.21559602703930203, 0.9725347805290097, 0.06177508746674525, 0.2239589618338054, -0.014224016457760126, 0.8338287133788124, -0.551035806008606, 0.022043010546564164, -0.024562632748227346, 0.9633980641900158, -0.2673236757690231, 0.018056317099323232, 0.008728785773340153, -0.1336980711990351, 0.9663855803372884, -0.04173831577537872, 0.2155960316658332, 0.9725347764611815, 0.06177510013854294, -0.2239589750420736, 0.014224031586576648, 0.833828670305466, -0.5510358718071292, -0.02204302348057051, 0.02456260723560276, 0.9633982456929496, -0.2673230219086038, -0.018056289743750415, -0.008728834635210405]
ROCKETBOX_TO_GAME_ENGINE_MAP = dict()
ROCKETBOX_TO_GAME_ENGINE_MAP["Hips"] = "Game_engine"
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



def get_quaternion_rotation_by_name(joint_name, frame, skeleton, root_offset=3):
    assert joint_name in skeleton.animated_joints
    joint_index = skeleton.animated_joints.index(joint_name)
    return frame[joint_index * 4 + root_offset : (joint_index + 1) * 4 + root_offset]


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
    if np.array_equal(a, b):
        return [1, 0, 0, 0]

    #aa = [a[0]*a[0], a[1]*a[1], a[2]*a[2]]
    #bb = [b[0] * b[0], b[1] * b[1], b[2] * b[2]]
    #if np.array_equal(aa,bb):
    #    return quaternion_from_euler()
    axis = normalize(np.cross(a, b))
    magnitude = np.linalg.norm(a) * np.linalg.norm(b)
    angle = math.acos(np.dot(a,b)/magnitude)
    #print "a",angle,axis,np.cross(a, b),a,b
    q = quaternion_from_axis_angle(axis, angle)
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
                joint_cos_map[j]["y"] = normalize(node.offset)
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


def rotate_bone2(src_skeleton,target_skeleton, src_name,target_name, src_to_target_joint_map, src_frame,target_frame, src_cos_map, target_cos_map, guess):
    q = guess
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
        print "found", src_name, q

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
    #target_cos_map["Root"]["y"] = [1,0,0]
    #target_cos_map["Root"]["x"] = [0,1,0]
    #target_cos_map["Game_engine"]["y"] = [1, 0, 0]
    target_cos_map["Game_engine"]["x"] = [1, 0, 0]
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
            #target = {"global_src_up_vec": [0,1,0], "global_src_x_vec": [1, 0, 0]}
            #target_frame[3:7] = find_rotation_analytically2(target_skeleton, "Root",
            #                                                   target, target_frame, target_cos_map)
            target_frame[3:7] = get_quaternion_rotation_by_name("Game_engine", GAME_ENGINE_T_POSE_QUAT, target_skeleton, root_offset=3)
        else:
            animated_joints = target_skeleton.animated_joints
            target_offset = 3

        for target_name in animated_joints:
            q = get_quaternion_rotation_by_name(target_name, GAME_ENGINE_T_POSE_QUAT, target_skeleton, root_offset=3)
            if target_name in target_to_src_joint_map.keys():
                src_name = target_to_src_joint_map[target_name]
                q = rotate_bone2(src_skeleton,target_skeleton, src_name,target_name,
                                  src_to_target_joint_map, src_frame,target_frame,
                                  src_cos_map, target_cos_map, q)
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
