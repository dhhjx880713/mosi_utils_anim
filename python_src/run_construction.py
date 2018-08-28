import os
import json
import numpy as np
import scipy.interpolate as si
import collections
from multiprocessing import Process
from morphablegraphs.animation_data.bvh import BVHReader
from morphablegraphs.animation_data import SkeletonBuilder, MotionVector
from morphablegraphs.construction.motion_model_constructor import MotionModelConstructor
from morphablegraphs.motion_model.motion_primitive_wrapper import MotionPrimitiveModelWrapper
from morphablegraphs.animation_data.skeleton_models import SKELETON_MODELS
from morphablegraphs.utilities import load_json_file
from morphablegraphs.construction.utils import get_cubic_b_spline_knots


ANIMATED_JOINTS = ["root", "pelvis","spine","spine_1","spine_2", "neck", "left_shoulder",
                   "left_elbow", "left_wrist",
                   "right_shoulder",
                   "right_elbow", "right_wrist", "left_hip",
                   "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"]
ANIMATED_JOINTS = dict()
ANIMATED_JOINTS["game_engine"] = [
    "Game_engine",
    "Root",
    "pelvis",
    "spine_01",
    "spine_02",
    "spine_03",
    "clavicle_l",
    "upperarm_l",
    "lowerarm_l",
    "hand_l",
    "clavicle_r",
    "upperarm_r",
    "lowerarm_r",
    "hand_r",
    "neck_01",
    "head",
    "thigh_l",
    "calf_l",
    "foot_l",
    "ball_l",
    "thigh_r",
    "calf_r",
    "foot_r",
    "ball_r"
]

ANIMATED_JOINTS["custom"] = [
    "FK_back1_jnt",
    "FK_back2_jnt",
    "FK_back4_jnt",
    "head_jnt",
    "R_shoulder_jnt",
    "R_upArm_jnt",
    "R_lowArm_jnt",
    "R_hand_jnt",
    "L_shoulder_jnt",
    "L_upArm_jnt",
    "L_lowArm_jnt",
    "L_hand_jnt",
    "L_upLeg_jnt",
    "L_lowLeg_jnt",
    "L_foot_jnt",
    "R_upLeg_jnt",
    "R_lowLeg_jnt",
    "R_foot_jnt"
]

MM_FILE_ENDING = "_quaternion_mm.json"


def load_skeleton(file_path, joint_filter=None, scale=1.0):
    target_bvh = BVHReader(file_path)
    bvh_joints = list(target_bvh.get_animated_joints())
    if joint_filter is not None:
        animated_joints = [j for j in bvh_joints if j in joint_filter]
    else:
        print("set default joints")
        animated_joints = bvh_joints
    skeleton = SkeletonBuilder().load_from_bvh(target_bvh, animated_joints, add_tool_joints=False)
    skeleton.scale(scale)
    return skeleton


def load_motion_vector_from_bvh_file(bvh_file_path, animated_joints):
    bvh_data = BVHReader(bvh_file_path)
    mv = MotionVector(None)
    mv.from_bvh_reader(bvh_data, filter_joints=False, animated_joints=animated_joints)
    return mv


def load_motion_data(motion_folder, max_count=np.inf, animated_joints=None):
    motions = collections.OrderedDict()
    for root, dirs, files in os.walk(motion_folder):
        for file_name in files:
            if file_name.endswith("bvh"):
                mv = load_motion_vector_from_bvh_file(motion_folder + os.sep + file_name, animated_joints)
                motions[file_name[:-4]] = np.array(mv.frames, dtype=np.float)
                if len(motions) > max_count:
                    break
    return motions


def get_standard_config():
    config = dict()
    config["n_basis_functions_spatial"] = 16#FIXME should be dynamic /n_frames divivded by 5
    config["n_spatial_basis_factor"] = 1.0/5.0
    config["fraction"] = 0.95
    config["n_basis_functions_temporal"] = 8
    config["npc_temporal"] = 3
    config["n_components"] = None
    config["precision_temporal"] = 0.99
    config["filter_window"] = 0
    return config


def export_frames_to_bvh(skeleton, frames, filename):
    print("export", len(frames[0]))
    mv = MotionVector()
    mv.frames = np.array([skeleton.add_fixed_joint_parameters_to_frame(f) for f in frames])
    print(mv.frames.shape)
    mv.export(skeleton, filename, add_time_stamp=False)


def export_motions(skeleton, motions):
    for idx, frames in enumerate(motions):
        export_frames_to_bvh(skeleton, frames, "out" + str(idx))


def define_sections_from_keyframes(motion_names, keyframes):
    sections = collections.OrderedDict()
    for key in motion_names:
        if key not in keyframes:
            continue
        m_sections = []
        keyframe = keyframes[key]
        section = dict()
        section["start_idx"] = 0
        section["end_idx"] = keyframe
        m_sections.append(section)
        section = dict()
        section["start_idx"] = keyframe
        section["end_idx"] = -1
        m_sections.append(section)
        sections[key] = m_sections
    return sections


def smooth_quaternion_frames(skeleton, frames, reference_frame):
    print("smooth", len(frames[0]), len(reference_frame))
    for frame in frames:
        for idx, node in enumerate(skeleton.animated_joints):
            o = idx*4 + 3
            ref_q = reference_frame[o:o+4]
            q = frame[o:o+4]
            if np.dot(q, ref_q) < 0:
                frame[o:o + 4] = -q
    return frames


def define_sections_from_annotations(motion_folder, motions):
    filtered_motions = collections.OrderedDict()
    sections = collections.OrderedDict()
    for key in motions.keys():
        annotations_file = motion_folder + os.sep + key + "_sections.json"
        if os.path.isfile(annotations_file):
            data = load_json_file(annotations_file)
            annotations = data["semantic_annotation"]
            motion_sections = dict()
            for label in annotations:
                annotations[label].sort()
                section = dict()
                section["start_idx"] = annotations[label][0]
                section["end_idx"] = annotations[label][-1]
                motion_sections[section["start_idx"]] = section
            motion_sections = collections.OrderedDict(sorted(motion_sections.items()))
            sections[key] = motion_sections.values()
            filtered_motions[key] = motions[key]

    if len(sections) > 0:
        motions = filtered_motions
        return motions, sections
    else:
        return motions, None


def convert_motion_to_static_motion_primitive(name, motion, skeleton, n_basis=7, degree=3):
    """
        Represent motion data as functional data, motion data should be narray<2d> n_frames * n_dims,
        the functional data has the shape n_basis * n_dims
    """

    motion_data = np.asarray(motion)
    n_frames, n_dims = motion_data.shape
    knots = get_cubic_b_spline_knots(n_basis, n_frames)
    x = list(range(n_frames))
    coeffs = [si.splrep(x, motion_data[:, i], k=degree,
                        t=knots[degree + 1: -(degree + 1)])[1][:-4] for i in range(n_dims)]
    coeffs = np.asarray(coeffs).T

    data = dict()
    data["name"] = name
    data["spatial_coeffs"] = coeffs.tolist()
    data["knots"] = knots.tolist()
    data["n_canonical_frames"] = len(motion)
    data["skeleton"] = skeleton.to_json()
    return data


def train_model(out_filename, name, motion_folder, skeleton, max_training_samples=100, animated_joints=None, save_skeleton=False):
    motions = load_motion_data(motion_folder, max_count=max_training_samples, animated_joints=animated_joints)

    ref_frame = None
    for key, m in motions.items():
        if ref_frame is None:
            ref_frame = m[0]
        motions[key] = smooth_quaternion_frames(skeleton, m, ref_frame)

    timewarping = None
    sections = None
    keyframes_filename = motion_folder+os.sep+"keyframes.json"
    timewarping_filename = motion_folder + os.sep + "timewarping.json"
    if os.path.isfile(keyframes_filename):
        keyframes = load_json_file(keyframes_filename)
        sections = define_sections_from_keyframes(motions.keys(), keyframes)
        filtered_motions = collections.OrderedDict()
        for key in motions.keys():
            if key in keyframes:
                filtered_motions[key] = motions[key]
        motions = filtered_motions
    elif os.path.isfile(timewarping_filename):
        timewarping = load_json_file(timewarping_filename)
        temp = dict()
        for key in timewarping.keys():
            print(key)
            temp[key[:-4]] = np.array(timewarping[key])

        timewarping = temp
    else:
        motions, sections = define_sections_from_annotations(motion_folder, motions)

    if len(motions) > 1:
        constructor = MotionModelConstructor(skeleton, get_standard_config())
        constructor.set_motions(motions)
        if timewarping is not None:
            constructor.set_timewarping(timewarping)
        elif sections is not None:
            constructor.set_dtw_sections(sections)
        #constructor.ground_node = "R_toe_tip_jnt_EndSite"
        model_data = constructor.construct_model(name, version=3, save_skeleton=save_skeleton)
        with open(out_filename, 'w') as outfile:
            json.dump(model_data, outfile)

    elif len(motions) == 1:
        keys = list(motions.keys())
        model_data = convert_motion_to_static_motion_primitive(name, motions[keys[0]], skeleton)
        with open(out_filename, 'w') as outfile:
            json.dump(model_data, outfile)
    else:
        print("Error: Did not find any BVH files in the directory", motion_folder)


def load_model(filename, skeleton):
    with open(filename, 'r') as infile:
        model_data = json.load(infile)
        model = MotionPrimitiveModelWrapper()
        model._initialize_from_json(skeleton.convert_to_mgrd_skeleton(), model_data)
        motion_spline = model.sample(False)
        frames = motion_spline.get_motion_vector()
        print(frames.shape)
        export_frames_to_bvh(skeleton, frames, "sample")


def main():
    # motion_file = motion_folder + os.sep + "17-11-20-Hybrit-VW_fix-screws-by-hand_002_snapPoseSkeleton.bvh"
    # skeleton = load_skeleton(skeleton_file, None, 10)
    skeleton_file = "skeleton.bvh"
    motion_folder = r"E:\projects\INTERACT\data\1 - MoCap\3 - Cutting\elementary_action_walk\leftStance"


    skeleton_file = "game_engine_target.bvh"

    #motion_folder = r"E:\projects\model_data\hybrit\retargeting\vw scenario\fix-screws-by-hand"
    #name = "fix-screws-by-hand"
    #motion_folder = r"E:\projects\model_data\hybrit\retargeting\vw scenario\screws-on-part"
    #name = "screws-on-part"
    motion_folder = r"E:\projects\model_data\hybrit\retargeting\vw scenario\pickup-screws"
    name = "pickup-screws"

    model_folder = r"E:\projects\model_data\hybrit\modeling"
    out_filename = model_folder + os.sep + name + MM_FILE_ENDING

    max_training_samples = 10
    joint_map = SKELETON_MODELS["game_engine"]["joints"]
    joint_filter = [joint_map[j] for j in ANIMATED_JOINTS]
    joint_filter += ["Root"]
    skeleton = load_skeleton(skeleton_file, joint_filter, 10)
    animated_joints = skeleton.animated_joints
    train_model(out_filename,name, motion_folder, skeleton, max_training_samples, animated_joints, save_skeleton=True)

    load_model(out_filename, skeleton)


def model_action():
    data_folder = r"E:\projects\model_data\hybrit\3_mirroring\bak\game_engine\place\out"
    model_folder = r"E:\projects\model_data\hybrit\3_mirroring\bak\game_engine\place\out"
    elementary_action = "place"
    max_training_samples = 100
    skeleton_file = "game_engine_target.bvh"
    joint_map = SKELETON_MODELS["game_engine"]["joints"]
    joint_filter = [joint_map[j] for j in ANIMATED_JOINTS]
    joint_filter += ["Root"]
    skeleton = load_skeleton(skeleton_file, joint_filter, 10)
    animated_joints = skeleton.animated_joints
    motion_folder = data_folder
    out_filename = model_folder + os.sep + elementary_action + MM_FILE_ENDING
    print("model", motion_folder, out_filename)
    train_model(out_filename, elementary_action, motion_folder, skeleton, max_training_samples, animated_joints, save_skeleton=True)


def model_actions(skeleton, data_folder, model_folder, input_folder_names, output_file_names, max_training_samples=200):

    for elementary_action in next(os.walk(data_folder))[1]:
        if input_folder_names is not None and elementary_action not in input_folder_names:
            continue
        motion_folder = data_folder + os.sep + elementary_action
        if output_file_names is None:
            out_filename = model_folder + os.sep + elementary_action + MM_FILE_ENDING
        else:
            idx = input_folder_names.index(elementary_action)
            out_filename = model_folder + os.sep + output_file_names[idx] + MM_FILE_ENDING

        train_model(out_filename, elementary_action, motion_folder, skeleton, max_training_samples, skeleton.animated_joints, save_skeleton=True)



def start_processes():
    processes = []
    actions = ["fix-screws-by-hand", "fix-screws-schrauber", "pickup-screws", "place-part", "screws-on-part", "pickup-part"]
    model_names = ["fixScrews_fixScrews", "fixScrewsSchrauber_fixScrewsSchrauber", "pickupScrew_pickupScrew", "placePart_placePart", "screwsOnPart_screwsOnPart", "pickupPart_pickupPart"]
    actions = [actions[-2]]
    model_names = [model_names[-2]]

    modeling_folder = r"E:\projects\model_data\hybrit\6_modeling"
    data_folder = modeling_folder + os.sep + "input - custom captury"
    model_folder = modeling_folder + os.sep + "output - custom"
    max_training_samples = 200
    skeleton_file = "game_engine_target.bvh"
    scale = 10
    skeleton_file = r"E:\projects\model_data\hybrit\game_engine_target2.bvh"
    scale = 1
    skeleton_file = r"E:\projects\model_data\hybrit\custom_target.bvh"
    scale = 2.54
    model_type = "custom"
    skeleton = load_skeleton(skeleton_file, None, scale)
    skeleton.skeleton_model = SKELETON_MODELS[model_type]
    skeleton.animated_joints = ANIMATED_JOINTS[model_type]

    actions = None
    model_names = None
    #actions = ["pickupScrew_pickupScrew", "placePart_placePart"]
    #model_names = ["pickupScrew_pickupScrew", "placePart_placePart"]

    for elementary_action in next(os.walk(data_folder))[1]:
        if actions is not None and elementary_action not in actions:
            continue
        motion_folder = data_folder + os.sep + elementary_action
        if model_names is None:
            out_filename = model_folder + os.sep + elementary_action + MM_FILE_ENDING
        else:
            idx = model_names.index(elementary_action)
            out_filename = model_folder + os.sep + model_names[idx] + MM_FILE_ENDING

        #p = Process(target=train_model,
        #            args=(out_filename, elementary_action, motion_folder, skeleton, max_training_samples,
        #                  skeleton.animated_joints, True))
        train_model(out_filename, elementary_action, motion_folder, skeleton, max_training_samples,
                    skeleton.animated_joints, save_skeleton=True)

        #processes.append(p)
        #p.start()

    for p in processes:
         p.join()

if __name__ == "__main__":
    start_processes()


