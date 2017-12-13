import os
import json
import numpy as np
import collections
from morphablegraphs.animation_data.bvh import BVHReader
from morphablegraphs.animation_data import SkeletonBuilder, MotionVector
from morphablegraphs.construction.motion_model_constructor import MotionModelConstructor
from morphablegraphs.motion_model.motion_primitive_wrapper import MotionPrimitiveModelWrapper
from morphablegraphs.animation_data.skeleton_models import SKELETON_MODELS, ANIMATED_JOINTS
from morphablegraphs.utilities import load_json_file


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
                motions[file_name[:-4]] = mv.frames
                if len(motions) > max_count:
                    break
    return motions


def get_standard_config():
    config = dict()
    config["n_basis_functions_spatial"] = 16
    config["fraction"] = 0.95
    config["n_basis_functions_temporal"] = 8
    config["npc_temporal"] = 3
    config["n_components"] = None
    config["precision_temporal"] = 0.99
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
    sections = []
    for key in motion_names:
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
        sections.append(m_sections)
    return sections

def load_sections_from_keyframes_file(motion_names, filename):
    keyframes = load_json_file(filename)
    return define_sections_from_keyframes(motion_names, keyframes)

def train_model(out_filename,name,  motion_folder, skeleton, max_training_samples=100, animated_joints=None, save_skeleton=False):
    motions = load_motion_data(motion_folder, max_count=max_training_samples, animated_joints=animated_joints)

    keyframes_filename = motion_folder+os.sep+"keyframes.json"
    if os.path.isfile(keyframes_filename):
        sections = load_sections_from_keyframes_file(motions.keys(), keyframes_filename)
    else:
        sections = None
    constructor = MotionModelConstructor(skeleton, get_standard_config())
    constructor.set_motions(motions.values())
    constructor.set_dtw_sections(sections)
    model_data = constructor.construct_model(name, version=3, save_skeleton=save_skeleton)
    with open(out_filename, 'w') as outfile:
        json.dump(model_data, outfile)


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
    skeleton_file = "skeleton.bvh"
    motion_folder = r"E:\projects\INTERACT\data\1 - MoCap\3 - Cutting\elementary_action_walk\leftStance"

    skeleton_file = "game_engine_target.bvh"

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

def model_actions():
    data_folder = r"E:\projects\model_data\hybrit\processed_data"
    model_folder = r"E:\projects\model_data\hybrit\modeling"
    max_training_samples = 100
    skeleton_file = "game_engine_target.bvh"
    joint_map = SKELETON_MODELS["game_engine"]["joints"]
    action_filter = ["pickup-part", "place-part"]
    joint_filter = [joint_map[j] for j in ANIMATED_JOINTS]
    joint_filter += ["Root"]
    skeleton = load_skeleton(skeleton_file, joint_filter, 10)
    animated_joints = skeleton.animated_joints
    for elementary_action in next(os.walk(data_folder))[1]:
        if elementary_action not in action_filter:
            continue
        motion_folder = data_folder + os.sep + elementary_action
        out_filename = model_folder + os.sep + elementary_action + MM_FILE_ENDING
        print("model", motion_folder, out_filename)
        train_model(out_filename, elementary_action, motion_folder, skeleton, max_training_samples, animated_joints, save_skeleton=True)


if __name__ == "__main__":
    model_actions()
