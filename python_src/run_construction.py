import os
import json
import numpy as np
from .morphablegraphs.animation_data.bvh import BVHReader
from .morphablegraphs.animation_data import SkeletonBuilder, MotionVector
from .morphablegraphs.construction.motion_model_constructor import MotionModelConstructor
from .morphablegraphs.motion_model.motion_primitive_wrapper import MotionPrimitiveModelWrapper
MM_FILE_ENDING = "_quaternion_mm.json"


def load_skeleton(file_path):
    target_bvh = BVHReader(file_path)
    #animated_joints = list(target_bvh.get_animated_joints())
    animated_joints = None
    skeleton = SkeletonBuilder().load_from_bvh(target_bvh, animated_joints, add_tool_joints=False)
    return skeleton


def load_motion_vector_from_bvh_file(bvh_file_path):
    bvh_data = BVHReader(bvh_file_path)
    mv = MotionVector(None)
    mv.from_bvh_reader(bvh_data, filter_joints=True)
    return mv


def load_motion_data(motion_folder, max_count=np.inf):
    motions = []
    for root, dirs, files in os.walk(motion_folder):
        for file_name in files:
            if file_name.endswith("bvh"):
                print("read", file_name)
                mv = load_motion_vector_from_bvh_file(motion_folder + os.sep + file_name)
                motions.append(mv.frames)
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
    mv = MotionVector()
    mv.frames = np.array([skeleton.generate_complete_frame_vector_from_reference(f) for f in frames])
    print(mv.frames.shape)
    mv.export(skeleton, filename, add_time_stamp=False)


def export_motions(skeleton, motions):
    for idx, frames in enumerate(motions):
        export_frames_to_bvh(skeleton, frames, "out" + str(idx))


def train_model(filename, motion_folder, skeleton_file, max_training_samples=100):
    bvh_reader = BVHReader(skeleton_file)
    motions = load_motion_data(motion_folder, max_count=max_training_samples)
    constructor = MotionModelConstructor(bvh_reader, get_standard_config())
    constructor.set_motions(motions)
    model_data = constructor.construct_model(name, version=3)
    with open(filename, 'w') as outfile:
        json.dump(model_data, outfile)


def load_model(filename, skeleton):
    with open(filename, 'r') as infile:
        model_data = json.load(infile)
        model = MotionPrimitiveModelWrapper()
        model._initialize_from_json(skeleton.convert_to_mgrd_skeleton(), model_data)
        motion_spline = model.sample(True)
        frames = motion_spline.get_motion_vector()
        print(frames.shape)
        export_frames_to_bvh(skeleton, frames, "sample")

if __name__ == "__main__":
    name = "check"
    skeleton_file = "skeleton.bvh"
    motion_folder = r"E:\projects\INTERACT\data\1 - MoCap\3 - Cutting\elementary_action_walk\leftStance"
    max_training_samples = 10
    filename = name + MM_FILE_ENDING
    skeleton = load_skeleton(skeleton_file)

    train_model(filename, motion_folder, skeleton_file, max_training_samples)
    load_model(filename, skeleton)
