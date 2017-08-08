import os
import numpy as np
from .morphablegraphs.construction.segmentation import Segmentation
from .morphablegraphs.animation_data.bvh import BVHReader
from .morphablegraphs.animation_data import MotionVector, SkeletonBuilder


def load_skeleton(file_path):
    target_bvh = BVHReader(file_path)
    animated_joints = None
    skeleton = SkeletonBuilder().load_from_bvh(target_bvh,animated_joints, add_tool_joints=False)
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


def export_frames_to_bvh(skeleton, frames, filename):
    mv = MotionVector()
    mv.frames = np.array([skeleton.generate_complete_frame_vector_from_reference(f) for f in frames])
    print(mv.frames.shape)
    mv.export(skeleton, ".", filename, add_time_stamp=False)


def export_motions(skeleton, motions, name):
    for idx, frames in enumerate(motions):
        export_frames_to_bvh(skeleton, frames, name + str(idx))


def run_segmentation(skeleton, motions, start_keyframe_coord, end_keyframe_coord):
    seg = Segmentation(skeleton)

    start_keyframe = motions[start_keyframe_coord[0]][start_keyframe_coord[1]]
    end_keyframe = motions[end_keyframe_coord[0]][end_keyframe_coord[1]]
    segments = seg.extract_segments(motions, start_keyframe, end_keyframe)
    print("found",len(segments), "segments")
    export_motions(skeleton, segments, "out")

if __name__ == "__main__":
    skeleton_file = "skeleton.bvh"
    motion_folder = r"E:\projects\INTERACT\data\1 - MoCap\2 - Rocketbox retargeting\Take_walk"
    max_samples = 5

    skeleton = load_skeleton(skeleton_file)
    motions = load_motion_data(motion_folder, max_count=max_samples)
    start_keyframe_coord = (3, 476)
    end_keyframe_coord = (3, 507)
    run_segmentation(skeleton, motions, start_keyframe_coord, end_keyframe_coord)
