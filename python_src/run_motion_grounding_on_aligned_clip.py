import os
from python_src.morphablegraphs.animation_data import BVHReader, SkeletonBuilder, MotionVector
from python_src.morphablegraphs.animation_data.skeleton_models import *
from python_src.morphablegraphs.animation_data.motion_editing.motion_primitive_grounding import MotionPrimitiveGrounding, MP_CONFIGURATIONS
from python_src.morphablegraphs.animation_data.motion_editing.utils import add_heels_to_skeleton


def get_files(path, max_number, suffix="bvh"):
    count = 0
    for root, dirs, files in os.walk(path):
        for file_name in files:
            if file_name.endswith(suffix):
                yield path + os.sep + file_name
                count += 1
                if count >= max_number:
                    return


def run_grounding_on_bvh_file(bvh_file, out_path, mg, step_type):
    print("apply on", bvh_file)
    bvh = BVHReader(bvh_file)
    mv = MotionVector()
    mv.from_bvh_reader(bvh)
    mv = mg.run_grounding_on_motion_vector(mv, step_type, step_offset=0, step_length=len(mv.frames))
    file_name = bvh_file.split("\\")[-1][:-4]
    out_filename = file_name + "_grounded"
    mv.export(mg.skeleton, out_path + os.sep + out_filename, add_time_stamp=False)


def run_motion_grounding(in_path, out_path, mg, step_type, max_number=100):
    bvh_files = list(get_files(in_path, max_number, "bvh"))
    for bvh_file in bvh_files:
        run_grounding_on_bvh_file(bvh_file, out_path, mg, step_type)


def init_motion_primitive_grounding(skeleton_path, skeleton_model):
    bvh = BVHReader(skeleton_path)
    animated_joints = list(bvh.get_animated_joints())
    skeleton = SkeletonBuilder().load_from_bvh(bvh, animated_joints)  # filter here
    skeleton.aligning_root_node = "pelvis"
    skeleton.skeleton_model = skeleton_model
    skeleton = add_heels_to_skeleton(skeleton, skeleton_model["joints"]["left_ankle"],
                                     skeleton_model["joints"]["right_ankle"],
                                     skeleton_model["joints"]["left_heel"],
                                     skeleton_model["joints"]["right_heel"],
                                     skeleton_model["heel_offset"])
    return MotionPrimitiveGrounding(skeleton, MP_CONFIGURATIONS, target_height=0)

if __name__ == "__main__":
    max_number = 10000
    skeleton_model = GAME_ENGINE_SKELETON_MODEL
    skeleton_file = "game_engine_skeleton.bvh"
    mg = init_motion_primitive_grounding(skeleton_file, skeleton_model)
    for step_type in list(MP_CONFIGURATIONS.keys()):
        ea_path = "E:\\projects\\INTERACT\\data\\1 - MoCap\\4 - Alignment\\elementary_action_walk"
        in_path = ea_path + os.sep + step_type + "_game_engine_skeleton_new"
        out_path = ea_path +os.sep +"grounding"+ os.sep + step_type + "_game_engine_skeleton_new_grounded"
        run_motion_grounding(in_path, out_path, mg, step_type, max_number)
