import os
import collections
import numpy as np
from morphablegraphs.animation_data import BVHReader, SkeletonBuilder, MotionVector
from morphablegraphs.animation_data.skeleton_models import SKELETON_MODELS
from morphablegraphs.animation_data.motion_editing.motion_primitive_grounding import MotionPrimitiveGrounding, MP_CONFIGURATIONS
from morphablegraphs.animation_data.motion_editing.utils import add_heels_to_skeleton


MP_CONFIGURATIONS = collections.OrderedDict()

MP_CONFIGURATIONS["pickup-part"] = dict()
MP_CONFIGURATIONS["pickup-part"]["start_stance_foot"] = "right"
MP_CONFIGURATIONS["pickup-part"]["stance_foot"] = "right"
MP_CONFIGURATIONS["pickup-part"]["swing_foot"] = "left"
MP_CONFIGURATIONS["pickup-part"]["end_stance_foot"] = "left"
MP_CONFIGURATIONS["pickup-part"]["stance_mode"] = "both"
MP_CONFIGURATIONS["pickup-part"]["start_window_size"] = 0
MP_CONFIGURATIONS["pickup-part"]["end_window_size"] = 0


MP_CONFIGURATIONS["place-part"] = dict()
MP_CONFIGURATIONS["place-part"]["start_stance_foot"] = "right"
MP_CONFIGURATIONS["place-part"]["stance_foot"] = "right"
MP_CONFIGURATIONS["place-part"]["swing_foot"] = "left"
MP_CONFIGURATIONS["place-part"]["end_stance_foot"] = "left"
MP_CONFIGURATIONS["place-part"]["stance_mode"] = "both"
MP_CONFIGURATIONS["place-part"]["start_window_size"] = 0
MP_CONFIGURATIONS["place-part"]["end_window_size"] = 0

MP_CONFIGURATIONS["fix-screws-schrauber"] = dict()
MP_CONFIGURATIONS["fix-screws-schrauber"]["start_stance_foot"] = "right"
MP_CONFIGURATIONS["fix-screws-schrauber"]["stance_foot"] = "right"
MP_CONFIGURATIONS["fix-screws-schrauber"]["swing_foot"] = "left"
MP_CONFIGURATIONS["fix-screws-schrauber"]["end_stance_foot"] = "left"
MP_CONFIGURATIONS["fix-screws-schrauber"]["stance_mode"] = "both"
MP_CONFIGURATIONS["fix-screws-schrauber"]["start_window_size"] = 0
MP_CONFIGURATIONS["fix-screws-schrauber"]["end_window_size"] = 0

MP_CONFIGURATIONS["fix-screws-by-hand"] = dict()
MP_CONFIGURATIONS["fix-screws-by-hand"]["start_stance_foot"] = "right"
MP_CONFIGURATIONS["fix-screws-by-hand"]["stance_foot"] = "right"
MP_CONFIGURATIONS["fix-screws-by-hand"]["swing_foot"] = "left"
MP_CONFIGURATIONS["fix-screws-by-hand"]["end_stance_foot"] = "left"
MP_CONFIGURATIONS["fix-screws-by-hand"]["stance_mode"] = "both"
MP_CONFIGURATIONS["fix-screws-by-hand"]["start_window_size"] = 0
MP_CONFIGURATIONS["fix-screws-by-hand"]["end_window_size"] = 0

MP_CONFIGURATIONS["pickup-screws"] = dict()
MP_CONFIGURATIONS["pickup-screws"]["start_stance_foot"] = "right"
MP_CONFIGURATIONS["pickup-screws"]["stance_foot"] = "right"
MP_CONFIGURATIONS["pickup-screws"]["swing_foot"] = "left"
MP_CONFIGURATIONS["pickup-screws"]["end_stance_foot"] = "left"
MP_CONFIGURATIONS["pickup-screws"]["stance_mode"] = "both"
MP_CONFIGURATIONS["pickup-screws"]["start_window_size"] = 0
MP_CONFIGURATIONS["pickup-screws"]["end_window_size"] = 0


MP_CONFIGURATIONS["screws-on-part"] = dict()
MP_CONFIGURATIONS["screws-on-part"]["start_stance_foot"] = "right"
MP_CONFIGURATIONS["screws-on-part"]["stance_foot"] = "right"
MP_CONFIGURATIONS["screws-on-part"]["swing_foot"] = "left"
MP_CONFIGURATIONS["screws-on-part"]["end_stance_foot"] = "left"
MP_CONFIGURATIONS["screws-on-part"]["stance_mode"] = "both"
MP_CONFIGURATIONS["screws-on-part"]["start_window_size"] = 0
MP_CONFIGURATIONS["screws-on-part"]["end_window_size"] = 0

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
    n_frames = len(mv.frames)
    mv = mg.move_motion_to_ground(mv, step_offset=0, step_length=n_frames)
    mv = mg.ground_feet_completely(mv)
    mv = mg.align_to_origin(mv)
    file_name = bvh_file.split("\\")[-1][:-4]
    out_filename = out_path + os.sep + file_name + ".bvh"
    print("export to", out_filename, len(mg.skeleton.nodes))
    mv.export(mg.skeleton, out_filename, add_time_stamp=False)


def run_motion_grounding(in_path, out_path, mg, step_type, max_number=100):
    bvh_files = list(get_files(in_path, max_number, "bvh"))
    if len(bvh_files) > 0 and not os.path.exists(out_path):
        os.makedirs(out_path)
    for bvh_file in bvh_files:
        #bvh_file = in_path +os.sep+"walk_002_1_leftStance_682_745.bvh"
        run_grounding_on_bvh_file(bvh_file, out_path, mg, step_type)
        #return


def init_motion_primitive_grounding(skeleton_path, skeleton_model):
    bvh = BVHReader(skeleton_path)
    animated_joints = list(bvh.get_animated_joints())
    skeleton = SkeletonBuilder().load_from_bvh(bvh, animated_joints)  # filter here
    #skeleton.add_heels(skeleton_model)
    skeleton.aligning_root_node = "pelvis"
    skeleton_model["heel_offset"] = np.array(skeleton_model["heel_offset"])*0.5
    skeleton.skeleton_model = skeleton_model
    skeleton = add_heels_to_skeleton(skeleton, skeleton_model["joints"]["left_ankle"],
                                     skeleton_model["joints"]["right_ankle"],
                                     skeleton_model["joints"]["left_heel"],
                                     skeleton_model["joints"]["right_heel"],
                                     skeleton_model["heel_offset"])
    return MotionPrimitiveGrounding(skeleton, MP_CONFIGURATIONS, target_height=0)

if __name__ == "__main__":
    max_number = np.inf
    skeleton_model = SKELETON_MODELS["game_engine"]
    skeleton_file = "game_engine_skeleton.bvh"
    skeleton_file = r"E:\projects\model_data\hybrit\4_filter\vw scenario 3\fix-screws-by-hand\fix-screws-by-hand_002-snapPoseSkeleton.bvh"
    #ea_path = r"E:\projects\model_data\hybrit\4_modeling\input"
    ea_path = r"E:\projects\model_data\hybrit\4_filter\vw scenario 3"
    out_path = r"E:\projects\model_data\hybrit\5_grounding\vw scenario 3"
    mg = init_motion_primitive_grounding(skeleton_file, skeleton_model)
    for step_type in list(MP_CONFIGURATIONS.keys()):
        in_path = ea_path + os.sep + step_type
        ea_out_path = out_path + os.sep + step_type
        if not os.path.isdir(ea_out_path):
            os.makedirs(ea_out_path)
        run_motion_grounding(in_path, ea_out_path, mg, step_type, max_number)
