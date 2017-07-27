import os
import collections
from morphablegraphs.animation_data import BVHReader, SkeletonBuilder, MotionVector
from morphablegraphs.animation_data.skeleton_models import *
from morphablegraphs.animation_data.motion_editing.utils import add_heels_to_skeleton, generate_root_constraint_for_one_foot, generate_root_constraint_for_two_feet, \
    guess_ground_height, smooth_root_translation_at_end, smooth_root_translation_at_start
from morphablegraphs.animation_data.motion_editing.motion_grounding import MotionGroundingConstraint, generate_ankle_constraint_from_toe, create_ankle_constraint_from_toe_and_heel
from morphablegraphs.animation_data.motion_editing.analytical_inverse_kinematics import AnalyticalLimbIK
from morphablegraphs.external.transformations import quaternion_from_matrix, quaternion_multiply, quaternion_matrix, quaternion_slerp

LEFT_FOOT = "LeftFoot"
RIGHT_FOOT = "RightFoot"
RIGHT_TOE = "RightToeBase"
LEFT_TOE = "LeftToeBase"
RIGHT_KNEE = "RightLeg"
LEFT_KNEE = "LeftLeg"
RIGHT_HIP = "RightUpLeg"
LEFT_HIP = "LeftUpLeg"
LEFT_HEEL = "LeftHeel"
RIGHT_HEEL = "RightHeel"



def generate_ankle_constraint_from_toe_without_orientation(skeleton, frames, frame_idx, ankle_joint_name, toe_joint_name, target_ground_height, toe_pos = None):
    """ create a constraint on the ankle position based on the toe constraint position"""
    #print "add toe constraint"
    if toe_pos is None:
        ct = skeleton.nodes[toe_joint_name].get_global_position(frames[frame_idx])
        ct[1] = target_ground_height  # set toe constraint on the ground
    else:
        ct = toe_pos
    a = skeleton.nodes[ankle_joint_name].get_global_position(frames[frame_idx])
    t = skeleton.nodes[toe_joint_name].get_global_position(frames[frame_idx])

    target_toe_offset = a - t  # difference between unmodified toe and ankle at the frame
    ca = ct + target_toe_offset  # move ankle so toe is on the ground
    return MotionGroundingConstraint(frame_idx, ankle_joint_name, ca, None, None)


def blend_between_frames(skeleton, frames, start, end, joint_list, window):
    for joint in joint_list:
        idx = skeleton.animated_joints.index(joint) * 4 + 3
        j_indices = [idx, idx + 1, idx + 2, idx + 3]
        start_q = frames[start][j_indices]
        end_q = frames[end][j_indices]
        print joint, window
        for i in xrange(window):
            t = float(i) / window
            slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
            frames[start + i][j_indices] = slerp_q
            #print "blend at frame", start + i, slerp_q


def apply_constraint(skeleton, frames, frame_idx, c, blend_start, blend_end, blend_window=5):
    ik_chain = skeleton.skeleton_model["ik_chains"][c.joint_name]
    ik = AnalyticalLimbIK.init_from_dict(skeleton, c.joint_name, ik_chain)
    #print "b",c.joint_name,frame_idx,c.position, skeleton.nodes[c.joint_name].get_global_position(frames[frame_idx])
    frames[frame_idx] = ik.apply2(frames[frame_idx], c.position, c.orientation)
    #print "a",c.joint_name,frame_idx,c.position, skeleton.nodes[c.joint_name].get_global_position(frames[frame_idx])
    joint_list = [ik_chain["root"], ik_chain["joint"], c.joint_name]
    if blend_start < blend_end:
        blend_between_frames(skeleton, frames, blend_start, blend_end, joint_list, blend_window)


def move_to_ground(skeleton,frames, foot_joints, target_ground_height,start_frame=0, n_frames=5):
    source_ground_height = guess_ground_height(skeleton, frames,start_frame, n_frames, foot_joints)
    for f in frames:
        f[1] += target_ground_height - source_ground_height




def ground_both_feet(skeleton, frames, target_height, frame_idx):
    constraints = []
    stance_foot = skeleton.skeleton_model["right_foot"]
    heel_joint = skeleton.skeleton_model["right_heel"]
    toe_joint = skeleton.skeleton_model["right_toe"]
    heel_offset = skeleton.skeleton_model["heel_offset"]
    c1 = create_ankle_constraint_from_toe_and_heel(skeleton, frames, frame_idx, stance_foot, heel_joint, toe_joint, heel_offset,
                           target_height)
    constraints.append(c1)

    stance_foot = skeleton.skeleton_model["left_foot"]
    toe_joint = skeleton.skeleton_model["left_toe"]
    heel_joint = skeleton.skeleton_model["left_heel"]
    c2 = create_ankle_constraint_from_toe_and_heel(skeleton, frames, frame_idx, stance_foot, heel_joint, toe_joint, heel_offset, target_height)
    constraints.append(c2)
    return constraints


def ground_right_stance(skeleton, frames, target_height, frame_idx):
    constraints = []
    stance_foot = skeleton.skeleton_model["right_foot"]
    heel_joint = skeleton.skeleton_model["right_heel"]
    toe_joint = skeleton.skeleton_model["right_toe"]
    heel_offset = skeleton.skeleton_model["heel_offset"]
    c1 = create_ankle_constraint_from_toe_and_heel(skeleton, frames, frame_idx, stance_foot, heel_joint, toe_joint, heel_offset,
                           target_height)
    constraints.append(c1)

    swing_foot = skeleton.skeleton_model["left_foot"]
    toe_joint = skeleton.skeleton_model["left_toe"]
    heel_joint = skeleton.skeleton_model["left_heel"]
    c2 = generate_ankle_constraint_from_toe(skeleton, frames, frame_idx, swing_foot, heel_joint, toe_joint, target_height)
    constraints.append(c2)
    return constraints


def ground_left_stance(skeleton, frames, target_height, frame_idx):
    constraints = []
    stance_foot = skeleton.skeleton_model["left_foot"]
    heel_joint = skeleton.skeleton_model["left_heel"]
    toe_joint = skeleton.skeleton_model["left_toe"]
    heel_offset = skeleton.skeleton_model["heel_offset"]
    c1 = create_ankle_constraint_from_toe_and_heel(skeleton, frames, frame_idx, stance_foot, heel_joint, toe_joint, heel_offset, target_height)
    constraints.append(c1)

    swing_foot = skeleton.skeleton_model["right_foot"]
    toe_joint = skeleton.skeleton_model["right_toe"]
    heel_joint = skeleton.skeleton_model["right_heel"]
    c2 = generate_ankle_constraint_from_toe(skeleton, frames, frame_idx, swing_foot,heel_joint, toe_joint, target_height)
    constraints.append(c2)
    return constraints


def ground_first_frame(skeleton, frames, target_height, window_size, stance_foot="right"):
    first_frame = 0
    if stance_foot == "both":
        constraints = ground_both_feet(skeleton, frames, target_height, first_frame)
    elif stance_foot == "right":
        constraints = ground_right_stance(skeleton, frames, target_height, first_frame)
    else:
        constraints = ground_left_stance(skeleton, frames, target_height, first_frame)

    c1 = constraints[0]
    c2 = constraints[1]
    root_pos = generate_root_constraint_for_two_feet(skeleton, frames[first_frame], c1, c2)
    if root_pos is not None:
        frames[first_frame][:3] = root_pos
        print "change root at frame", first_frame
        smooth_root_translation_at_start(frames, first_frame, window_size)
    for c in constraints:
        apply_constraint(skeleton, frames, first_frame, c, first_frame, first_frame + window_size, window_size)


def ground_last_frame(skeleton, frames, target_height, window_size, stance_foot="left"):
    last_frame = len(frames) - 1
    if stance_foot == "both":
        constraints = ground_both_feet(skeleton, frames, target_height, last_frame)
    elif stance_foot == "left":
        constraints = ground_left_stance(skeleton, frames, target_height, last_frame)
    else:
        constraints = ground_right_stance(skeleton, frames, target_height, last_frame)

    c1 = constraints[0]
    c2 = constraints[1]
    root_pos = generate_root_constraint_for_two_feet(skeleton, frames[last_frame], c1, c2)
    if root_pos is not None:
        frames[last_frame][:3] = root_pos
        print "change root at frame", last_frame
        smooth_root_translation_at_end(frames, last_frame, window_size)
    for c in constraints:
        apply_constraint(skeleton, frames, last_frame, c, last_frame - window_size, last_frame, window_size)


def ground_initial_stance_foot_unconstrained(skeleton, frames, target_height, stance_foot="right", mode="toe"):
    foot_joint = skeleton.skeleton_model[stance_foot+"_foot"]
    toe_joint = skeleton.skeleton_model[stance_foot+"_toe"]
    heel_joint = skeleton.skeleton_model[stance_foot+"_heel"]
    heel_offset = skeleton.skeleton_model["heel_offset"]

    toe_pos = None
    heel_pos = None
    for frame_idx in xrange(0, len(frames)):
        if toe_pos is None:
            toe_pos = skeleton.nodes[toe_joint].get_global_position(frames[frame_idx])
            toe_pos[1] = target_height
            heel_pos = skeleton.nodes[heel_joint].get_global_position(frames[frame_idx])
            heel_pos[1] = target_height
        if mode == "toe":
            c = generate_ankle_constraint_from_toe(skeleton, frames, frame_idx, foot_joint, toe_joint, target_height, toe_pos)
        else:
            c = create_ankle_constraint_from_toe_and_heel(skeleton, frames, frame_idx, foot_joint, heel_joint, toe_joint, heel_offset, target_height, heel_pos, toe_pos)
        root_pos = generate_root_constraint_for_one_foot(skeleton, frames[frame_idx], c)
        if root_pos is not None:
            frames[frame_idx][:3] = root_pos
        apply_constraint(skeleton, frames, frame_idx, c, frame_idx, frame_idx)

def ground_initial_stance_foot(skeleton, frames, target_height, stance_foot="right", swing_foot="left", stance_mode="toe"):
    stance_foot_joint = skeleton.skeleton_model[stance_foot + "_foot"]
    stance_toe_joint = skeleton.skeleton_model[stance_foot + "_toe"]
    stance_heel_joint = skeleton.skeleton_model[stance_foot + "_heel"]
    swing_foot_joint = skeleton.skeleton_model[swing_foot + "_foot"]
    swing_toe_joint = skeleton.skeleton_model[swing_foot + "_toe"]
    swing_heel_joint = skeleton.skeleton_model[swing_foot + "_heel"]
    heel_offset = skeleton.skeleton_model["heel_offset"]

    stance_toe_pos = None
    stance_heel_pos = None
    for frame_idx in xrange(0, len(frames)):
        if stance_toe_pos is None:
            stance_toe_pos = skeleton.nodes[stance_toe_joint].get_global_position(frames[frame_idx])
            stance_toe_pos[1] = target_height
            stance_heel_pos = skeleton.nodes[stance_heel_joint].get_global_position(frames[frame_idx])
            stance_heel_pos[1] = target_height
        if stance_mode == "toe":
            stance_c = generate_ankle_constraint_from_toe(skeleton, frames, frame_idx, stance_foot_joint, stance_heel_joint, stance_toe_joint,
                                                   target_height, stance_toe_pos)

            #stance_heel_pos = skeleton.nodes[stance_heel_joint].get_global_position(frames[frame_idx])
            #stance_c = create_ankle_constraint_from_toe_and_heel(skeleton, frames, frame_idx, stance_foot_joint, stance_heel_joint,
            #                             stance_toe_joint, heel_offset,
            #                             target_height, stance_heel_pos, stance_toe_pos)
            print "toe",stance_toe_pos
        else:
            stance_c = create_ankle_constraint_from_toe_and_heel(skeleton, frames, frame_idx, stance_foot_joint, stance_heel_joint, stance_toe_joint, heel_offset,
                                  target_height, stance_heel_pos, stance_toe_pos)
            
        swing_heel_pos = skeleton.nodes[swing_heel_joint].get_global_position(frames[frame_idx])
        swing_toe_pos = skeleton.nodes[swing_toe_joint].get_global_position(frames[frame_idx])
        swing_c = create_ankle_constraint_from_toe_and_heel(skeleton, frames, frame_idx, swing_foot_joint, swing_heel_joint, swing_toe_joint, heel_offset,
                                     target_height, swing_heel_pos, swing_toe_pos)

        #print "swing_c",swing_c.position,swing_heel_pos,swing_toe_pos
        #root_pos = generate_root_constraint_for_two_feet(skeleton, frames[frame_idx], stance_c, swing_c)
        root_pos = generate_root_constraint_for_one_foot(skeleton, frames[frame_idx], stance_c)
        if root_pos is not None:
            frames[frame_idx][:3] = root_pos
        #print "toe pos before ", frame_idx, skeleton.nodes[stance_toe_joint].get_global_position(frames[frame_idx])

        apply_constraint(skeleton, frames, frame_idx, stance_c, frame_idx, frame_idx)
        apply_constraint(skeleton, frames, frame_idx, swing_c, frame_idx, frame_idx)
        #print "toe pos after ",frame_idx, skeleton.nodes[stance_toe_joint].get_global_position(frames[frame_idx])


def get_files(path, max_number, suffix="bvh"):
    count = 0
    for root, dirs, files in os.walk(path):
        for file_name in files:
            if file_name.endswith(suffix):
                yield path + os.sep + file_name
                count += 1
                if count >= max_number:
                    return


def align_xz_to_origin(skeleton, frames):
    root = skeleton.aligning_root_node
    tframe = frames[0]
    offset = np.array([0, 0, 0]) - skeleton.nodes[root].get_global_position(tframe)
    for frame_idx in xrange(0, len(frames)):
        frames[frame_idx, 0] += offset[0]
        frames[frame_idx, 2] += offset[2]


def run_grounding_on_bvh_file(bvh_file, out_path, skeleton_model, configuration):
    print "apply on", bvh_file
    bvh = BVHReader(bvh_file)
    animated_joints = list(bvh.get_animated_joints())
    skeleton = SkeletonBuilder().load_from_bvh(bvh, animated_joints)  # filter here
    skeleton.aligning_root_node = "pelvis"
    skeleton.skeleton_model = skeleton_model
    mv = MotionVector()
    mv.from_bvh_reader(bvh)
    skeleton = add_heels_to_skeleton(skeleton, skeleton_model["left_foot"],
                                     skeleton_model["right_foot"],
                                     skeleton_model["left_heel"],
                                     skeleton_model["right_heel"],
                                     skeleton_model["heel_offset"])

    target_height = 0
    foot_joints = skeleton.skeleton_model["foot_joints"]
    search_window_start = int(len(mv.frames)/2)
    start_stance_foot = configuration["start_stance_foot"]
    stance_foot = configuration["stance_foot"]
    swing_foot = configuration["swing_foot"]
    end_stance_foot = configuration["end_stance_foot"]
    stance_mode = configuration["stance_mode"]
    start_window_size = configuration["start_window_size"]
    end_window_size = configuration["end_window_size"]
    move_to_ground(skeleton, mv.frames, foot_joints, target_height, search_window_start, len(mv.frames)-search_window_start)  #20 45
    ground_first_frame(skeleton, mv.frames, target_height, start_window_size, start_stance_foot)

    ground_last_frame(skeleton, mv.frames, target_height, end_window_size, end_stance_foot)
    if stance_mode is not "none":
        ground_initial_stance_foot(skeleton, mv.frames, target_height, stance_foot, swing_foot, stance_mode)
        # ground_initial_stance_foot_unconstrained(skeleton, mv.frames, target_height, stance_foot, stance_mode)

    align_xz_to_origin(skeleton, mv.frames)
    file_name = bvh_file.split("\\")[-1][:-4]
    out_filename = file_name + "_grounded"
    mv.export(skeleton, out_path, out_filename, add_time_stamp=False)


def run_motion_grounding(in_path, out_path, skeleton_type, configuration, max_number=100):
    bvh_files = list(get_files(in_path, max_number, "bvh"))
    for bvh_file in bvh_files:
        run_grounding_on_bvh_file(bvh_file, out_path, skeleton_type, configuration)

configurations = collections.OrderedDict()

configurations["leftStance"] = dict()
configurations["leftStance"]["start_stance_foot"] = "right"
configurations["leftStance"]["stance_foot"] = "right"
configurations["leftStance"]["swing_foot"] = "left"
configurations["leftStance"]["end_stance_foot"] = "left"
configurations["leftStance"]["stance_mode"] = "toe"
configurations["leftStance"]["start_window_size"] = 10
configurations["leftStance"]["end_window_size"] = 10

configurations["rightStance"] = dict()
configurations["rightStance"]["start_stance_foot"] = "left"
configurations["rightStance"]["stance_foot"] = "left"
configurations["rightStance"]["swing_foot"] = "right"
configurations["rightStance"]["end_stance_foot"] = "right"
configurations["rightStance"]["stance_mode"] = "toe"
configurations["rightStance"]["start_window_size"] = 10
configurations["rightStance"]["end_window_size"] = 10

configurations["beginLeftStance"] = dict()
configurations["beginLeftStance"]["start_stance_foot"] = "both"
configurations["beginLeftStance"]["stance_foot"] = "right"
configurations["beginLeftStance"]["swing_foot"] = "left"
configurations["beginLeftStance"]["end_stance_foot"] = "left"
configurations["beginLeftStance"]["stance_mode"] = "toe"
configurations["beginLeftStance"]["start_window_size"] = 10
configurations["beginLeftStance"]["end_window_size"] = 10

configurations["beginRightStance"] = dict()
configurations["beginRightStance"]["start_stance_foot"] = "both"
configurations["beginRightStance"]["stance_foot"] = "left"
configurations["beginRightStance"]["swing_foot"] = "right"
configurations["beginRightStance"]["end_stance_foot"] = "right"
configurations["beginRightStance"]["stance_mode"] = "toe"
configurations["beginRightStance"]["start_window_size"] = 10
configurations["beginRightStance"]["end_window_size"] = 10

configurations["endRightStance"] = dict()
configurations["endRightStance"]["start_stance_foot"] = "left"
configurations["endRightStance"]["stance_foot"] = "left"
configurations["endRightStance"]["swing_foot"] = "right"
configurations["endRightStance"]["end_stance_foot"] = "both"
configurations["endRightStance"]["swing_foot"] = "right"
configurations["endRightStance"]["stance_mode"] = "none"
configurations["endRightStance"]["start_window_size"] = 10
configurations["endRightStance"]["end_window_size"] = 10

configurations["endLeftStance"] = dict()
configurations["endLeftStance"]["start_stance_foot"] = "right"
configurations["endLeftStance"]["stance_foot"] = "right"
configurations["endLeftStance"]["swing_foot"] = "left"
configurations["endLeftStance"]["end_stance_foot"] = "both"
configurations["endLeftStance"]["swing_foot"] = "left"
configurations["endLeftStance"]["stance_mode"] = "none"
configurations["endLeftStance"]["start_window_size"] = 10
configurations["endLeftStance"]["end_window_size"] = 10


configurations["turnLeftRightStance"] = dict()
configurations["turnLeftRightStance"]["start_stance_foot"] = "both"
configurations["turnLeftRightStance"]["stance_foot"] = "left"
configurations["turnLeftRightStance"]["swing_foot"] = "right"
configurations["turnLeftRightStance"]["end_stance_foot"] = "right"
configurations["turnLeftRightStance"]["stance_mode"] = "none"
configurations["turnLeftRightStance"]["start_window_size"] = 20
configurations["turnLeftRightStance"]["end_window_size"] = 20

configurations["turnRightLeftStance"] = dict()
configurations["turnRightLeftStance"]["start_stance_foot"] = "both"
configurations["turnRightLeftStance"]["stance_foot"] = "right"
configurations["turnRightLeftStance"]["swing_foot"] = "left"
configurations["turnRightLeftStance"]["end_stance_foot"] = "left"
configurations["turnRightLeftStance"]["stance_mode"] = "none"
configurations["turnRightLeftStance"]["start_window_size"] = 20
configurations["turnRightLeftStance"]["end_window_size"] = 20


if __name__ == "__main__":
    bvh_file = "skeleton.bvh"
    bvh_file = "walk_001_1.bvh"
    #bvh_file = "walk_014_2.bvh"
    bvh_file = "game_engine_left_stance.bvh"
    skeleton_type = "game_engine"
    max_number = 10000
    skeleton_model = GAME_ENGINE_SKELETON_MODEL
    for step_type in configurations.keys():
        ea_path = "E:\\projects\\INTERACT\\data\\1 - MoCap\\4 - Alignment\\elementary_action_walk"
        in_path = ea_path + "\\" + step_type + "_game_engine_skeleton_new"
        out_path = ea_path + "\\" + step_type + "_game_engine_skeleton_new_grounded"#"out\\foot_sliding"
        config = configurations[step_type]
        #run_grounding_on_bvh_file(bvh_file, "out//foot_sliding", skeleton_type, configuration)
        run_motion_grounding(in_path, out_path, skeleton_model, config, max_number)
