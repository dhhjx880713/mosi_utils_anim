import os
from morphablegraphs.animation_data import BVHReader, Skeleton, MotionVector
from morphablegraphs.motion_generator.algorithm_configuration import AlgorithmConfigurationBuilder
from morphablegraphs.animation_data.motion_editing import FootplantConstraintGenerator
from morphablegraphs.animation_data.motion_editing import MotionGrounding
from morphablegraphs.animation_data.motion_editing.constants import *
from morphablegraphs.animation_data.motion_editing.motion_grounding import IKConstraintSet
from morphablegraphs.animation_data.motion_editing.utils import get_average_joint_position, get_average_joint_direction, plot_joint_heights, add_heels_to_skeleton, get_joint_height, \
    save_ground_contact_annotation
from morphablegraphs.animation_data.motion_editing.motion_grounding import create_grounding_constraint_from_frame, MotionGroundingConstraint
from morphablegraphs.animation_data.motion_editing.utils import guess_ground_height, normalize, quaternion_from_vector_to_vector, project_on_intersection_circle
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


def create_constraint(skeleton, frames, frame_idx, ankle_joint, heel_joint, toe_joint,heel_offset, target_ground_height):
    ct = skeleton.nodes[toe_joint].get_global_position(frames[frame_idx])
    ch = skeleton.nodes[heel_joint].get_global_position(frames[frame_idx])
    ct[1] = target_ground_height

    ch[1] = target_ground_height

    target_direction = normalize(ct - ch)

    t = skeleton.nodes[toe_joint].get_global_position(frames[frame_idx])
    h = skeleton.nodes[heel_joint].get_global_position(frames[frame_idx])
    original_direction = normalize(t - h)

    global_delta_q = quaternion_from_vector_to_vector(original_direction, target_direction)
    global_delta_q = normalize(global_delta_q)

    m = skeleton.nodes[heel_joint].get_global_matrix(frames[frame_idx])
    m[:3, 3] = [0, 0, 0]
    oq = quaternion_from_matrix(m)
    oq = normalize(oq)
    orientation = normalize(quaternion_multiply(global_delta_q, oq))

    # set target ankle position based on the  grounded heel and the global target orientation of the ankle
    m = quaternion_matrix(orientation)[:3, :3]
    target_heel_offset = np.dot(m, heel_offset)
    ca = ch - target_heel_offset
    print "set ankle constraint both", ch, ca, target_heel_offset
    return MotionGroundingConstraint(frame_idx, ankle_joint, ca, None, orientation)


def generate_ankle_constraint_from_toe(skeleton, frames, frame_idx, ankle_joint_name, toe_joint_name, target_ground_height, toe_pos = None):
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


def get_limb_length(skeleton, joint_name, offset=1):
    limb_length = np.linalg.norm(skeleton.nodes[joint_name].offset)
    limb_length += np.linalg.norm(skeleton.nodes[joint_name].parent.offset)
    return limb_length + offset


def global_position_to_root_translation(skeleton, frame, joint_name, p):
    """ determine necessary root translation to achieve a global position"""

    tframe = np.array(frame)
    tframe[:3] = [0,0,0]
    parent_joint = skeleton.nodes[joint_name].parent.node_name
    parent_m = skeleton.nodes[parent_joint].get_global_matrix(tframe, use_cache=False)
    old_global = np.dot(parent_m, skeleton.nodes[joint_name].get_local_matrix(tframe))
    return p - old_global[:3,3]


def generate_root_constraint_for_one_foot(skeleton, frame, c):
    root = skeleton.aligning_root_node
    root_pos = skeleton.nodes[root].get_global_position(frame)
    target_length = np.linalg.norm(c.position - root_pos)
    limb_length = get_limb_length(skeleton, c.joint_name)
    if target_length >= limb_length:
        new_root_pos = (c.position + normalize(root_pos - c.position) * limb_length)
        print "one constraint on ", c.joint_name, "- before", root_pos, "after", new_root_pos
        return global_position_to_root_translation(skeleton, frame, root, new_root_pos)

    else:
        print "no change"


def generate_root_constraint_for_two_feet(skeleton, frame, constraint1, constraint2):
    """ Set the root position to the projection on the intersection of two spheres """
    root = skeleton.aligning_root_node
    # root = self.skeleton.root
    p = skeleton.nodes[root].get_global_position(frame)
    offset = skeleton.nodes[root].get_global_position(skeleton.identity_frame)#[0, skeleton.nodes[root].offset[0], -skeleton.nodes[root].offset[1]]
    print p, offset

    t1 = np.linalg.norm(constraint1.position - p)
    t2 = np.linalg.norm(constraint2.position - p)

    c1 = constraint1.position
    r1 = get_limb_length(skeleton, constraint1.joint_name)
    # p1 = c1 + r1 * normalize(p-c1)
    c2 = constraint2.position
    r2 = get_limb_length(skeleton, constraint2.joint_name)
    # p2 = c2 + r2 * normalize(p-c2)
    if r1 > t1 and r2 > t2:
        print "no root constraint", t1,t2, r1, r2
        return None
    print "adapt root for two constraints", t1, t2,  r1, r2

    p_c = project_on_intersection_circle(p, c1, r1, c2, r2)
    p = global_position_to_root_translation(skeleton, frame, root, p_c)
    tframe = np.array(frame)
    tframe[:3] = p
    new_p = skeleton.nodes[root].get_global_position(tframe)
    print "compare",p_c, new_p
    return p#p_c - offset


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
            print "blend at frame", start + i, slerp_q

def apply_constraint(skeleton, frames, frame_idx, c, blend_start, blend_end, blend_window=5):
    ik_chain = skeleton.annotation["ik_chains"][c.joint_name]
    ik = AnalyticalLimbIK.init_from_dict(skeleton, c.joint_name, ik_chain)
    print "b",c.joint_name,frame_idx,skeleton.nodes[c.joint_name].get_global_position(frames[frame_idx])
    frames[frame_idx] = ik.apply2(frames[frame_idx], c.position, c.orientation)
    print "a",c.joint_name,frame_idx,skeleton.nodes[c.joint_name].get_global_position(frames[frame_idx])
    joint_list = [ik_chain["root"], ik_chain["joint"], c.joint_name]
    if blend_start < blend_end:
        blend_between_frames(skeleton, frames, blend_start, blend_end, joint_list, blend_window)


def move_to_ground(skeleton,frames, foot_joints, target_ground_height,start_frame=0, n_frames=5):
    source_ground_height = guess_ground_height(skeleton, frames,start_frame, n_frames, foot_joints)
    for f in frames:
        f[1] += target_ground_height - source_ground_height


def smooth_root_translation_at_start(frames, d, window):
    start = frames[d, :3]
    start_idx = d+window
    end = frames[start_idx, :3]
    for i in xrange(window):
        t = float(i) / (window)
        frames[d + i, :3] = start * (1 - t) + end * t


def smooth_root_translation_at_end(frames, d, window):
    root_pos = frames[d, :3]
    start_idx = d-window
    start = frames[start_idx, :3]
    end = root_pos
    for i in xrange(window):
        t = float(i) / (window)
        frames[start_idx + i, :3] = start * (1 - t) + end * t


def ground_both_feet(skeleton, frames, target_height, frame_idx):
    constraints = []
    stance_foot = skeleton.annotation["right_foot"]
    heel_joint = skeleton.annotation["right_heel"]
    toe_joint = skeleton.annotation["right_toe"]
    heel_offset = skeleton.annotation["heel_offset"]
    c1 = create_constraint(skeleton, frames, frame_idx, stance_foot, heel_joint, toe_joint, heel_offset,
                           target_height)
    constraints.append(c1)

    stance_foot = skeleton.annotation["left_foot"]
    toe_joint = skeleton.annotation["left_toe"]
    heel_joint = skeleton.annotation["left_heel"]
    c2 = create_constraint(skeleton, frames, frame_idx, stance_foot, heel_joint, toe_joint, heel_offset, target_height)
    constraints.append(c2)
    return constraints


def ground_right_stance(skeleton, frames, target_height, frame_idx):
    constraints = []
    stance_foot = skeleton.annotation["right_foot"]
    heel_joint = skeleton.annotation["right_heel"]
    toe_joint = skeleton.annotation["right_toe"]
    heel_offset = skeleton.annotation["heel_offset"]
    c1 = create_constraint(skeleton, frames, frame_idx, stance_foot, heel_joint, toe_joint, heel_offset,
                           target_height)
    constraints.append(c1)

    swing_foot = skeleton.annotation["left_foot"]
    toe_joint = skeleton.annotation["left_toe"]
    c2 = generate_ankle_constraint_from_toe(skeleton, frames, frame_idx, swing_foot, toe_joint, target_height)
    constraints.append(c2)
    return constraints


def ground_left_stance(skeleton, frames, target_height, frame_idx):
    constraints = []
    stance_foot = skeleton.annotation["left_foot"]
    heel_joint = skeleton.annotation["left_heel"]
    toe_joint = skeleton.annotation["left_toe"]
    heel_offset = skeleton.annotation["heel_offset"]
    c1 = create_constraint(skeleton, frames, frame_idx, stance_foot, heel_joint, toe_joint, heel_offset, target_height)
    constraints.append(c1)

    swing_foot = skeleton.annotation["right_foot"]
    toe_joint = skeleton.annotation["right_toe"]
    c2 = generate_ankle_constraint_from_toe(skeleton, frames, frame_idx, swing_foot, toe_joint, target_height)
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


def ground_initial_stance_foot(skeleton, frames, target_height, stance_foot="right"):
    foot_joint = skeleton.annotation[stance_foot+"_foot"]
    toe_joint = skeleton.annotation[stance_foot+"_toe"]
    toe_pos = None
    for frame_idx in xrange(0, len(frames)):
        if toe_pos is None:
            toe_pos = skeleton.nodes[toe_joint].get_global_position(frames[frame_idx])
            toe_pos[1] = target_height
        c = generate_ankle_constraint_from_toe(skeleton, frames, frame_idx, foot_joint, toe_joint, target_height, toe_pos)
        root_pos = generate_root_constraint_for_one_foot(skeleton, frames[frame_idx], c)
        if root_pos is not None:
            frames[frame_idx][:3] = root_pos
        apply_constraint(skeleton, frames, frame_idx, c, frame_idx, frame_idx)


def get_files(path, max_number, suffix="bvh"):
    count = 0
    for root, dirs, files in os.walk(path):
        for file_name in files:
            if file_name.endswith(suffix):
                yield path + os.sep + file_name
                count += 1
                if count >= max_number:
                    return


def run_grounding_on_bvh_file(bvh_file, out_path, skeleton_type, configuration):
    print "apply on", bvh_file
    annotation = SKELETON_ANNOTATIONS[skeleton_type]
    bvh = BVHReader(bvh_file)
    animated_joints = list(bvh.get_animated_joints())
    skeleton = Skeleton()
    skeleton.load_from_bvh(bvh, animated_joints)  # filter here
    skeleton.aligning_root_node = "pelvis"
    skeleton.annotation = annotation
    mv = MotionVector()
    mv.from_bvh_reader(bvh)
    skeleton = add_heels_to_skeleton(skeleton, annotation["left_foot"],
                                     annotation["right_foot"],
                                     annotation["left_heel"],
                                     annotation["right_heel"],
                                     annotation["heel_offset"])

    target_height = 0
    foot_joints = skeleton.annotation["foot_joints"]
    search_window_start = int(len(mv.frames)/2)
    window_size = 10
    start_stance_foot = configuration["start_stance_foot"]
    stance_foot = configuration["stance_foot"]
    end_stance_foot = configuration["end_stance_foot"]
    move_to_ground(skeleton, mv.frames, foot_joints, target_height, search_window_start, window_size)  #20 45
    ground_first_frame(skeleton, mv.frames, target_height, window_size, start_stance_foot)
    ground_initial_stance_foot(skeleton, mv.frames, target_height, stance_foot)
    ground_last_frame(skeleton, mv.frames, target_height, window_size, end_stance_foot)
    file_name = bvh_file.split("\\")[-1][:-4]
    out_filename = file_name + "_grounded"
    mv.export(skeleton, out_path, out_filename, add_time_stamp=False)


def run_motion_grounding(in_path, out_path, skeleton_type, configuration, max_number=100):
    bvh_files = list(get_files(in_path, max_number, "bvh"))
    for bvh_file in bvh_files:
        run_grounding_on_bvh_file(bvh_file, out_path, skeleton_type, configuration)

configuration = dict()
configuration["leftStance"] = dict()
configuration["leftStance"]["start_stance_foot"] = "right"
configuration["leftStance"]["stance_foot"] = "right"
configuration["leftStance"]["end_stance_foot"] = "left"
configuration["rightStance"] = dict()
configuration["rightStance"]["start_stance_foot"] = "left"
configuration["rightStance"]["stance_foot"] = "left"
configuration["rightStance"]["end_stance_foot"] = "right"
configuration["beginLeftStance"] = dict()
configuration["beginLeftStance"]["start_stance_foot"] = "both"
configuration["beginLeftStance"]["stance_foot"] = "right"
configuration["beginLeftStance"]["end_stance_foot"] = "left"
configuration["beginRightStance"] = dict()
configuration["beginRightStance"]["start_stance_foot"] = "both"
configuration["beginRightStance"]["stance_foot"] = "left"
configuration["beginRightStance"]["end_stance_foot"] = "right"
configuration["endRightStance"] = dict()
configuration["endRightStance"]["start_stance_foot"] = "left"
configuration["endRightStance"]["stance_foot"] = "left"
configuration["endRightStance"]["end_stance_foot"] = "both"
configuration["endLeftStance"] = dict()
configuration["endLeftStance"]["start_stance_foot"] = "right"
configuration["endLeftStance"]["stance_foot"] = "right"
configuration["endLeftStance"]["end_stance_foot"] = "both"

if __name__ == "__main__":
    bvh_file = "skeleton.bvh"
    bvh_file = "walk_001_1.bvh"
    #bvh_file = "walk_014_2.bvh"
    bvh_file = "game_engine_left_stance.bvh"
    skeleton_type = "game_engine"
    step_type = "beginLeftStance"
    ea_path = "E:\\projects\\INTERACT\\data\\1 - MoCap\\4 - Alignment\\elementary_action_walk"
    in_path = ea_path+"\\"+step_type+"_game_engine_skeleton_new"
    out_path = ea_path+"\\" +step_type+"_game_engine_skeleton_new_grounded"#"out\\foot_sliding"
    configuration = configuration[step_type]
    #run_grounding_on_bvh_file(bvh_file, "out//foot_sliding", skeleton_type, configuration)
    max_number = 10
    run_motion_grounding(in_path, out_path, skeleton_type, configuration, max_number)
