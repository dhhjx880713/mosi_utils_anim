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
from morphablegraphs.animation_data.motion_concatenation import blend_between_frames

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


def generate_ankle_constraint_from_toe(skeleton, frames, frame_idx, ankle_joint_name, toe_joint_name, target_ground_height):
    """ create a constraint on the ankle position based on the toe constraint position"""
    #print "add toe constraint"
    ct = skeleton.nodes[toe_joint_name].get_global_position(frames[frame_idx])
    ct[1] = target_ground_height  # set toe constraint on the ground
    a = skeleton.nodes[ankle_joint_name].get_global_position(frames[frame_idx])
    t = skeleton.nodes[toe_joint_name].get_global_position(frames[frame_idx])

    target_toe_offset = a - t  # difference between unmodified toe and ankle at the frame
    ca = ct + target_toe_offset  # move ankle so toe is on the ground
    return MotionGroundingConstraint(frame_idx, ankle_joint_name, ca, None, None)


def get_limb_length(skeleton, joint_name):
    limb_length = np.linalg.norm(skeleton.nodes[joint_name].offset)
    limb_length += np.linalg.norm(skeleton.nodes[joint_name].parent.offset)
    return limb_length


def global_position_to_root_translation(skeleton, frame, joint_name, p):
    """ determine necessary root translation to achieve a global position"""

    tframe = np.array(frame)
    tframe[:3] = [0,0,0]
    parent_joint = skeleton.nodes[joint_name].parent.node_name
    parent_m = skeleton.nodes[parent_joint].get_global_matrix(tframe, use_cache=False)
    old_global = np.dot(parent_m, skeleton.nodes[joint_name].get_local_matrix(tframe))
    #new_global = np.array(old_global)#np.dot(delta_m, old_global)
    #new_global[:3, 3] = p
    #new_local = np.dot(np.linalg.inv(parent_m), new_global)
    #new_local[:3, 3]  # use the translation
    return p - old_global[:3,3]


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


def apply_constraint(skeleton, frames, frame_idx, c, blend_start, blend_end, blend_window=5):
    ik_chain = skeleton.annotation["ik_chains"][c.joint_name]
    ik = AnalyticalLimbIK.init_from_dict(skeleton, c.joint_name, ik_chain)
    print "b",c.joint_name,frame_idx,skeleton.nodes[c.joint_name].get_global_position(frames[frame_idx])
    frames[frame_idx] = ik.apply2(frames[frame_idx], c.position, c.orientation)
    print "a",c.joint_name,frame_idx,skeleton.nodes[c.joint_name].get_global_position(frames[frame_idx])
    joint_list = [ik_chain["root"], ik_chain["joint"], c.joint_name]
    blend_between_frames(skeleton, frames, blend_start, blend_end, joint_list, blend_window)


def move_to_ground(skeleton,frames, foot_joints, target_ground_height):
    source_ground_height = guess_ground_height(skeleton, frames, 5, foot_joints)
    for f in frames:
        f[1] += target_ground_height - source_ground_height




def ground_first_frame(skeleton, frames, target_height, window_size):
    first_frame = 0
    constraints = []
    stance_foot = skeleton.annotation["right_foot"]
    heel_joint = skeleton.annotation["right_heel"]
    toe_joint = skeleton.annotation["right_toe"]
    heel_offset = skeleton.annotation["heel_offset"]
    c1 = create_constraint(skeleton, frames, first_frame, stance_foot, heel_joint, toe_joint, heel_offset, target_height)
    constraints.append(c1)

    swing_foot = skeleton.annotation["left_foot"]
    toe_joint = skeleton.annotation["left_toe"]
    c2 = generate_ankle_constraint_from_toe(skeleton, frames, first_frame, swing_foot, toe_joint, target_height)
    constraints.append(c2)
    root_pos = generate_root_constraint_for_two_feet(skeleton, frames[first_frame], c1, c2)
    if root_pos is not None:
        frames[first_frame][:3] = root_pos
        print "change root at frame", first_frame
    for c in constraints:
        apply_constraint(skeleton, frames, first_frame, c, first_frame, first_frame + window_size)


def ground_last_frame(skeleton, frames, target_height, window_size):
    last_frame = len(frames) - 1
    constraints = []
    swing_foot = skeleton.annotation["left_foot"]
    heel_joint = skeleton.annotation["left_heel"]
    toe_joint = skeleton.annotation["left_toe"]
    heel_offset = skeleton.annotation["heel_offset"]
    c1 = create_constraint(skeleton, frames, last_frame, swing_foot, heel_joint, toe_joint, heel_offset, target_height)
    constraints.append(c1)

    stance_foot = skeleton.annotation["right_foot"]
    toe_joint = skeleton.annotation["right_toe"]
    c2 = generate_ankle_constraint_from_toe(skeleton, frames, last_frame, stance_foot, toe_joint, target_height)
    constraints.append(c2)
    root_pos = generate_root_constraint_for_two_feet(skeleton, frames[last_frame], c1, c2)
    if root_pos is not None:
        frames[last_frame][:3] = root_pos
        print "change root at frame", last_frame

    for c in constraints:
        apply_constraint(skeleton, frames, last_frame, c, last_frame - window_size, last_frame)


def run_motion_grounding(bvh_file, skeleton_type):
    annotation = SKELETON_ANNOTATIONS[skeleton_type]
    bvh = BVHReader(bvh_file)
    animated_joints = list(bvh.get_animated_joints())
    skeleton = Skeleton()
    skeleton.load_from_bvh(bvh, animated_joints) # filter here
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
    move_to_ground(skeleton, mv.frames, foot_joints, target_height)
    window_size = 5
    ground_first_frame(skeleton, mv.frames, target_height, window_size)
    ground_last_frame(skeleton, mv.frames, target_height, window_size)


    mv.export(skeleton, "out\\foot_sliding", "out")

if __name__ == "__main__":
    bvh_file = "skeleton.bvh"
    bvh_file = "walk_001_1.bvh"
    #bvh_file = "walk_014_2.bvh"
    bvh_file = "game_engine_left_stance_2.bvh"
    skeleton_type = "game_engine"
    run_motion_grounding(bvh_file, skeleton_type)
