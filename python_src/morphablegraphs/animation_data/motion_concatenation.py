import numpy as np
from .utils import euler_substraction, point_to_euler_angle, euler_to_quaternion, euler_angles_to_rotation_matrix, get_rotation_angle, DEFAULT_ROTATION_ORDER, LEN_QUAT, LEN_EULER, LEN_ROOT_POS
from ..external.transformations import quaternion_matrix, quaternion_about_axis, quaternion_multiply, quaternion_from_matrix, quaternion_from_euler, quaternion_slerp, euler_matrix
from .motion_blending import smooth_quaternion_frames_with_slerp, smooth_quaternion_frames, blend_quaternion_frames_linearly, smooth_quaternion_frames_with_slerp2
from .motion_editing.motion_grounding import create_grounding_constraint_from_frame, generate_ankle_constraint_from_toe, interpolate_constraints
from .motion_editing.analytical_inverse_kinematics import AnalyticalLimbIK
from .motion_editing.utils import normalize, generate_root_constraint_for_two_feet, smooth_root_translation_at_start, smooth_root_translation_at_end
from .motion_blending import blend_between_frames, smooth_translation_in_quat_frames, generate_blended_frames, interpolate_frames, smooth_root_translation_around_transition, blend_quaternions_to_next_step, smooth_quaternion_frames_joint_filter, blend_towards_next_step_linear_with_original

ALIGNMENT_MODE_FAST = 0
ALIGNMENT_MODE_PCL = 1


def concatenate_frames(prev_frames, new_frames):
    frames = prev_frames.tolist()
    for idx in range(1, len(new_frames)):  # skip first frame
        frames.append(new_frames[idx])
    frames = np.array(frames)
    return frames


def convert_quat_frame_to_point_cloud(skeleton, frame, joints=None):
    points = []
    if joints is None:
        joints = [k for k, n in list(skeleton.nodes.items()) if len(n.children) > 0 and "Bip" not in n.node_name]
    for j in joints:
        p = skeleton.nodes[j].get_global_position(frame)
        points.append(p)
    return points


def _align_point_clouds_2D(a, b, weights):
    '''
     Finds aligning 2d transformation of two equally sized point clouds.
     from Motion Graphs paper by Kovar et al.
     Parameters
     ---------
     *a: list
     \t point cloud
     *b: list
     \t point cloud
     *weights: list
     \t weights of correspondences
     Returns
     -------
     *theta: float
     \t angle around y axis in radians
     *offset_x: float
     \t
     *offset_z: float

     '''
    if len(a) != len(b):
        raise ValueError("two point cloud should have the same number points: "+str(len(a))+","+str(len(b)))
    n_points = len(a)
    numerator_left = 0
    denominator_left = 0
    weighted_sum_a_x = 0
    weighted_sum_b_x = 0
    weighted_sum_a_z = 0
    weighted_sum_b_z = 0
    sum_of_weights = 0.0
    #    if not weights:
    #        weight = 1.0/n_points # todo set weight base on joint level
    for index in range(n_points):
        numerator_left += weights[index] * (a[index][0] * b[index][2] -
                                            b[index][0] * a[index][2])
        denominator_left += weights[index] * (a[index][0] * b[index][0] +
                                              a[index][2] * b[index][2])
        sum_of_weights += weights[index]
        weighted_sum_a_x += weights[index] * a[index][0]
        weighted_sum_b_x += weights[index] * b[index][0]
        weighted_sum_a_z += weights[index] * a[index][2]
        weighted_sum_b_z += weights[index] * b[index][2]
    numerator_right = 1.0 / sum_of_weights * \
        (weighted_sum_a_x * weighted_sum_b_z -
         weighted_sum_b_x * weighted_sum_a_z)
    numerator = numerator_left - numerator_right
    denominator_right = 1.0 / sum_of_weights * \
        (weighted_sum_a_x * weighted_sum_b_x +
         weighted_sum_a_z * weighted_sum_b_z)
    denominator = denominator_left - denominator_right
    theta = np.arctan2(numerator, denominator)
    offset_x = (weighted_sum_a_x - weighted_sum_b_x *
                np.cos(theta) - weighted_sum_b_z * np.sin(theta)) / sum_of_weights
    offset_z = (weighted_sum_a_z + weighted_sum_b_x *
                np.sin(theta) - weighted_sum_b_z * np.cos(theta)) / sum_of_weights

    return theta, offset_x, offset_z


def transform_quaternion_frames_legacy(quat_frames, angles, offset, rotation_order=None):
    """ Applies a transformation on the root joint of a list quaternion frames.
    Parameters
    ----------
    *quat_frames: np.ndarray
    \tList of frames where the rotation is represented as euler angles in degrees.
    *angles: list of floats
    \tRotation angles in degrees
    *offset:  np.ndarray
    \tTranslation
    """
    if rotation_order is None:
        rotation_order = DEFAULT_ROTATION_ORDER
    offset = np.array(offset)
    if round(angles[0], 3) == 0 and round(angles[2], 3) == 0:
        rotation_q = quaternion_about_axis(np.deg2rad(angles[1]), [0, 1, 0])
    else:
        rotation_q = euler_to_quaternion(angles, rotation_order)
    rotation_matrix = euler_angles_to_rotation_matrix(angles, rotation_order)[:3, :3]
    for frame in quat_frames:
        ot = frame[:3]
        oq = frame[3:7]
        frame[:3] = np.dot(rotation_matrix, ot) + offset
        frame[3:7] = quaternion_multiply(rotation_q, oq)
    return quat_frames

def pose_orientation_quat(quaternion_frame):
    """Estimate pose orientation from root orientation
    """
    ref_offset = np.array([0, 0, 1, 1])
    rotmat = quaternion_matrix(quaternion_frame[3:7])
    rotated_point = np.dot(rotmat, ref_offset)
    dir_vec = np.array([rotated_point[0], rotated_point[2]])
    dir_vec /= np.linalg.norm(dir_vec)
    return dir_vec

def fast_quat_frames_transformation(quaternion_frames_a,
                                    quaternion_frames_b):
    dir_vec_a = pose_orientation_quat(quaternion_frames_a[-1])
    dir_vec_b = pose_orientation_quat(quaternion_frames_b[0])
    angle = get_rotation_angle(dir_vec_a, dir_vec_b)
    offset_x = quaternion_frames_a[-1][0] - quaternion_frames_b[0][0]
    offset_z = quaternion_frames_a[-1][2] - quaternion_frames_b[0][2]
    offset = [offset_x, 0.0, offset_z]
    return angle, offset



def get_orientation_vector_from_matrix(m, v=[0, 0, 1]):
    p = np.dot(m, v)
    dir_vec = np.array([p[0], p[2]])
    dir_vec /= np.linalg.norm(dir_vec)
    return dir_vec


def get_global_node_orientation_vector(skeleton, node_name, frame, v=[0, 0, 1]):
    v = np.array(v)
    m = skeleton.nodes[node_name].get_global_matrix(frame)[:3, :3]
    p = np.dot(m, v)
    dir_vec = np.array([p[0], p[2]])
    dir_vec /= np.linalg.norm(dir_vec)
    return dir_vec


def get_node_aligning_2d_transform(skeleton, node_name, prev_frames, new_frames):
    """from last of prev frames to first of new frames"""

    m_a = skeleton.nodes[node_name].get_global_matrix(prev_frames[-1])
    m_b = skeleton.nodes[node_name].get_global_matrix(new_frames[0])
    dir_vec_a = get_orientation_vector_from_matrix(m_a[:3, :3])
    dir_vec_b = get_orientation_vector_from_matrix(m_b[:3, :3])
    angle = get_rotation_angle(dir_vec_a, dir_vec_b)
    q = quaternion_about_axis(np.deg2rad(angle), [0, 1, 0])
    m = quaternion_matrix(q)

    first_frame_pos = m_b[:4,3]
    rotated_first_frame_pos = np.dot(m, first_frame_pos)[:3]
    offset_x = m_a[0, 3] - rotated_first_frame_pos[0]
    offset_z = m_a[2, 3] - rotated_first_frame_pos[2]
    m[:3,3] = [offset_x, 0.0, offset_z]
    return m

def get_transform_from_point_cloud_alignment(skeleton, prev_frames, new_frames):
    weights = skeleton.get_joint_weights()
    p_a = convert_quat_frame_to_point_cloud(skeleton, prev_frames[-1])
    p_b = convert_quat_frame_to_point_cloud(skeleton, new_frames[0])
    theta, offset_x, offset_z = _align_point_clouds_2D(p_a, p_b, weights)
    euler = [0, np.radians(theta), 0]
    m = np.eye(4)
    m[:3,:3] = euler_matrix(*euler)[:3,:3]
    m[0,3] = offset_x
    m[2,3] = offset_z
    print("run point cloud alignment", theta, offset_x, offset_z, m)
    return m

def transform_quaternion_frames(frames, m,
                                translation_param_range=(0, 3),
                                orientation_param_range=(3, 7)):
    """ possibly broken because not 3,7 is the root orientation but 7,11
    """
    q = quaternion_from_matrix(m)
    for frame in frames:
        ot = frame[translation_param_range[0]:translation_param_range[1]].tolist() + [1]
        oq = frame[orientation_param_range[0]:orientation_param_range[1]]
        transformed_t = np.dot(m, ot)[:3]
        frame[translation_param_range[0]:translation_param_range[1]] = transformed_t
        frame[orientation_param_range[0]:orientation_param_range[1]] = quaternion_multiply(q, oq)
    return frames


def concatenate_frames_smoothly(new_frames, prev_frames, smoothing_window=0):
    d = len(prev_frames)
    frames = concatenate_frames(prev_frames, new_frames)
    if smoothing_window > 0:
        frames = smooth_quaternion_frames(frames, d, smoothing_window)
    return frames


def concatenate_frames_with_slerp(new_frames, prev_frames, smoothing_window=0):
    '''

    :param new_frames (numpy.array): n_frames * n_dims
    :param prev_frames (numpy.array): n_frames * n_dims
    :param smoothing_window:
    :return:
    '''
    d = len(prev_frames)
    frames = concatenate_frames(prev_frames, new_frames)
    if smoothing_window > 0:
        frames = smooth_quaternion_frames_with_slerp(frames, d, smoothing_window)
    return frames



def align_quaternion_frames_automatically(skeleton, node_name, new_frames, prev_frames, alignment_mode=ALIGNMENT_MODE_FAST):
    if alignment_mode == ALIGNMENT_MODE_FAST:
        m = get_node_aligning_2d_transform(skeleton, node_name, prev_frames, new_frames)
    else:
        m = get_transform_from_point_cloud_alignment(skeleton, prev_frames, new_frames)

    new_frames = transform_quaternion_frames(new_frames, m)
    return new_frames


def align_quaternion_frames_automatically2(skeleton, node_name, new_frames, prev_frames):
    angle, offset = fast_quat_frames_transformation(prev_frames, new_frames)
    new_frames = transform_quaternion_frames_legacy(new_frames, [0, angle, 0], offset)
    return new_frames


def align_quaternion_frames(skeleton, node_name, new_frames, prev_frames=None,  start_pose=None):
    if prev_frames is not None:
        return align_quaternion_frames_automatically(skeleton, node_name, new_frames,  prev_frames)
    elif start_pose is not None:
        m = get_transform_from_start_pose(start_pose)
        first_frame_pos = new_frames[0][:3].tolist() + [1]
        t_pos = np.dot(m, first_frame_pos)[:3]
        delta = start_pose["position"]
        # FIXME this assumes the y translation is the up axis and can be ignored
        delta[0] -= t_pos[0]
        delta[2] -= t_pos[2]
        m[:3, 3] = delta
        transformed_frames = transform_quaternion_frames(new_frames, m)
        return transformed_frames
    else:
        return new_frames


def get_transform_from_start_pose(start_pose):
    e = np.deg2rad(start_pose["orientation"])
    p = start_pose["position"]
    if None not in e:
        q = quaternion_from_euler(*e)
        m = quaternion_matrix(q)
    else:
        m = np.eye(4)
    if None not in p:
        m[:3,3] = p
    return m


def align_and_concatenate_frames(skeleton, joint_name, new_frames, prev_frames=None, start_pose=None, smoothing_window=0,
                                 blending_method='smoothing'):
    new_frames = align_quaternion_frames(skeleton, joint_name, new_frames, prev_frames, start_pose)

    if prev_frames is not None:
        d = len(prev_frames)
        new_frames = concatenate_frames(prev_frames, new_frames)
        if smoothing_window > 0:
            if blending_method == 'smoothing':
                new_frames = smooth_quaternion_frames(new_frames, d, smoothing_window)
            elif blending_method == 'slerp':
                new_frames = smooth_quaternion_frames_with_slerp(new_frames, d, smoothing_window)
            elif blending_method == 'slerp2':
                new_frames = smooth_quaternion_frames_with_slerp2(skeleton, new_frames, smoothing_window)
            elif blending_method == 'linear':
                new_frames = blend_quaternion_frames_linearly(new_frames, prev_frames, skeleton, smoothing_window)
            else:
                raise KeyError('Unknown method!')
    return new_frames


blend = lambda x: 2 * x * x * x - 3 * x * x + 1

def align_frames_and_fix_foot_to_prev(skeleton, aligning_joint, new_frames, prev_frames, start_pose, foot_joint, ik_chain, ik_window=7, smoothing_window=0):
    new_frames = align_quaternion_frames(skeleton, aligning_joint, new_frames, prev_frames, start_pose)
    if prev_frames is not None:
        offset = prev_frames[-1][:3] - new_frames[0][:3]

        d = len(prev_frames)
        frames = prev_frames.tolist()
        for idx in range(1, len(new_frames)):  # skip first frame
            frames.append(new_frames[idx])
        frames = np.array(frames)

        transition_start = d
        c = create_grounding_constraint_from_frame(skeleton, frames, d-1, foot_joint)
        ik = AnalyticalLimbIK.init_from_dict(skeleton, c.joint_name, ik_chain)
        before = skeleton.nodes[foot_joint].get_global_position(frames[transition_start])

        frames[transition_start] = ik.apply2(frames[transition_start], c.position, c.orientation)

        transition_end = d+ik_window
        print("allign frames", c.position, foot_joint, d-1, transition_end, before, skeleton.nodes[foot_joint].get_global_position(frames[transition_start]))
        print(skeleton.nodes[foot_joint].get_global_position(frames[d]))

        chain_joints =  [ik_chain["root"], ik_chain["joint"], foot_joint]#[skeleton.root] +
        #chain_joints = []
        for c_joint in chain_joints:
            idx = skeleton.animated_joints.index(c_joint) * 4 + 3
            j_indices = [idx, idx + 1, idx + 2, idx + 3]
            start_q = frames[transition_start][j_indices]
            end_q = frames[transition_end][j_indices]
            print(c_joint, start_q, end_q,j_indices)
            for i in range(ik_window):
                t = float(i) / ik_window
                # nlerp_q = self.nlerp(start_q, end_q, t)
                slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
                print(transition_start+i+1, frames[transition_start + 1 + i][j_indices], slerp_q)
                # print "slerp",start_q,  end_q, t, nlerp_q, slerp_q
                frames[transition_start + 1 + i][j_indices] = slerp_q#[1,0,0,0]
            #smooth_quaternion_frames_using_slerp_(frames, j_indices, d, ik_window)
            #apply_slerp2(frames, j_indices, transition_start, transition_end, transition_end - transition_start, BLEND_DIRECTION_BACKWARD)
            #for j, frame in enumerate(frames[transition_start+1:transition_end]):
            #    w = (j + 1) / (ik_window + 1)
            #    frames[transition_start+1+j][j_indices] = normalize(quaternion_slerp(bq, fq, blend(w), spin=0, shortestpath=True))
        idx = skeleton.animated_joints.index(foot_joint) * 4 + 3
        j_indices = [idx, idx + 1, idx + 2, idx + 3]
        print("after smoothing",frames[transition_start + 1][j_indices])
        if smoothing_window > 0 and False:
            frames = smooth_quaternion_frames(frames, d, smoothing_window)
        return frames
    else:
        return new_frames

def get_limb_length(skeleton, joint_name):
    limb_length = np.linalg.norm(skeleton.nodes[joint_name].offset)
    limb_length += np.linalg.norm(skeleton.nodes[joint_name].parent.offset)
    return limb_length

def generate_root_constraint_for_one_foot(skeleton, frame, root, c):
        root_pos = skeleton.nodes[root].get_global_position(frame)
        target_length = np.linalg.norm(c.position - root_pos)
        limb_length = get_limb_length(skeleton, c.joint_name)
        if target_length >= limb_length:
            new_root_pos = (c.position + normalize(root_pos - c.position) * limb_length)
            print("one constraint on ", c.joint_name, "- before", root_pos, "after", new_root_pos)
            return new_root_pos
            #frame[:3] = new_root_pos

        else:
            print("no change")


def translate_root(skeleton, frames, target_frame_idx, plant_heel, ground_height=0):
    """ translate the next frames closer to the previous frames root translation"""
    #delta = frames[target_frame_idx-1][:3]-frames[target_frame_idx][:3]
    n_frames = len(frames)
    foot_pos = skeleton.nodes[plant_heel].get_global_position(frames[target_frame_idx-1])
    print("foot pos before", foot_pos)
    delta = ground_height - foot_pos[1]
    n_frames = len(frames)
    for f in range(target_frame_idx, n_frames):
        frames[f][1] += delta
    print("after", skeleton.nodes[plant_heel].get_global_position(frames[target_frame_idx]))
    for f in range(target_frame_idx, n_frames):
        frames[f,:3] += delta/2


def apply_constraint(skeleton, frames, c, ik_chain, frame_idx, start, end, window):
    print("apply swing foot constraint on frame", frame_idx, start, end)
    ik = AnalyticalLimbIK.init_from_dict(skeleton, c.joint_name, ik_chain)
    frames[frame_idx] = ik.apply2(frames[frame_idx], c.position, c.orientation)
    joint_list = [ik_chain["root"], ik_chain["joint"], c.joint_name]
    blend_between_frames(skeleton, frames, start, end, joint_list, window)


def apply_constraint_on_window_prev(skeleton, frames, c, ik_chain, start, end, window):
    ik = AnalyticalLimbIK.init_from_dict(skeleton, c.joint_name, ik_chain)
    indices = list(range(start, end + 1))
    print("apply on frames", indices)
    for f in indices:
        frames[f] = ik.apply2(frames[f], c.position, c.orientation)
    joint_list = [ik_chain["root"], ik_chain["joint"], c.joint_name]
    blend_between_frames(skeleton, frames, end, end+window, joint_list, window)


def apply_constraint_on_window_next(skeleton, frames, c, ik_chain, start, end, window):
    ik = AnalyticalLimbIK.init_from_dict(skeleton, c.joint_name, ik_chain)
    indices = list(range(start, end + 1))
    print("apply on frames", indices)
    for f in indices:
        frames[f] = ik.apply2(frames[f], c.position, c.orientation)
    joint_list = [ik_chain["root"], ik_chain["joint"], c.joint_name]
    print("blend between frames",start-window, start)
    blend_between_frames(skeleton, frames, start-window, start, joint_list, window)


def align_foot_to_next_step(skeleton, frames, foot_joint, ik_chain, target_frame_idx, window):
    start = target_frame_idx - window  # start of blending range
    end = target_frame_idx - 1 # modified frame

    c = create_grounding_constraint_from_frame(skeleton, frames, target_frame_idx, foot_joint)
    apply_constraint(skeleton, frames, c, ik_chain, target_frame_idx, start, end, window)


def align_foot_to_prev_step(skeleton, frames, foot_joint, ik_chain, target_frame_idx, window):
    start = target_frame_idx # modified frame
    end = target_frame_idx + window  # end of blending range
    c = create_grounding_constraint_from_frame(skeleton, frames, target_frame_idx-1, foot_joint)
    apply_constraint(skeleton, frames, c, ik_chain, target_frame_idx, start, end, window)




def generate_feet_constraints(skeleton, frames, frame_idx, plant_side, swing_side, target_ground_height):
    plant_foot_joint = skeleton.skeleton_model["joints"][plant_side + "_ankle"]
    plant_toe_joint = skeleton.skeleton_model["joints"][plant_side + "_toe"]
    plant_heel_joint = skeleton.skeleton_model["joints"][plant_side + "_heel"]
    swing_foot_joint = skeleton.skeleton_model["joints"][swing_side + "_ankle"]
    swing_toe_joint = skeleton.skeleton_model["joints"][swing_side + "_toe"]
    swing_heel_joint = skeleton.skeleton_model["joints"][swing_side + "_heel"]
    plant_constraint = generate_ankle_constraint_from_toe(skeleton, frames, frame_idx, plant_foot_joint,
                                                          plant_heel_joint, plant_toe_joint, target_ground_height)
    swing_constraint = generate_ankle_constraint_from_toe(skeleton, frames, frame_idx, swing_foot_joint,
                                                          swing_heel_joint, swing_toe_joint, target_ground_height)
    return plant_constraint, swing_constraint


def generate_feet_constraints2(skeleton, frames, frame_idx, plant_side, swing_side):
    plant_foot_joint = skeleton.skeleton_model["joints"][plant_side + "_ankle"]
    swing_foot_joint = skeleton.skeleton_model["joints"][swing_side + "_ankle"]
    plant_constraint = create_grounding_constraint_from_frame(skeleton, frames, frame_idx - 1, plant_foot_joint)
    swing_constraint = create_grounding_constraint_from_frame(skeleton, frames, frame_idx - 1, swing_foot_joint)
    return plant_constraint, swing_constraint


def align_feet_to_prev_step(skeleton, frames, frame_idx, plant_constraint, swing_constraint, ik_chains, window):
    start = frame_idx  # modified frame
    end = frame_idx + window  # end of blending range
    apply_constraint_on_window_prev(skeleton, frames, plant_constraint, ik_chains[plant_constraint.joint_name], start, end, window)
    apply_constraint(skeleton, frames, swing_constraint, ik_chains[swing_constraint.joint_name], frame_idx, start, end, window)


def align_feet_to_next_step(skeleton, frames, frame_idx, plant_constraint, swing_constraint, ik_chains, plant_window, window):
    start = frame_idx - window  # end of blending range
    end = frame_idx  # modified frame
    apply_constraint_on_window_next(skeleton, frames, plant_constraint, ik_chains[plant_constraint.joint_name], start, end, plant_window)
    apply_constraint(skeleton, frames, swing_constraint, ik_chains[swing_constraint.joint_name], frame_idx, start, end, window)


def align_feet_to_next_step2(skeleton, frames, frame_idx, plant_constraint, swing_constraint, ik_chains, plant_window, window):
    start = frame_idx - window  # end of blending range
    end = frame_idx  # modified frame
    ik_chain = ik_chains[plant_constraint.joint_name]
    joint_list = [ik_chain["root"], ik_chain["joint"], plant_constraint.joint_name]
    blend_between_frames(skeleton, frames, start, end, joint_list, window)
    ik_chain = ik_chains[swing_constraint.joint_name]
    joint_list = [ik_chain["root"], ik_chain["joint"], plant_constraint.joint_name]
    blend_between_frames(skeleton, frames, start, end, joint_list, window)


def fix_feet_at_transition(skeleton, frames, d,  plant_side, swing_side, ik_chains, ik_window=8, plant_window=20):
    target_ground_height = 0
    smooth_root_translation_around_transition(frames, d, 2 * ik_window)

    plant_constraint, swing_constraint = generate_feet_constraints(skeleton, frames, d, plant_side, swing_side, target_ground_height)
    root_pos = generate_root_constraint_for_two_feet(skeleton, frames[d-1], plant_constraint, swing_constraint)
    if root_pos is not None:
        frames[d - 1][:3] = root_pos
        smooth_root_translation_at_end(frames, d - 1, ik_window)
        smooth_root_translation_at_start(frames, d, ik_window)

    align_feet_to_next_step(skeleton, frames, d-1, plant_constraint, swing_constraint, ik_chains, plant_window, ik_window)
    align_feet_to_prev_step(skeleton, frames, d, plant_constraint, swing_constraint, ik_chains, ik_window)
    #swing_foot = skeleton.skeleton_model["joints"][swing_side + "_ankle"]
    #align_foot_to_prev_step(skeleton, frames, swing_foot, ik_chains[swing_foot], d, ik_window)


def align_frames_using_forward_blending(skeleton, aligning_joint, new_frames, prev_frames, prev_start, start_pose, ik_chains, smoothing_window=0):
    """ applies foot ik constraint to fit the prev motion primitive to the next motion primitive
    """

    new_frames = align_quaternion_frames(skeleton, aligning_joint, new_frames, prev_frames, start_pose)
    if prev_frames is not None:
        d = len(prev_frames)
        frames = prev_frames.tolist()
        for idx in range(1, len(new_frames)):  # skip first frame
            frames.append(new_frames[idx])
        frames = np.array(frames)

        blend_end = d
        blend_start = int(prev_start + (blend_end-prev_start)/2)  # start blending from the middle of the step

        left_joint = skeleton.skeleton_model["joints"]["left_ankle"]
        right_joint = skeleton.skeleton_model["joints"]["right_ankle"]
        pelvis = skeleton.skeleton_model["joints"]["pelvis"]
        left_ik_chain = ik_chains[left_joint]
        right_ik_chain = ik_chains[right_joint]
        leg_joint_list = [skeleton.root, left_ik_chain["root"], left_ik_chain["joint"], left_joint,
                      right_ik_chain["root"], right_ik_chain["joint"], right_joint]
        if pelvis != skeleton.root:
            leg_joint_list.append(pelvis)

        frames = blend_towards_next_step_linear_with_original(skeleton, frames, blend_start, blend_end, leg_joint_list)
        joint_list = [j for j in skeleton.animated_joints if j not in leg_joint_list]

        if smoothing_window > 0:
            frames = smooth_quaternion_frames_joint_filter(skeleton, frames, d, joint_list, smoothing_window)
        return frames
    else:
        return new_frames

def blend_towards_next_step_linear(skeleton, frames, d,  plant_side, swing_side, ik_chains, window=8):
    target_ground_height = 0
    smooth_root_translation_around_transition(frames, d, 2 * window)
    plant_constraint, swing_constraint = generate_feet_constraints(skeleton, frames, d, plant_side, swing_side, target_ground_height)
    root_pos = generate_root_constraint_for_two_feet(skeleton, frames[d-1], plant_constraint, swing_constraint)
    if root_pos is not None:
        frames[d - 1][:3] = root_pos
        smooth_root_translation_at_end(frames, d - 1, window)
        smooth_root_translation_at_start(frames, d, window)
    blend_quaternions_to_next_step(skeleton, frames, d, plant_constraint.joint_name, swing_constraint.joint_name, ik_chains, window)



def blend_towards_next_step3(skeleton, frames, start, end, plant_side, swing_side, ik_chains, window=8):
    plant_joint = skeleton.skeleton_model["joints"][plant_side + "_ankle"]
    swing_joint = skeleton.skeleton_model["joints"][swing_side + "_ankle"]
    plant_ik_chain = ik_chains[plant_joint]
    swing_ik_chain = ik_chains[swing_joint]
    joint_list = [skeleton.root, "pelvis", plant_ik_chain["root"], plant_ik_chain["joint"], plant_joint,
                  swing_ik_chain["root"], swing_ik_chain["joint"], swing_joint]
    middle = int(start + (end-start)/2)
    new_frames = generate_blended_frames(skeleton, frames, middle, end, joint_list, end-middle)
    frames = interpolate_frames(skeleton, frames, new_frames, joint_list, middle, end)
    return frames
