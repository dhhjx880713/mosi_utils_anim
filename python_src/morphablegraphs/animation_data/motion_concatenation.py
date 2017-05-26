import numpy as np
from motion_editing import euler_substraction, point_to_euler_angle, euler_to_quaternion, euler_angles_to_rotation_matrix, get_rotation_angle, DEFAULT_ROTATION_ORDER, LEN_QUAT, LEN_EULER, LEN_ROOT_POS
from ..external.transformations import quaternion_matrix, quaternion_about_axis, quaternion_multiply, quaternion_from_matrix, quaternion_from_euler, quaternion_slerp
import copy


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


def slerp_quaternion_frame(frame_a, frame_b, weight):
    frame_a = np.asarray(frame_a)
    frame_b = np.asarray(frame_b)
    assert len(frame_a) == len(frame_b)
    n_joints = (len(frame_a) - 3) / 4
    new_frame = np.zeros(len(frame_a))
    # linear interpolate root translation
    new_frame[:3] = (1 - weight) * frame_a[:3] + weight * frame_b[:3]
    for i in range(n_joints):
        new_frame[3+ i*4 : 3 + (i+1) * 4] = quaternion_slerp(frame_a[3 + i*4 : 3 + (i+1) * 4],
                                                             frame_b[3 + i*4 : 3 + (i+1) * 4],
                                                             weight)
    return new_frame


def smooth_quaternion_frames_with_slerp(frames, discontinuity, window=20):
    n_frames = len(frames)
    d = float(discontinuity)
    ref_pose = slerp_quaternion_frame(frames[int(d)-1], frames[int(d)], 0.5)
    w = float(window)
    new_quaternion_frames = []
    for f in xrange(n_frames):
        if f < d - w:
            new_quaternion_frames.append(frames[f])
        elif d - w <= f < d:
            tmp = (f - d + w) / w
            weight =2 * ( 0.5 * tmp ** 2)
            new_quaternion_frames.append(slerp_quaternion_frame(frames[f], ref_pose, weight))
        elif d <= f <= d + w:
            tmp = (f - d + w) / w
            weight =2 * ( 0.5 * tmp ** 2 - 2 * tmp + 2)
            new_quaternion_frames.append(slerp_quaternion_frame(frames[f], ref_pose, weight))
        else:
            new_quaternion_frames.append(frames[f])
    return np.asarray(new_quaternion_frames)



def smooth_quaternion_frames(frames, discontinuity, window=20):
    """ Smooth quaternion frames given discontinuity frame

    Parameters
    ----------
    frames: list
    \tA list of quaternion frames
    discontinuity : int
    The frame where the discontinuity is. (e.g. the transitionframe)
    window : (optional) int, default is 20
    The smoothing window
    Returns
    -------
    None.
    """
    n_joints = (len(frames[0]) - 3) / 4
    # smooth quaternion
    n_frames = len(frames)
    for i in xrange(n_joints):
        for j in xrange(n_frames - 1):
            q1 = np.array(frames[j][3 + i * 4: 3 + (i + 1) * 4])
            q2 = np.array(frames[j + 1][3 + i * 4:3 + (i + 1) * 4])
            if np.dot(q1, q2) < 0:
                frames[j + 1][3 + i * 4:3 + (i + 1) * 4] = -frames[j + 1][3 + i * 4:3 + (i + 1) * 4]
    # generate curve of smoothing factors
    d = float(discontinuity)
    w = float(window)
    smoothing_factors = []
    for f in xrange(n_frames):
        value = 0.0
        if d - w <= f < d:
            tmp = (f - d + w) / w
            value = 0.5 * tmp ** 2
        elif d <= f <= d + w:
            tmp = (f - d + w) / w
            value = -0.5 * tmp ** 2 + 2 * tmp - 2
        smoothing_factors.append(value)
    smoothing_factors = np.array(smoothing_factors)
    new_quaternion_frames = []
    for i in xrange(len(frames[0])):
        current_value = frames[:, i]
        magnitude = current_value[int(d)] - current_value[int(d) - 1]
        new_value = current_value + (magnitude * smoothing_factors)
        new_quaternion_frames.append(new_value)
    new_quaternion_frames = np.array(new_quaternion_frames).T
    return new_quaternion_frames


def linear_blending(ref_pose, quat_frames, skeleton, weights, joint_list=None):
    '''
    Apply linear blending on quaternion motion data
    :param ref_pose (QuaternionFrame):
    :param quat_frames:
    :param skeleton (morphablegraphs.animation_data.Skeleton):
    :param weights (numpy.array): weights used for slerp
    :param joint_list (list): animated joint to be blended
    :return:
    '''
    if joint_list is None:
        joint_list = skeleton.animated_joints
    new_frames = copy.deepcopy(quat_frames)
    for i in range(len(quat_frames)):
        for joint in joint_list:
            joint_index = skeleton.nodes[joint].quaternion_frame_index
            start_index = LEN_ROOT_POS + LEN_QUAT * joint_index
            end_index = LEN_ROOT_POS + LEN_QUAT * (joint_index + 1)
            ref_q = ref_pose[start_index: end_index]
            motion_q = quat_frames[i, start_index: end_index]
            new_frames[i, start_index: end_index] = quaternion_slerp(ref_q, motion_q, weights[i])
    return new_frames

def blend_quaternion_frames(new_frames, prev_frames, skeleton, smoothing_window=None):
    '''
    Blend new frames linearly based on the last pose of previous frames
    :param new_frames (Quaternion Frames):
    :param prev_frames (Quaternion Frames):
    :param skeleton (morphablegraphs.animation_data.Skeleton):
    :param smoothing_window (int): smoothing window decides how many frames will be blended, if is None, then blend all
    :return:
    '''
    if smoothing_window is not None and smoothing_window != 0:
        slerp_weights = np.linspace(0, 1, smoothing_window)
    else:
        slerp_weights = np.linspace(0, 1, len(new_frames))

    return linear_blending(prev_frames[-1], new_frames, skeleton, slerp_weights)


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


def concatenate_frames(new_frames, prev_frames, smoothing_window=0):
    d = len(prev_frames)
    frames = prev_frames.tolist()
    for f in new_frames:
        frames.append(f)
    frames = np.array(frames)
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
    frames = prev_frames.tolist()
    for f in new_frames:
        frames.append(f)
    frames = np.array(frames)
    if smoothing_window > 0:
        frames = smooth_quaternion_frames_with_slerp(frames, d, smoothing_window)
    return frames


def align_and_concatenate_frames(skeleton, node_name, new_frames, prev_frames=None, start_pose=None, smoothing_window=0,
                                 method='smoothing'):
    new_frames = align_quaternion_frames_with_start(skeleton, node_name, new_frames, prev_frames, start_pose)
    if prev_frames is not None:
        if method == 'smoothing':
            return concatenate_frames(new_frames, prev_frames, smoothing_window)
        elif method == 'blending':
            return blend_quaternion_frames(new_frames, prev_frames, skeleton, smoothing_window)
        elif method == 'slerp_smoothing':
            return concatenate_frames_with_slerp(new_frames, prev_frames, smoothing_window)
        else:
            raise KeyError('Unknown method!')
    else:
        return new_frames


def align_quaternion_frames(skeleton, node_name, new_frames, prev_frames):
    """new general version"""
    m = get_node_aligning_2d_transform(skeleton, node_name, prev_frames, new_frames)
    new_frames = transform_quaternion_frames(new_frames, m)
    return new_frames


def align_quaternion_frames2(skeleton, node_name, new_frames, prev_frames):
    angle, offset = fast_quat_frames_transformation(prev_frames, new_frames)
    new_frames = transform_quaternion_frames_legacy(new_frames, [0, angle, 0], offset)
    return new_frames


def align_quaternion_frames_with_start(skeleton, node_name, new_frames, prev_frames=None,  start_pose=None):
    if prev_frames is not None:
        return align_quaternion_frames(skeleton, node_name, new_frames,  prev_frames)
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

