import numpy as np
from motion_editing import euler_substraction, point_to_euler_angle, euler_to_quaternion, euler_angles_to_rotation_matrix, get_rotation_angle, DEFAULT_ROTATION_ORDER
from ..external.transformations import quaternion_matrix, quaternion_about_axis, quaternion_multiply, quaternion_from_matrix, quaternion_from_euler


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
                frames[
                    j + 1][3 + i * 4:3 + (i + 1) * 4] = -frames[j + 1][3 + i * 4:3 + (i + 1) * 4]
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
    offset_x = m_a[0, 3] - m_b[0, 3]
    offset_z = m_a[2, 3] - m_b[2, 3]
    q = quaternion_about_axis(np.deg2rad(angle), [0, 1, 0])
    m = quaternion_matrix(q)
    m[:3,3] = [offset_x, 0.0, offset_z]
    return m


def transform_quaternion_frames(frames, m,
                                translation_param_range=(0, 3),
                                orientation_param_range=(3, 7)):
    """ possibly broken
    """
    q = quaternion_from_matrix(m)
    for frame in frames:
        ot = frame[translation_param_range[0]:translation_param_range[1]].tolist() + [1]
        oq = frame[orientation_param_range[0]:orientation_param_range[1]]
        frame[translation_param_range[0]:translation_param_range[1]] = np.dot(m, ot)[:3]
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


def align_and_concatenate_frames(skeleton, node_name, new_frames, prev_frames=None, start_pose=None, smoothing_window=0):
    new_frames = align_quaternion_frames_with_start(skeleton, node_name, new_frames, prev_frames, start_pose)
    if prev_frames is not None:
        return concatenate_frames(new_frames, prev_frames, smoothing_window)
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
        return transform_quaternion_frames(new_frames, m)
    else:
        return new_frames


def get_transform_from_start_pose(start_pose):
    e = np.deg2rad(start_pose["orientation"])
    p = start_pose["position"]
    if None not in e:
        q = quaternion_from_euler(*e)
        m = quaternion_matrix(q)
    else:
        m = np.eyes(4)
    if None not in p:
        m[:3,3] = p
    return m

