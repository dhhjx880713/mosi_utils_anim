from ..external.transformations import quaternion_slerp
import numpy as np
import copy
from copy import deepcopy
from .constants import LEN_QUAT, LEN_ROOT_POS


BLEND_DIRECTION_FORWARD = 0
BLEND_DIRECTION_BACKWARD = 1


def blend_quaternion(a, b, w):
    return quaternion_slerp(a, b, w, spin=0, shortestpath=True)


def create_transition(a, b, steps):
    transition = []
    for i in range(steps):
        t = float(i) / steps
        new_t = (1-t)*a + t*b
        transition.append(new_t)
    return transition


def smooth_translation(quat_frames, event_frame, window):
    h_window = window / 2
    start_frame = max(event_frame - h_window, 0)
    end_frame = min(event_frame + h_window, quat_frames.shape[0] - 1)

    start_t = quat_frames[start_frame, :3]
    event_t = quat_frames[event_frame, :3]
    end_t = quat_frames[end_frame, :3]
    from_start_to_event = create_transition(start_t, event_t, h_window)
    from_event_to_end = create_transition(event_t, end_t, h_window)

    # blend transition frames with original frames
    steps = event_frame - start_frame
    for i in range(steps):
        t = float(i) / steps
        orig = quat_frames[start_frame+i, :3]
        quat_frames[start_frame + i, :3] = (1-t)*orig + t*from_start_to_event[i]

    steps = end_frame - event_frame
    for i in range(steps):
        t = float(i) / steps
        orig = quat_frames[event_frame + i, :3]
        quat_frames[start_frame + i, :3] = (1 - t) * orig + t * from_event_to_end[i]


def smooth_quaternion_frames_using_slerp(quat_frames, joint_param_indices, event_frame, window):
    h_window = window/2
    start_frame = max(event_frame-h_window, 0)
    end_frame = min(event_frame+h_window, quat_frames.shape[0]-1)
    apply_slerp2(quat_frames, joint_param_indices, start_frame, event_frame, h_window, BLEND_DIRECTION_FORWARD)
    apply_slerp2(quat_frames, joint_param_indices, event_frame, end_frame, h_window, BLEND_DIRECTION_BACKWARD)

def smooth_quaternion_frames_using_slerp_(quat_frames, joint_parameter_indices, event_frame, window):
    start_frame = event_frame-window/2
    end_frame = event_frame+window/2
    start_q = quat_frames[start_frame, joint_parameter_indices]
    end_q = quat_frames[end_frame, joint_parameter_indices]
    for i in range(window):
        t = float(i)/window
        #nlerp_q = self.nlerp(start_q, end_q, t)
        slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
        #print "slerp",start_q,  end_q, t, nlerp_q, slerp_q
        quat_frames[start_frame+i, joint_parameter_indices] = slerp_q

def apply_slerp2(quat_frames, joint_param_indices, start_frame, end_frame, steps, direction=BLEND_DIRECTION_FORWARD):
    new_quats = create_frames_using_slerp(quat_frames, start_frame, end_frame, steps, joint_param_indices)
    for i in range(steps):
        if direction==BLEND_DIRECTION_FORWARD:
            t = float(i)/steps
        else:
            t = 1.0 - (i / steps)
        old_quat = quat_frames[start_frame+i, joint_param_indices]
        blended_quat = blend_quaternion(old_quat, new_quats[i], t)
        quat_frames[start_frame + i, joint_param_indices] = blended_quat



def smooth_quaternion_frames_using_slerp_old(quat_frames, joint_param_indices, event_frame, window):
    h_window = window/2
    start_frame = max(event_frame-h_window, 0)
    end_frame = min(event_frame+h_window, quat_frames.shape[0]-1)
    # create transition frames
    from_start_to_event = create_frames_using_slerp(quat_frames, start_frame, event_frame, h_window, joint_param_indices)
    from_event_to_end = create_frames_using_slerp(quat_frames, event_frame, end_frame, h_window, joint_param_indices)

    #blend transition frames with original frames
    steps = event_frame-start_frame
    for i in range(steps):
        t = float(i)/steps
        quat_frames[start_frame+i, joint_param_indices] = blend_quaternion(quat_frames[start_frame+i, joint_param_indices], from_start_to_event[i], t)

    steps = end_frame-event_frame
    for i in range(steps):
        t = 1.0-(i/steps)
        quat_frames[event_frame+i, joint_param_indices] = blend_quaternion(quat_frames[start_frame+i, joint_param_indices], from_event_to_end[i], t)



def smooth_quaternion_frames_using_slerp_overwrite_frames(quat_frames, joint_param_indices, event_frame, window):
    h_window = window/2
    start_frame = max(event_frame-h_window, 0)
    end_frame = min(event_frame+h_window, quat_frames.shape[0]-1)
    apply_slerp(quat_frames, start_frame, event_frame, joint_param_indices)
    apply_slerp(quat_frames, event_frame, end_frame, joint_param_indices)


def blend_frames(quat_frames, start, end, new_frames, joint_parameter_indices):
    steps = end-start
    for i in range(steps):
        t = i/steps
        quat_frames[start+i, joint_parameter_indices] = blend_quaternion(quat_frames[start+i, joint_parameter_indices], new_frames[i], t)


def create_frames_using_slerp(quat_frames, start_frame, end_frame, steps, joint_parameter_indices):
    start_q = quat_frames[start_frame, joint_parameter_indices]
    end_q = quat_frames[end_frame, joint_parameter_indices]
    frames = []
    for i in range(steps):
        t = float(i)/steps
        slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
        frames.append(slerp_q)
    return frames


def apply_slerp(quat_frames, start_frame, end_frame, joint_parameter_indices):
    start_q = quat_frames[start_frame, joint_parameter_indices]
    end_q = quat_frames[end_frame, joint_parameter_indices]
    steps = end_frame-start_frame
    for i in range(steps):
        t = float(i)/steps
        slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
        quat_frames[start_frame+i, joint_parameter_indices] = slerp_q

###################################
#from motion_concatenation

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
    for f in range(n_frames):
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
    n_joints = int((len(frames[0]) - 3) / 4)
    # smooth quaternion
    n_frames = len(frames)
    for i in range(n_joints):
        for j in range(n_frames - 1):
            q1 = np.array(frames[j][3 + i * 4: 3 + (i + 1) * 4])
            q2 = np.array(frames[j + 1][3 + i * 4:3 + (i + 1) * 4])
            if np.dot(q1, q2) < 0:
                frames[j + 1][3 + i * 4:3 + (i + 1) * 4] = -frames[j + 1][3 + i * 4:3 + (i + 1) * 4]
    # generate curve of smoothing factors
    d = float(discontinuity)
    w = float(window)
    smoothing_factors = []
    for f in range(n_frames):
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
    for i in range(len(frames[0])):
        current_value = frames[:, i]
        magnitude = current_value[int(d)] - current_value[int(d) - 1]
        new_value = current_value + (magnitude * smoothing_factors)
        new_quaternion_frames.append(new_value)
    new_quaternion_frames = np.array(new_quaternion_frames).T
    return new_quaternion_frames


def smooth_translation_in_quat_frames(frames, discontinuity, window=20):
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

    n_frames = len(frames)
    # generate curve of smoothing factors
    d = float(discontinuity)
    w = float(window)
    smoothing_factors = []
    for f in range(n_frames):
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
    for i in range(3):
        current_value = frames[:, i]
        magnitude = current_value[int(d)] - current_value[int(d) - 1]
        new_value = current_value + (magnitude * smoothing_factors)
        new_quaternion_frames.append(new_value)
    new_quaternion_frames = np.array(new_quaternion_frames).T
    return new_quaternion_frames




def smooth_root_translation(frames, target_frame_idx, transition_start,transition_end, window):
    root_pos = frames[target_frame_idx, :3]
    frames[transition_end, :3] = root_pos
    #print "root after", frames[transition_end, :3]
    start = frames[transition_start, :3]
    end = frames[target_frame_idx, :3]
    for i in range(window - 1):
        t = float(i) / (window - 1)
        frames[transition_start + i, :3] = start * (1 - t) + end * t


def smooth_root_translation_around_transition(frames, d, window):
    hwindow = int(window/2.0)
    root_pos1 = frames[d-1, :3]
    root_pos2 = frames[d, :3]
    root_pos = (root_pos1 + root_pos2)/2
    #frames[d, :3] = root_pos
    #print "root after", frames[transition_end, :3]
    start_idx = d-hwindow
    end_idx = d + hwindow
    #print start_idx, end_idx,hwindow,d
    start = frames[start_idx, :3]
    end = root_pos
    for i in range(hwindow):
        t = float(i) / hwindow
        frames[start_idx + i, :3] = start * (1 - t) + end * t
        #print start_idx +i, frames[start_idx + i,1]
    start = root_pos
    end = frames[end_idx, :3]
    for i in range(hwindow):
        t = float(i) / hwindow
        frames[d + i, :3] = start * (1 - t) + end * t
        #print d + i, frames[d + i, 1]


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


def blend_between_frames(skeleton, frames, transition_start, transition_end, joint_list, ik_window):
    for c_joint in joint_list:
        idx = skeleton.animated_joints.index(c_joint) * 4 + 3
        j_indices = [idx, idx + 1, idx + 2, idx + 3]
        start_q = frames[transition_start][j_indices]
        end_q = frames[transition_end][j_indices]
        for i in range(ik_window):
            t = float(i) / ik_window
            slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
            frames[transition_start + i][j_indices] = slerp_q


def generated_blend(start_q, end_q, window):
    blend = np.zeros((window, 4))
    for i in range(window):
        t = float(i) / window
        slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
        blend[i] = slerp_q
    return blend


def generate_blended_frames(skeleton, frames, start, end, joint_list, window):
    blended_frames = deepcopy(frames[:])
    for c_joint in joint_list:
        idx = skeleton.animated_joints.index(c_joint) * 4 + 3
        j_indices = [idx, idx + 1, idx + 2, idx + 3]
        start_q = frames[start][j_indices]
        end_q = frames[end][j_indices]
        blended_qs = generated_blend(start_q, end_q, window)
        for fi, q in enumerate(blended_qs):
            blended_frames[start+fi][j_indices] = q
    return blended_frames



def blend_quaternions_to_next_step(skeleton, frames, frame_idx, plant_joint, swing_joint, ik_chains,  window):
    start = frame_idx - window  # end of blending range
    end = frame_idx  # modified frame
    plant_ik_chain = ik_chains[plant_joint]
    swing_ik_chain = ik_chains[swing_joint]
    joint_list = [skeleton.root, "pelvis", plant_ik_chain["root"], plant_ik_chain["joint"], plant_joint, swing_ik_chain["root"], swing_ik_chain["joint"], swing_joint]
    blend_between_frames(skeleton, frames, start, end, joint_list, window)


def interpolate_frames(skeleton, frames_a, frames_b, joint_list, start, end):
    blended_frames = deepcopy(frames_a[:])
    window = end - start
    for joint in joint_list:
        idx = skeleton.animated_joints.index(joint) * 4 + 3
        j_indices = [idx, idx + 1, idx + 2, idx + 3]
        for f in range(window):
            t = (float(f) / window)
            q_a = frames_a[start + f][j_indices]
            q_b = frames_b[start + f][j_indices]
            blended_frames[start + f][j_indices] = quaternion_slerp(q_a, q_b, t, spin=0, shortestpath=True)
    return blended_frames





