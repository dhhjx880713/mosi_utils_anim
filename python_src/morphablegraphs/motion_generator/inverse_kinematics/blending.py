from ...external.transformations import quaternion_slerp


def smooth_quaternion_frames_using_slerp(quat_frames, joint_param_indices, event_frame, window):
    h_window = window/2
    start_frame = event_frame-h_window
    end_frame = event_frame+h_window
    new_frames = create_frames_using_slerp(quat_frames, start_frame, event_frame, h_window, joint_param_indices)
    steps = event_frame-start_frame
    for i in range(steps):
        t = float(i)/steps
        quat_frames[start_frame+i, joint_param_indices] = blend_quaternion(quat_frames[start_frame+i, joint_param_indices], new_frames[i], t)
    new_frames = create_frames_using_slerp(quat_frames, event_frame, end_frame, h_window, joint_param_indices)
    steps = end_frame-event_frame
    for i in range(steps):
        t = 1.0-(i/steps)
        quat_frames[event_frame+i, joint_param_indices] = blend_quaternion(quat_frames[start_frame+i, joint_param_indices], new_frames[i], t)


def smooth_quaternion_frames_using_slerp_overwrite_frames(quat_frames, joint_param_indices, event_frame, window):
    h_window = window/2
    start_frame = event_frame-h_window
    end_frame = event_frame+h_window
    apply_slerp(quat_frames, start_frame, event_frame, h_window, joint_param_indices)
    apply_slerp(quat_frames, event_frame, end_frame, h_window, joint_param_indices)


def blend_frames(self, quat_frames, start, end, new_frames, joint_parameter_indices):
    steps = end-start
    for i in range(steps):
        t = i/steps
        quat_frames[start+i, joint_parameter_indices] = blend_quaternion(quat_frames[start+i, joint_parameter_indices], new_frames[i], t)


def create_frames_using_slerp(quat_frames, start_frame, end_frame, steps, joint_parameter_indices):
    start_q = quat_frames[start_frame, joint_parameter_indices]
    end_q = quat_frames[end_frame, joint_parameter_indices]
    frames = []
    for i in xrange(steps):
        t = float(i)/steps
        #nlerp_q = self.nlerp(start_q, end_q, t)
        slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
        #print "slerp",start_q,  end_q, t, slerp_q
        frames.append(slerp_q)
    return frames


def apply_slerp(quat_frames, start_frame, end_frame, steps, joint_parameter_indices):
    start_q = quat_frames[start_frame, joint_parameter_indices]
    end_q = quat_frames[end_frame, joint_parameter_indices]
    for i in xrange(steps):
        t = float(i)/steps
        #nlerp_q = self.nlerp(start_q, end_q, t)
        slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
        #print "slerp",start_q,  end_q, t, slerp_q
        quat_frames[start_frame+i, joint_parameter_indices] = slerp_q


def blend_quaternion(a, b, w):
    return quaternion_slerp(a, b, w, spin=0, shortestpath=True)#a * w + b * (w-1.0)