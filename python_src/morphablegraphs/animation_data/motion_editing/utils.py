import numpy as np
import math
from matplotlib import pyplot as plt
import json
from scipy.interpolate import UnivariateSpline
from ..utils import euler_to_quaternion
from ...external.transformations import quaternion_multiply, quaternion_inverse, quaternion_matrix, quaternion_from_matrix, euler_from_quaternion
from ..skeleton_node import SkeletonEndSiteNode

LEN_QUATERNION = 4
LEN_TRANSLATION = 3


def normalize(v):
    return v/np.linalg.norm(v)


def quaternion_from_axis_angle(axis, angle):
    q = [1,0,0,0]
    if np.linalg.norm(axis) > 0:
        q[1] = axis[0] * math.sin(angle / 2)
        q[2] = axis[1] * math.sin(angle / 2)
        q[3] = axis[2] * math.sin(angle / 2)
        q[0] = math.cos(angle / 2)
        q = normalize(q)
    return q


def exp_map_to_quaternion(e):
    angle = np.linalg.norm(e)
    axis = e / angle
    q = quaternion_from_axis_angle(axis, angle)
    return q


def convert_exp_frame_to_quat_frame(skeleton, e):
    src_offset = 0
    dest_offset = 0
    n_joints = len(skeleton.animated_joints)
    q = np.zeros(n_joints*4)
    for node in skeleton.animated_joints:
        e_i = e[src_offset:src_offset+3]
        q[dest_offset:dest_offset+4] = exp_map_to_quaternion(e_i)
        src_offset += 3
        dest_offset += 4
    return q


def add_quat_frames(skeleton, q_frame1, q_frame2):
    src_offset = 0
    dest_offset = 3
    new_quat_frame = np.zeros(len(q_frame1))
    new_quat_frame[:3] = q_frame1[:3]
    for node in skeleton.animated_joints:
        new_q = quaternion_multiply(q_frame1[dest_offset:dest_offset + 4], q_frame2[src_offset:src_offset + 4])
        new_quat_frame[dest_offset:dest_offset+4] = new_q
        dest_offset += 4
        src_offset += 4
    return new_quat_frame


def get_3d_rotation_between_vectors(a, b):
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    if s ==0:
        return np.eye(3)
    c = np.dot(a,b)
    v_x = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
    v_x_2 = np.dot(v_x,v_x)
    r = np.eye(3) + v_x + (v_x_2* (1-c/s**2))
    return r

def quaternion_from_vector_to_vector(a, b):
    "src: http://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another"
    v = np.cross(a, b)
    w = np.sqrt((np.linalg.norm(a) ** 2) * (np.linalg.norm(b) ** 2)) + np.dot(a, b)
    q = np.array([w, v[0], v[1], v[2]])
    return q/ np.linalg.norm(q)


def convert_euler_to_quat(euler_frame, joints):
    quat_frame = euler_frame[:3].tolist()
    offset = 3
    step = 3
    for joint in joints:
        e = euler_frame[offset:offset+step]
        q = euler_to_quaternion(e)
        quat_frame += list(q)
        offset += step
    return np.array(quat_frame)


def normalize_quaternion(q):
    return quaternion_inverse(q) / np.dot(q, q)


def get_average_joint_position(skeleton, frames, joint_name, start_frame, end_frame):
    end_frame = min(end_frame, frames.shape[0])
    temp_positions = []
    for idx in xrange(start_frame, end_frame):
        frame = frames[idx]
        pos = skeleton.nodes[joint_name].get_global_position(frame)
        temp_positions.append(pos)
    return np.mean(temp_positions, axis=0)


def get_average_joint_direction(skeleton, frames, joint_name, child_joint_name, start_frame, end_frame,ground_height=0):
    temp_dirs = []
    for idx in xrange(start_frame, end_frame):
        frame = frames[idx]
        pos1 = skeleton.nodes[joint_name].get_global_position(frame)
        pos2 = skeleton.nodes[child_joint_name].get_global_position(frame)
        #pos2[1] = ground_height
        joint_dir = pos2 - pos1
        joint_dir /= np.linalg.norm(joint_dir)
        temp_dirs.append(joint_dir)
    return np.mean(temp_dirs, axis=0)

def get_average_direction_from_target(skeleton, frames, target_pos, child_joint_name, start_frame, end_frame,ground_height=0):
    temp_dirs = []
    for idx in xrange(start_frame, end_frame):
        frame = frames[idx]
        pos2 = skeleton.nodes[child_joint_name].get_global_position(frame)
        pos2[1] = ground_height
        joint_dir = pos2 - target_pos
        joint_dir /= np.linalg.norm(joint_dir)
        temp_dirs.append(joint_dir)
    return np.mean(temp_dirs, axis=0)


def to_local_cos(skeleton, node_name, frame, q):
    # bring into parent coordinate system
    pm = skeleton.nodes[node_name].get_global_matrix(frame)[:3,:3]
    #pm[:3, 3] = [0, 0, 0]
    inv_pm = np.linalg.inv(pm)
    r = quaternion_matrix(q)[:3,:3]
    lr = np.dot(inv_pm, r)[:3,:3]
    q = quaternion_from_matrix(lr)
    return q

def get_dir_on_plane(x, n):
    axb = np.cross(x,n)
    d = np.cross(n, normalize(axb))
    d = normalize(d)
    return d

def project2(x,n):
    """ get direction on plane based on cross product and then project onto the direction """
    d = get_dir_on_plane(x, n)
    return project_on_line(x, d)

def project_vec3(x, n):
    """" project vector on normal of plane and then substract from vector to get projection on plane """
    w = project_on_line(x, n)
    v = x-w
    return v

def project(x, n):
    """ http://www.euclideanspace.com/maths/geometry/elements/plane/lineOnPlane/"""
    l = np.linalg.norm(x)
    a = normalize(x)
    b = normalize(n)
    axb = np.cross(a,b)
    bxaxb = np.cross(b, axb)
    return l * bxaxb

def project_on_line(x, v):
    """https://en.wikipedia.org/wiki/Scalar_projection"""
    s = np.dot(x, v) / np.dot(v, v)
    return s * v

def project_onto_plane(x, n):
    """https://stackoverflow.com/questions/17915475/how-may-i-project-vectors-onto-a-plane-defined-by-its-orthogonal-vector-in-pytho"""
    nl = np.linalg.norm(n)
    d = np.dot(x, n) / nl
    p = [d * normalize(n)[i] for i in range(len(n))]
    return [x[i] - p[i] for i in range(len(x))]


def project_vec_on_plane(vec, n):
    """https://math.stackexchange.com/questions/633181/formula-to-project-a-vector-onto-a-plane"""
    n = normalize(n)
    d = np.dot(vec, n)
    return vec - np.dot(d, n)


def distance_from_point_to_line(p1, p2, vec):
    proj = p2+project_on_line(p1, vec)
    return np.linalg.norm(proj - p1)


def limb_projection(p1, center, n):
    #s1 = np.dot(p1, n) / np.dot(p1, p1)
    #proj_p1 = p1 - s1*n

    #s2 = np.dot(p2, n) / np.dot(p2, p2)
    #proj_p2 = p2 - s2 * n
    proj_p1 = project_vec3(p1, n)
    proj_center = project_vec3(center, n)

    d = np.linalg.norm(proj_p1-proj_center)
    #print proj_p1, proj_p2, s1, s2, proj_p1, proj_p2, d, n
    #print "d", d
    return d


def plot_line(ax, start, end,label=None, color=None):
    x = start[0], end[0]
    y = start[1], end[1]
    ax.plot(x, y, label=label, color=color)

def convert_to_foot_positions(joint_heights):

    n_frames = len(joint_heights.items()[0][1][0])
    print n_frames
    foot_positions = []
    for f in xrange(n_frames):
        foot_positions.append(dict())
    for joint, data in joint_heights.items():
        ps, yv, ya = data
        for frame_idx, p in enumerate(ps):
            foot_positions[frame_idx].update({joint: p})
    return foot_positions

def plot_foot_positions(ax, foot_positions, bodies,step_size=5):
    for f, data in enumerate(foot_positions):
        if f%step_size != 0:
            continue
        for body in [bodies.values()[0] ]:
            start_j = body["start"]
            end_j = body["end"]
            start = f, data[start_j][1]
            end = f+5, data[end_j][1]
            plot_line(ax, start, end, color="k")


def get_vertical_acceleration(skeleton, frames, joint_name):
    """ https://stackoverflow.com/questions/40226357/second-derivative-in-python-scipy-numpy-pandas
    """
    ps = []
    for frame in frames:
        p = skeleton.nodes[joint_name].get_global_position(frame)
        ps.append(p)
    ps = np.array(ps)
    x = np.linspace(0, len(frames), len(frames))
    ys = np.array(ps[:, 1])
    y_spl = UnivariateSpline(x, ys, s=0, k=4)
    velocity = y_spl.derivative(n=1)
    acceleration = y_spl.derivative(n=2)
    return ps, velocity(x), acceleration(x)


def get_joint_height(skeleton, frames, joints):
    joint_heights = dict()
    for joint in joints:
        joint_heights[joint] = get_vertical_acceleration(skeleton, frames, joint)
        # print ys
        # close_to_ground = np.where(y - o - self.ground_height < self.tolerance)

        # zero_crossings = np.where(np.diff(np.sign(ya)))[0]
        # zero_crossings = np.diff(np.sign(ya)) != 0
        # print len(zero_crossings)
        # joint_ground_contacts = np.where(close_to_ground and zero_crossings)
        # print close_to_ground
    return joint_heights

def quaternion_to_axis_angle(q):
    """http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/

    """
    a = 2* math.acos(q[0])
    x = q[1] / math.sqrt(1-q[0]*q[0])
    y = q[2] / math.sqrt(1-q[0]*q[0])
    z = q[3] / math.sqrt(1-q[0]*q[0])
    return normalize([x,y,z]),a

def get_delta_quaternion(q1,q2):
    return quaternion_multiply(quaternion_inverse(q1), q2)


def get_angular_velocity(skeleton, frames, joint):
    """   http://answers.unity3d.com/questions/49082/rotation-quaternion-to-angular-velocity.html
    """
    idx = skeleton.animated_joints.index(joint) * 4 + 3
    angular_velocity = [[0,0,0]]
    prev_q = frames[0, idx:idx + 4]
    for frame_idx, frame in enumerate(frames[1:]):
        q = frames[frame_idx, idx:idx+4]
        q_delta = get_delta_quaternion(prev_q, q)
        axis, angle = quaternion_to_axis_angle(q_delta)
        a = axis * angle
        angular_velocity.append(a)
        prev_q = q
    return np.array(angular_velocity)


def get_angular_velocities(skeleton, frames, joints):
    anglular_velocity = dict()
    for joint in joints:
        anglular_velocity[joint] = get_angular_velocity(skeleton, frames, joint)
    return anglular_velocity


def plot_joint_heights(joint_heights, ground_height=0, frame_range=(None,None)):
    plt.figure(1)
    ax = plt.subplot(111)
    #ax.set_ylim(ymin=-10, ymax=500)
    #ax.set_xlim(xmin=-10, xmax=500)
    n_frames = 0
    for joint, data in joint_heights.items():
        ps, yv, ya = data
        if frame_range == (None, None):
            start, end = 0, len(ps)
        else:
            start, end = frame_range
        n_frames = end- start
        x = np.linspace(start,end, n_frames)
        plt.plot(x, ps[start:end,1], label=joint)
    plot_line(ax, (start, ground_height),(end, ground_height), "ground")
    foot_positions = convert_to_foot_positions(joint_heights)
    bodies = {"left":{"start":"LeftHeel", "end": "LeftToeBase"}, "right":{"start":"RightHeel", "end": "RightToeBase"}}
    #plot_foot_positions(ax, foot_positions, bodies)
    plt.legend()
    plt.show(True)


def plot_angular_velocities(angular_velocities, frame_range=(None,None)):
    plt.figure(1)
    ax = plt.subplot(111)
    #ax.set_ylim(ymin=-10, ymax=500)
    #ax.set_xlim(xmin=-10, xmax=500)
    n_frames = 0
    for joint, data in angular_velocities.items():
        if frame_range == (None, None):
            start, end = 0, len(data)
        else:
            start, end = frame_range
        n_frames = end- start
        x = np.linspace(start,end, n_frames)
        v = map(np.linalg.norm, data[start:end])
        plt.plot(x, np.rad2deg(v), label=joint)
    plt.legend()
    plt.show(True)


def add_heels_to_skeleton(skeleton, left_foot, right_foot, left_heel, right_heel, offset=[0, -5, 0]):
    left_heel_node = SkeletonEndSiteNode(left_heel, [], skeleton.nodes[left_foot])
    left_heel_node.offset = np.array(offset)
    skeleton.nodes[left_heel] = left_heel_node
    skeleton.nodes[left_foot].children.append(left_heel_node)

    right_heel_node = SkeletonEndSiteNode(right_heel, [], skeleton.nodes[right_foot])
    right_heel_node.offset = np.array(offset)
    skeleton.nodes[right_heel] = right_heel_node
    skeleton.nodes[right_foot].children.append(right_heel_node)
    return skeleton

def export_constraints(constraints, file_path):
    unique_dict = dict()
    for frame_idx in constraints:
        for c in constraints[frame_idx]:
            key = tuple(c.position)
            unique_dict[key] = None

    points = []
    for p in unique_dict.keys():
        points.append(p)
    data = dict()
    data["points"] = points
    with open(file_path, "wb") as out:
        json.dump(data, out)

def plot_constraints(constraints, ground_height=0):
    colors ={"RightFoot":"r", "LeftFoot":"g"}
    plt.figure(1)
    joint_constraints = dict()
    ax = plt.subplot(111)

    for frame_idx in constraints:
        for c in constraints[frame_idx]:
            if c.joint_name not in joint_constraints.keys():
                joint_constraints[c.joint_name] = []
            joint_constraints[c.joint_name].append(c.position)
    for joint_name in joint_constraints.keys():
        temp = np.array(joint_constraints[joint_name])
        y = temp[:, 1]
        n_frames = len(y)
        x = np.linspace(0, n_frames, n_frames)

        ax.scatter(x,y, label=joint_name, c=colors[joint_name])

    plot_line(ax, (0, ground_height), (n_frames, ground_height), "ground")
    plt.legend()
    plt.show(True)


def get_random_color():
    random_color = np.random.rand(3, )
    if np.sum(random_color) < 0.5:
        random_color += np.array([0, 0, 1])
    return random_color.tolist()


def convert_ground_contacts_to_annotation(ground_contacts, joints, n_frames):
    data = dict()
    data["color_map"] = {j : get_random_color() for j in joints}
    data["semantic_annotation"] = dict()
    for idx in xrange(n_frames):
        for label in ground_contacts[idx]:
            if label not in data["semantic_annotation"]:
                data["semantic_annotation"][label] = []
            data["semantic_annotation"][label].append(idx)
    return data


def save_ground_contact_annotation(ground_contacts, joints, n_frames, filename):
    data = convert_ground_contacts_to_annotation(ground_contacts, joints, n_frames)
    with open(filename, "wb") as out:
        json.dump(data, out)


def load_ground_contact_annotation(filename, n_frames):
    ground_contacts = [[] for f in xrange(n_frames)]
    with open(filename, "r") as in_file:
        annotation_data = json.load(in_file)
        semantic_annotation = annotation_data["semantic_annotation"]
        for label in semantic_annotation.keys():
            for idx in semantic_annotation[label]:
                ground_contacts[idx].append(label)

    return ground_contacts


def save_ground_contact_annotation_merge_labels(ground_contacts, n_frames, left_foot, right_foot, filename):
    data = dict()
    contact_label = "contact"
    no_contact_label = "no_contact"
    data["color_map"] = {left_foot: [1,0,0],
                         right_foot: [0,1,0],
                         contact_label: [0,0,1],
                         no_contact_label: [1,1,1]}
    data["frame_annotation"] = []
    for idx in xrange(n_frames):
        if left_foot in ground_contacts[idx] and right_foot in ground_contacts[idx]:
            annotation = contact_label
        elif left_foot in ground_contacts[idx]:
            annotation = left_foot
        elif right_foot in ground_contacts[idx]:
            annotation = right_foot
        else:
            annotation = no_contact_label

        data["frame_annotation"].append(annotation)
    with open(filename, "wb") as out:
        json.dump(data, out)

def get_intersection_circle(center1, radius1, center2, radius2):
    """ Calculate the projection on the intersection of two spheres defined by two constraints on the ankles and the leg lengths

        http://mrl.snu.ac.kr/Papers/tog2001.pdf
    """
    delta = center2 - center1
    d = np.linalg.norm(delta)
    c_n = delta / d  # normal

    # radius (eq 27)
    r1_sq = radius1 * radius1
    r2_sq = radius2 * radius2
    d_sq = d * d
    nom = r1_sq - r2_sq + d_sq
    c_r_sq = r1_sq - ((nom * nom) / (4 * d_sq))
    if c_r_sq < 0:  # no intersection
        print "no intersection", c_r_sq
        return
    c_r = math.sqrt(c_r_sq)

    # center (eq 29)
    x = (r1_sq - r2_sq + d_sq) / (2 * d)
    c_c = center1 + x * c_n
    return c_c, c_r, c_n


def get_intersection_circle2(center1, radius1, center2, radius2):
    """ COPIED FROM http://mathworld.wolfram.com/Sphere-SphereIntersection.html
    """
    delta = center2 - center1
    d = np.linalg.norm(delta)
    c_n = delta / d  # normal

    R = radius1
    r = radius2

    # translation of the circle center along the normal (eq 5)
    x = (d*d - r*r + R*R)/(2*d)
    c_c = center1 + x * c_n

    # radius of the sphere (eq 9)
    sq = (-d+r-R)*(-d-r+R)*(-d+r+R)*(d+r+R)
    c_r = (1/(2*d)) * math.sqrt(sq)
    return c_c, c_r, c_n

def project_point_onto_plane(point, point_on_plane, normal):
    """  calculate the projection on the intersection of two spheres defined by two constraints on the ankles and the leg lengths

           http://mrl.snu.ac.kr/Papers/tog2001.pdf
    """
    h = np.dot(np.dot(normal, point_on_plane - point), normal)
    pp = point + h
    return pp

def project_on_intersection_circle(p, center1, radius1, center2, radius2):
    """  calculate the projection on the intersection of two spheres defined by two constraints on the ankles and the leg lengths

        http://mrl.snu.ac.kr/Papers/tog2001.pdf
    """

    c_c, c_r, c_n = get_intersection_circle(center1, radius1, center2, radius2)

    # project root position onto plane on which the circle lies
    pp = project_point_onto_plane(p, c_c, c_n)

    # project projected point onto circle
    delta = normalize(pp - c_c)
    p_c = c_c + delta * c_r

    # set the root position to the projection on the intersection
    print "two constraints - before", p, "after", p_c
    return p_c

#h = np.dot(np.dot(c_n, c_c-p), c_n)
#pp = p+h

def smooth_root_positions(positions, window):
    h_window = int(window/2)
    smoothed_positions = []
    n_pos = len(positions)
    for idx, p in enumerate(positions):
        start = max(idx-h_window, 0)
        end = min(idx + h_window, n_pos)
        #print start, end, positions[start:end]
        avg_p = np.average(positions[start:end], axis=0)
        smoothed_positions.append(avg_p)
    return smoothed_positions


def guess_ground_height(skeleton, frames, start_frame, n_frames, foot_joints):
    minimum_height = np.inf
    joint_heights = get_joint_height(skeleton, frames[start_frame:start_frame+n_frames], foot_joints)
    for joint in joint_heights.keys():
        p, v, a = joint_heights[joint]
        pT = np.array(p).T
        new_min_height = min(pT[1])
        if new_min_height < minimum_height:
            minimum_height = new_min_height
    return minimum_height


def move_to_ground(skeleton, frames, ground_height, foot_joints, start_frame=0, n_frames=5):
    minimum_height = guess_ground_height(skeleton, frames, start_frame, n_frames, foot_joints)
    for f in frames:
        f[:3] += [0, ground_height-minimum_height, 0]
    return frames


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
        #print "one constraint on ", c.joint_name, "- before", root_pos, "after", new_root_pos
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
        #print "no root constraint", t1,t2, r1, r2
        return None
    #print "adapt root for two constraints", t1, t2,  r1, r2

    p_c = project_on_intersection_circle(p, c1, r1, c2, r2)
    p = global_position_to_root_translation(skeleton, frame, root, p_c)
    tframe = np.array(frame)
    tframe[:3] = p
    new_p = skeleton.nodes[root].get_global_position(tframe)
    #print "compare",p_c, new_p
    return p#p_c - offset


def smooth_root_translation_at_end(frames, d, window):
    root_pos = frames[d, :3]
    start_idx = d-window
    start = frames[start_idx, :3]
    end = root_pos
    for i in xrange(window):
        t = float(i) / (window)
        frames[start_idx + i, :3] = start * (1 - t) + end * t


def smooth_root_translation_at_start(frames, d, window):
    start = frames[d, :3]
    start_idx = d+window
    end = frames[start_idx, :3]
    for i in xrange(window):
        t = float(i) / (window)
        frames[d + i, :3] = start * (1 - t) + end * t
