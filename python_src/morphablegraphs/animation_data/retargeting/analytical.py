"""
Functions for retargeting based on the paper "Using an Intermediate Skeleton and Inverse Kinematics for Motion Retargeting"
by Monzani et al.
See: http://www.vis.uni-stuttgart.de/plain/vdl/vdl_upload/91_35_retargeting%20monzani00using.pdf
"""
import numpy as np
import math
from .constants import OPENGL_UP_AXIS, GAME_ENGINE_SPINE_OFFSET_LIST
from .utils import normalize, align_axis, find_rotation_between_vectors, align_root_translation, to_local_cos, get_quaternion_rotation_by_name, apply_additional_rotation_on_frames
from ...external.transformations import quaternion_matrix, quaternion_multiply


def create_local_cos_map(skeleton, up_vector, x_vector, z_vector, child_map=None):
    joint_cos_map = dict()
    for j in list(skeleton.nodes.keys()):
        joint_cos_map[j] = dict()
        joint_cos_map[j]["y"] = up_vector
        joint_cos_map[j]["x"] = x_vector
        joint_cos_map[j]["z"] = z_vector

        if j == skeleton.root:
            joint_cos_map[j]["x"] = (-np.array(x_vector)).tolist()
        else:
            o = np.array([0, 0, 0])
            if child_map is not None and j in child_map:
                child_name = child_map[j]
                node = skeleton.nodes[child_name]
                o = np.array(node.offset)
            elif len(skeleton.nodes[j].children) > 0:
                node = skeleton.nodes[j].children[0]
                o = np.array(node.offset)
            o = normalize(o)
            if sum(o * o) > 0:
                joint_cos_map[j]["y"] = o
    return joint_cos_map

def align_root_joint(axes, global_src_x_vec, max_iter_count=10):
    # handle special case for the root joint
    # apply only the y axis rotation of the Hip to the Game_engine node
    not_aligned = True
    q = [1, 0, 0, 0]
    iter_count = 0
    while not_aligned:
        qx, axes = align_axis(axes, "x", global_src_x_vec)  # first find rotation to align x axis
        q = quaternion_multiply(qx, q)
        q = normalize(q)
        qy, axes = align_axis(axes, "y", OPENGL_UP_AXIS)  # then add a rotation to let the y axis point up
        q = quaternion_multiply(qy, q)
        q = normalize(q)
        a_y = math.acos(np.dot(axes["y"], OPENGL_UP_AXIS))
        a_x = math.acos(np.dot(axes["x"], global_src_x_vec))
        iter_count += 1
        not_aligned = a_y > 0.1 or a_x > 0.1 and iter_count < max_iter_count
    return q

def align_joint(new_skeleton, free_joint_name, axes, global_src_up_vec,global_src_x_vec, joint_cos_map, spine_offset_list=GAME_ENGINE_SPINE_OFFSET_LIST):
    # first align the bone vectors
    q = [1, 0, 0, 0]
    qy, axes = align_axis(axes, "y", global_src_up_vec)
    q = quaternion_multiply(qy, q)
    q = normalize(q)
    if free_joint_name in spine_offset_list:
        # handle special case of applying the x axis rotation of the Hip to the pelvis
        node = new_skeleton.nodes[free_joint_name]
        t_pose_global_m = node.get_global_matrix(new_skeleton.reference_frame)[:3, :3]
        global_original = np.dot(t_pose_global_m, joint_cos_map[free_joint_name]["y"])
        global_original = normalize(global_original)
        qoffset = find_rotation_between_vectors(OPENGL_UP_AXIS, global_original)
        q = quaternion_multiply(qoffset, q)
        q = normalize(q)

    # then align the twisting angles
    qx, axes = align_axis(axes, "x", global_src_x_vec)
    q = quaternion_multiply(qx, q)
    q = normalize(q)
    return q

def find_rotation_analytically(new_skeleton, free_joint_name, target, frame, joint_cos_map, max_iter_count=10):
    global_src_up_vec = target["global_src_up_vec"]
    global_src_x_vec = target["global_src_x_vec"]

    axes = joint_cos_map[free_joint_name]
    if free_joint_name == new_skeleton.root:
        q = align_root_joint(axes, global_src_x_vec, max_iter_count)
    else:
        q = align_joint(new_skeleton, free_joint_name, axes, global_src_up_vec,global_src_x_vec, joint_cos_map)
    return to_local_cos(new_skeleton, free_joint_name, frame, q)


class Retargeting(object):
    def __init__(self, src_skeleton, target_skeleton, target_to_src_joint_map, scale_factor=1.0, constant_offset=None):#[0,-7.63486613913*(1/0.0881),0]
        self.src_skeleton = src_skeleton
        self.target_skeleton = target_skeleton
        self.target_to_src_joint_map = target_to_src_joint_map
        self.src_to_target_joint_map = {v: k for k, v in list(self.target_to_src_joint_map.items())}
        self.scale_factor = scale_factor
        self.n_params = len(self.target_skeleton.animated_joints) * 4 + 3

        src_child_map = {"RightHand": "Bip01_R_Finger3", "LeftHand": "Bip01_L_Finger3"}
        target_child_map = {"hand_r": "middle_01_r", "hand_l": "middle_01_l"}
        src_cos_map = create_local_cos_map(self.src_skeleton, [1, 0, 0], [0, 1, 0], [0, 0, 1], src_child_map)
        target_cos_map = create_local_cos_map(self.target_skeleton, [0, 1, 0], [1, 0, 0], [0, 0, 1], target_child_map)
        target_cos_map["Game_engine"]["x"] = [1, 0, 0]
        #target_cos_map["spine_01"]["x"] = [1, 0, 0]
        #target_cos_map["spine_03"]["y"] = [0, 0, 1]

        #target_cos_map["spine_03"]["y"] = [0, 1, 0]
        #target_cos_map["spine_03"]["y"] = [-1, 0, 0]
        target_cos_map["spine_01"]["x"] = [-1, 0, 0]
        target_cos_map["spine_02"]["x"] = [-1, 0, 0]
        target_cos_map["spine_03"]["x"] = [-1, 0, 0]
        #target_cos_map["spine_03"]["y"] = [0, 1, 0]

        #target_cos_map["spine_03"]["z"] = [0, 0, -1]

        target_cos_map["neck_01"]["x"] = [-1, 0, 0]

        target_cos_map["hand_r"]["x"] = [0, 0, -1]
        target_cos_map["hand_l"]["x"] = [0, 0, 1]
        target_cos_map["hand_r"]["y"] = [0, 1, 0]
        target_cos_map["hand_l"]["y"] = [0, 1, 0]
        target_cos_map["hand_r"]["z"] = [1, 0, 0]
        target_cos_map["hand_l"]["z"] = [1, 0, 0]

        src_cos_map["RightHand"]["x"] = [0, 1, 0]
        src_cos_map["LeftHand"]["x"] = [0, 1, 0]
        src_cos_map["RightHand"]["y"] = [1, 0, 0]
        src_cos_map["LeftHand"]["y"] = [1, 0, 0]
        #src_cos_map["Spine_1"]["y"] = [1, 0, 0]
        #src_cos_map["Spine_1"]["x"] = [0, -1, 0]


        self.src_cos_map = src_cos_map
        self.target_cos_map = target_cos_map
        self.constant_offset = constant_offset

    def rotate_bone(self, src_name,target_name, src_frame, target_frame, guess):
        q = guess
        src_child_name = self.src_skeleton.nodes[src_name].children[0].node_name # TODO take into account changed targets
        rocketbox_x_axis = self.src_cos_map[src_name]["x"]
        rocketbox_up_axis = self.src_cos_map[src_name]["y"]

        if src_child_name in self.src_to_target_joint_map or target_name =="neck_01" or target_name.startswith("hand"):
            global_m = self.src_skeleton.nodes[src_name].get_global_matrix(src_frame)[:3, :3]
            local_m = self.src_skeleton.nodes[src_name].get_local_matrix(src_frame)[:3, :3]
            global_src_up_vec = normalize(np.dot(global_m, rocketbox_up_axis))
            global_src_x_vec = normalize(np.dot(global_m, rocketbox_x_axis))
            local_src_x_vec = normalize(np.dot(local_m, rocketbox_x_axis))

            target = {"global_src_up_vec": global_src_up_vec, "global_src_x_vec":global_src_x_vec, "local_src_x_vec": local_src_x_vec}
            q = find_rotation_analytically(self.target_skeleton, target_name, target, target_frame, self.target_cos_map)
        return q

    def retarget_frame(self, src_frame, ref_frame):
        #print "apply y offset", self.target_skeleton.nodes["pelvis"].offset[0]
        target_frame = np.zeros(self.n_params)
        # copy the root translation assuming the rocketbox skeleton with static offset on the hips is used as source
        target_frame[0] = src_frame[0] #* self.scale_factor
        target_frame[1] = src_frame[1]#- self.target_skeleton.nodes["pelvis"].offset[0]# * self.scale_factor
        target_frame[2] = src_frame[2] #* self.scale_factor

        if self.constant_offset is not None:
            target_frame[:3] += self.constant_offset
        animated_joints = self.target_skeleton.animated_joints
        target_offset = 3

        for target_name in animated_joints:
            q = get_quaternion_rotation_by_name(target_name, self.target_skeleton.reference_frame, self.target_skeleton, root_offset=3)
            if target_name in list(self.target_to_src_joint_map.keys()) or target_name == "Game_engine":
                if target_name != "Game_engine":  # special case for splitting the rotation onto two joints
                    src_name = self.target_to_src_joint_map[target_name]
                else:
                    src_name = "Hips"
                q = self.rotate_bone(src_name, target_name, src_frame, target_frame, q)

            if ref_frame is not None:
                #  align quaternion to the reference frame to allow interpolation
                #  http://physicsforgames.blogspot.de/2010/02/quaternions.html
                ref_q = ref_frame[target_offset:target_offset + 4]
                if np.dot(ref_q, q) < 0:
                    q = -q
            target_frame[target_offset:target_offset + 4] = q
            target_offset += 4

        # apply offset on the root taking the orientation into account
        q = target_frame[3:7]
        m = quaternion_matrix(q)[:3, :3]
        target_frame[:3] -= np.dot(m, self.target_skeleton.nodes["Root"].offset)
        target_frame = align_root_translation(self.target_skeleton, target_frame, src_frame, "pelvis")
        return target_frame

    def run(self, src_frames, frame_range, additional_rotation_map=None):
        #TODO get up axes and cross vector from skeleton heuristically, bone_dir = up, left leg to right leg = cross for all bones
        if frame_range is None:
            frame_range = (0, len(src_frames))
        #if additional_rotation_map is not None:
        #    src_frames = apply_additional_rotation_on_frames(src_skeleton.animated_joints, src_frames, additional_rotation_map)
        target_frames = []
        ref_frame = None
        print(frame_range)
        for idx, src_frame in enumerate(src_frames[frame_range[0]:frame_range[1]]):
            target_frame = self.retarget_frame(src_frame, ref_frame)
            if ref_frame is None:
                ref_frame = target_frame
            target_frames.append(target_frame)
        target_frames = np.array(target_frames)
        return target_frames


def retarget_from_src_to_target(src_skeleton, target_skeleton, src_frames, target_to_src_joint_map, additional_rotation_map=None, scale_factor=1.0, frame_range=None):
    retargeting = Retargeting(src_skeleton, target_skeleton, target_to_src_joint_map, scale_factor)
    return retargeting.run(src_frames, frame_range, additional_rotation_map)
