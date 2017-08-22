import os
from morphablegraphs.animation_data import SkeletonBuilder, MotionVector, BVHReader
from morphablegraphs.animation_data.motion_concatenation import align_frames_using_forward_blending, align_and_concatenate_frames
#from morphablegraphs.animation_data.motion_editing.motion_grounding import MotionGrounding
#from morphablegraphs.animation_data.motion_editing.footplant_constraint_generator import FootplantConstraintGenerator
from morphablegraphs.animation_data.motion_editing.motion_primitive_grounding import MotionPrimitiveGrounding
from morphablegraphs.animation_data.motion_editing.utils import add_heels_to_skeleton
from morphablegraphs.animation_data.skeleton_models import GAME_ENGINE_SKELETON_MODEL


def init_motion_primitive_grounding(skeleton_path, skeleton_model):
    bvh = BVHReader(skeleton_path)
    animated_joints = list(bvh.get_animated_joints())
    skeleton = SkeletonBuilder().load_from_bvh(bvh, animated_joints)  # filter here
    skeleton.aligning_root_node = "pelvis"
    skeleton.skeleton_model = skeleton_model
    skeleton = add_heels_to_skeleton(skeleton, skeleton_model["joints"]["left_ankle"],
                                     skeleton_model["joints"]["right_ankle"],
                                     skeleton_model["joints"]["left_heel"],
                                     skeleton_model["joints"]["right_heel"],
                                     skeleton_model["heel_offset"])
    return MotionPrimitiveGrounding(skeleton)

def concatenate(skeleton, m_a, m_b, method, prev_start=0,  window=20):
    prev_frames = m_a.frames
    new_frames = m_b.frames
    print ("a" ,len(prev_frames), "b", len(new_frames))
    start_pose = {"position":[0,0,0], "orientation":[0,0,0]}
    ik_chains = skeleton.skeleton_model["ik_chains"]

    m_concat = MotionVector()
    if method == "be_stupid":
        m_concat.frames = align_frames_using_forward_blending(skeleton, skeleton.aligning_root_node, new_frames, prev_frames, prev_start, start_pose, ik_chains, window)
    else:
        m_concat.frames = align_and_concatenate_frames(skeleton, skeleton.aligning_root_node, new_frames, prev_frames, smoothing_window=window, blending_method=method)
    m_concat.n_frames = len(m_concat.frames)
    return m_concat


def main_single():
    window = 10
    in_dir = r"E:\projects\model_data"
    out_dir = "."
    file_a = in_dir+os.sep+"a.bvh"
    file_b = in_dir+os.sep+"b.bvh"
    skeleton_file = in_dir+os.sep+"skeleton_model.json"
    bvh_a = BVHReader(file_a)
    bvh_b = BVHReader(file_b)
    m_a = MotionVector()
    m_a.from_bvh_reader(bvh_a, False)
    m_b = MotionVector()
    m_b.from_bvh_reader(bvh_b, False)
    #skeleton = SkeletonBuilder().load_from_json_file(skeleton_file)
    animated_joints = list(bvh_a.get_animated_joints())
    skeleton = SkeletonBuilder().load_from_bvh(bvh_a, animated_joints)
    skeleton.animated_joints = list(bvh_a.get_animated_joints())
    skeleton.skeleton_model = GAME_ENGINE_SKELETON_MODEL


    skeleton = SkeletonBuilder().load_from_bvh(bvh_a, animated_joints)  # filter here
    skeleton.skeleton_model = GAME_ENGINE_SKELETON_MODEL
    skeleton.aligning_root_node = "pelvis"
    skeleton = add_heels_to_skeleton(skeleton, skeleton.skeleton_model["joints"]["left_ankle"],
                                     skeleton.skeleton_model["joints"]["right_ankle"],
                                     skeleton.skeleton_model["joints"]["left_heel"],
                                     skeleton.skeleton_model["joints"]["right_heel"],
                                     skeleton.skeleton_model["heel_offset"])
    mg = MotionPrimitiveGrounding(skeleton)

    m_concat = concatenate(skeleton, m_a, m_b, method="smoothing")
    print(m_a.n_frames, m_b.n_frames, m_concat.n_frames)
    m_concat = mg.run_grounding_on_motion_vector(m_concat, "rightStance", step_offset=0, step_length=len(m_a.frames))
    mg.run_grounding_on_motion_vector(m_concat, "leftStance", step_offset=len(m_a.frames), step_length=len(m_b.frames))
    m_concat.export(skeleton, out_dir+os.sep+ "concatenate", False)


    #m_concat = concatenate(skeleton, m_a, m_b, method="be_stupid")
    #m_concat.export(skeleton, out_dir+os.sep+ "concatenate_stupid", False)

    #m_concat = concatenate(skeleton, m_a, m_b, method="slerp2")
    #m_concat.export(skeleton, out_dir+os.sep+ "concatenate_slerp2", False)

    #m_concat = concatenate(skeleton, m_a, m_b, method="slerp")
    #m_concat.export(skeleton, out_dir + os.sep + "concatenate_slerp1", False)

    #m_concat = concatenate(skeleton, m_a, m_b, method="transition", window=window)
    #m_concat.export(skeleton, out_dir + os.sep + "concatenate_transition", False)

def main_loop(n_steps=2, method="smoothing", window = 10):
    in_dir = r"E:\projects\model_data"
    out_dir = "."
    file_a = in_dir+os.sep+"a.bvh"
    file_b = in_dir+os.sep+"b.bvh"
    skeleton_file = in_dir+os.sep+"skeleton_model.json"
    bvh_a = BVHReader(file_a)
    bvh_b = BVHReader(file_b)
    m_a = MotionVector()
    m_a.from_bvh_reader(bvh_a, False)
    m_b = MotionVector()
    m_b.from_bvh_reader(bvh_b, False)
    animated_joints = list(bvh_a.get_animated_joints())
    skeleton = SkeletonBuilder().load_from_bvh(bvh_a, animated_joints)
    skeleton.skeleton_model = GAME_ENGINE_SKELETON_MODEL
    skeleton.aligning_root_node = "pelvis"
    skeleton = add_heels_to_skeleton(skeleton, skeleton.skeleton_model["joints"]["left_ankle"],
                                     skeleton.skeleton_model["joints"]["right_ankle"],
                                     skeleton.skeleton_model["joints"]["left_heel"],
                                     skeleton.skeleton_model["joints"]["right_heel"],
                                     skeleton.skeleton_model["heel_offset"])
    mg = MotionPrimitiveGrounding(skeleton)
    current_step = 1
    steps = [m_a, m_b]
    step_types = ["leftStance","rightStance"]
    step_lengths = [m_a.n_frames, m_b.n_frames]
    motion = m_a
    prev_start = 0
    for i in range(n_steps):
        motion = concatenate(skeleton, motion, steps[current_step], method, prev_start=prev_start, window=window)
        mg.ground_feet(motion, step_types[current_step], step_offset=prev_start, step_length=step_lengths[current_step])
        prev_start += len(steps[current_step].frames)
        print(current_step, motion.n_frames)
        current_step = (current_step + 1) % 2
    motion.export(skeleton, out_dir+os.sep+"concatenate"+method, False)


if __name__ == "__main__":
    #main_single()
    main_loop(15, method="smoothing")
