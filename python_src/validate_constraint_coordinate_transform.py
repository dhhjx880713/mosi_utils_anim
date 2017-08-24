import os
import numpy as np
from python_src.morphablegraphs import DEFAULT_ALGORITHM_CONFIG, AnnotatedMotionVector
from python_src.morphablegraphs.animation_data import SkeletonBuilder
from python_src.morphablegraphs.motion_model.motion_state import MotionPrimitiveModelWrapper
from python_src.morphablegraphs.motion_generator.motion_primitive_generator import MotionPrimitiveGenerator
from python_src.morphablegraphs.utilities import load_json_file, write_to_json_file, set_log_mode, LOG_MODE_DEBUG
from python_src.morphablegraphs.constraints.elementary_action_constraints import ElementaryActionConstraints
from python_src.morphablegraphs.constraints.motion_primitive_constraints import MotionPrimitiveConstraints
from python_src.morphablegraphs.constraints.spatial_constraints import GlobalTransformConstraint
from python_src.morphablegraphs.animation_data.motion_concatenation import get_node_aligning_2d_transform


class MockGraph(object):
    def __init__(self, skeleton):
        self.skeleton = skeleton


def load_skeleton(filename):
    data = load_json_file(filename)
    return SkeletonBuilder().load_from_json_data(data)


def load_mp(filename, skeleton):
    data = load_json_file(filename)
    motion_primitive = MotionPrimitiveModelWrapper()
    motion_primitive.cluster_tree = None
    motion_primitive._initialize_from_json(skeleton.convert_to_mgrd_skeleton(), data)
    return motion_primitive


def generate_constraints(skeleton, joint_name, frame_idx, position, n_frames):
    mp_constraints = MotionPrimitiveConstraints()
    mp_constraints.skeleton = skeleton
    c_desc = {"joint": joint_name, "canonical_keyframe": frame_idx, "position": position,
              "n_canonical_frames": n_frames, "semanticAnnotation": {"keyframeLabel": "none"}}
    c = GlobalTransformConstraint(skeleton, c_desc, 1.0, 1.0)
    mp_constraints.constraints.append(c)
    return mp_constraints


def get_random_position_from_mp(motion_primitive, skeleton, joint_name, frame_idx):
    spline = motion_primitive.sample(use_time=False)
    frames = spline.get_motion_vector()
    return skeleton.nodes[joint_name].get_global_position(frames[frame_idx])


def generate_step(action_constraints, motion_primitive, mp_constraints, algorithm_config, prev_frames=None):
    mp_generator = MotionPrimitiveGenerator(action_constraints, algorithm_config)
    vector = mp_generator.generate_constrained_sample(motion_primitive, mp_constraints, prev_frames=prev_frames)
    spline = motion_primitive.back_project(vector, use_time_parameters=False)
    return spline.get_motion_vector()


def generate_random_constraint(motion_primitive, skeleton, joint_name, frame_idx, aligning_transform):
    position = get_random_position_from_mp(motion_primitive, skeleton, joint_name, frame_idx)
    position = list(position) + [1]
    position = np.dot(aligning_transform, position)
    print("global_position", position)
    return generate_constraints(skeleton, joint_name, frame_idx, position, motion_primitive.get_n_canonical_frames())


def get_aligning_transform(motion_primitive, skeleton, prev_frames=None):
    if prev_frames is None:
        return np.eye(4)
    sample = motion_primitive.sample(False).get_motion_vector()
    return get_node_aligning_2d_transform(skeleton, skeleton.aligning_root_node, prev_frames, sample)


def generate_motion_from_mps(skeleton_filename, mp_filenames, joint_names, n_steps=2, use_local_coordinates=False):
    algorithm_config = DEFAULT_ALGORITHM_CONFIG
    algorithm_config["use_local_coordinates"] = use_local_coordinates
    skeleton = load_skeleton(skeleton_filename)
    n_mps = len(mp_filenames)
    mps = dict()
    for filename in mp_filenames:
        mps[filename] = load_mp(filename, skeleton)
    mock_graph = MockGraph(skeleton)
    action_constraints = ElementaryActionConstraints()
    action_constraints.motion_state_graph = mock_graph
    mv = AnnotatedMotionVector(skeleton=skeleton)
    current_step = 0
    positions = []
    for n in range(n_steps):
        mp_name = mp_filenames[current_step]
        mp = mps[mp_name]
        joint_name = joint_names[current_step]
        n_mp_frames = int(mp.get_n_canonical_frames())
        frame_idx = n_mp_frames-1
        aligning_transform = get_aligning_transform(mp, skeleton, mv.frames)
        mp_constraints = generate_random_constraint(mp, skeleton, joint_name, frame_idx, aligning_transform)
        p = mp_constraints.constraints[0].position
        positions.append(list(p))
        mp_constraints.aligning_transform = aligning_transform
        if use_local_coordinates:
            local_constraints = mp_constraints.transform_constraints_to_local_cos()
            frames = generate_step(action_constraints, mp, local_constraints, algorithm_config)
        else:
            frames = generate_step(action_constraints, mp, mp_constraints, algorithm_config, mv.frames)
        mv.append_frames(frames)
        current_step = (current_step + 1) % n_mps
    mv.frames = skeleton.add_fixed_joint_parameters_to_motion(mv.frames)
    return mv, positions


def main():
    set_log_mode(LOG_MODE_DEBUG)
    in_dir = r"E:\projects\model_data"
    skeleton_filename = in_dir + os.sep + "skeleton_model.json"
    mp_filenames = [in_dir + os.sep + "walk_leftStance_quaternion_mm.mp",
                    in_dir + os.sep + "walk_rightStance_quaternion_mm.mp"]
    joint_names = ["Root", "Root"]
    n_steps = 4
    np.random.seed(100)
    use_local_coordinates = False
    mv, positions = generate_motion_from_mps(skeleton_filename, mp_filenames, joint_names, n_steps, use_local_coordinates)
    mv.export(in_dir + os.sep + "out")
    data = {"points": positions}
    write_to_json_file(in_dir + os.sep + "outconstraints.data", data)

if __name__ == "__main__":
    main()
