import os
import numpy as np
from python_src.morphablegraphs import DEFAULT_ALGORITHM_CONFIG, AnnotatedMotionVector
from python_src.morphablegraphs.animation_data import SkeletonBuilder
from python_src.morphablegraphs.motion_model.motion_state import MotionPrimitiveModelWrapper
from python_src.morphablegraphs.motion_generator.motion_primitive_generator import MotionPrimitiveGenerator
from python_src.morphablegraphs.utilities import load_json_file, write_to_json_file, set_log_mode, LOG_MODE_DEBUG
from python_src.morphablegraphs.constraints.elementary_action_constraints import ElementaryActionConstraints
from python_src.morphablegraphs.constraints.motion_primitive_constraints import MotionPrimitiveConstraints
from python_src.morphablegraphs.constraints.spatial_constraints import GlobalTransformConstraint, Direction2DConstraint
from python_src.morphablegraphs.constraints.spatial_constraints.splines import ParameterizedSpline
from python_src.morphablegraphs.animation_data.motion_concatenation import get_node_aligning_2d_transform
from python_src.morphablegraphs.external.transformations import euler_matrix, quaternion_matrix


FOOT_OFFSETS = dict()
FOOT_OFFSETS["foot_l"] = np.array([-18,0,0])
FOOT_OFFSETS["foot_r"] = np.array([18,0,0])


def quaternion_from_vector_to_vector(a, b):
    "src: http://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another"
    v = np.cross(a, b)
    w = np.sqrt((np.linalg.norm(a) ** 2) * (np.linalg.norm(b) ** 2)) + np.dot(a, b)
    q = np.array([w, v[0], v[1], v[2]])
    return q / np.linalg.norm(q)


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

def generate_dir_constraint(skeleton, joint_name, frame_idx, dir_vector, n_frames):
    mp_constraints = MotionPrimitiveConstraints()
    mp_constraints.skeleton = skeleton
    c_desc = {"joint": joint_name, "canonical_keyframe": frame_idx, "dir_vector": dir_vector,
              "n_canonical_frames": n_frames, "semanticAnnotation": {"keyframeLabel": "none"}}
    c = Direction2DConstraint(skeleton, c_desc, 1.0, 1.0)
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



def generate_constraint_from_trajectory(motion_primitive, skeleton, joint_name, frame_idx, trajectory, arc_length, generate_foot_step_constraints=False):
    REF_VECTOR = [0.0, 0.0, -1.0]
    offset = FOOT_OFFSETS[joint_name]
    goal, dir_vector = trajectory.get_tangent_at_arc_length(arc_length)
    q = quaternion_from_vector_to_vector(REF_VECTOR, dir_vector)
    m = quaternion_matrix(q)[:3,:3]
    offset = np.dot(m, offset)[:3]
    position = goal + offset
    print("global_position from trajectory", position)
    mp_constraints1 = generate_constraints(skeleton, joint_name, frame_idx, position, motion_primitive.get_n_canonical_frames())
    mp_constraints2 = generate_dir_constraint(skeleton, "Root", frame_idx, goal, motion_primitive.get_n_canonical_frames())
    mp_constraints1.constraints += mp_constraints2.constraints
    return mp_constraints1


def get_aligning_transform(motion_primitive, skeleton, prev_frames=None):
    if prev_frames is None:
        return np.eye(4)
    sample = motion_primitive.sample(False).get_motion_vector()
    return get_node_aligning_2d_transform(skeleton, skeleton.aligning_root_node, prev_frames, sample)


def generate_random_points(start, n_points, step_length=50.0):
    random_points = [start]
    angle = 0.0
    for i in range(n_points):
        direction_vector = [0.0,0.0,-1.0, 1.0]
        delta = np.random.uniform(low=-1.0, high=1.0)
        angle += delta * np.pi/4.0
        e = [0.0, angle, 0.0]
        direction_vector = np.dot(euler_matrix(*e), direction_vector)[:3]
        print(direction_vector)
        direction_vector /= np.linalg.norm(direction_vector)
        point = random_points[-1] + direction_vector * step_length
        random_points.append(point.tolist())
    print(random_points)
    return random_points


in_dir = r"E:\projects\model_data"


def generate_motion_from_random_constraints(skeleton_filename, mp_filenames, joint_names, n_steps=2, use_local_coordinates=False):
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


def generate_motion_from_random_trajectory(skeleton_filename, mp_filenames, joint_names, n_steps=2, use_local_coordinates=False, step_length=50.0, generate_foot_step_constraints=False):
    algorithm_config = DEFAULT_ALGORITHM_CONFIG
    start_position = [0,0,0]
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
    arc_length = 0.0
    random_points = generate_random_points(start_position, n_steps, step_length)
    trajectory = ParameterizedSpline(random_points)
    data = {"points": random_points}
    write_to_json_file(in_dir + os.sep + "random_points.data", data)

    positions = []
    full_arc_length = trajectory.get_full_arc_length()
    while arc_length < full_arc_length:
        mp_name = mp_filenames[current_step]
        mp = mps[mp_name]
        joint_name = joint_names[current_step]
        n_mp_frames = int(mp.get_n_canonical_frames())
        frame_idx = n_mp_frames-1
        aligning_transform = get_aligning_transform(mp, skeleton, mv.frames)
        mp_constraints = generate_constraint_from_trajectory(mp, skeleton, joint_name, frame_idx, trajectory, arc_length+step_length, generate_foot_step_constraints)
        p = mp_constraints.constraints[0].position
        positions.append(list(p))
        mp_constraints.aligning_transform = aligning_transform
        if use_local_coordinates:
            local_constraints = mp_constraints.transform_constraints_to_local_cos()
            frames = generate_step(action_constraints, mp, local_constraints, algorithm_config)
        else:
            frames = generate_step(action_constraints, mp, mp_constraints, algorithm_config, mv.frames)
        arc_length += np.linalg.norm(frames[-1][:3])
        mv.append_frames(frames)
        current_step = (current_step + 1) % n_mps
    mv.frames = skeleton.add_fixed_joint_parameters_to_motion(mv.frames)
    return mv, positions

def main():
    set_log_mode(LOG_MODE_DEBUG)
    skeleton_filename = in_dir + os.sep + "skeleton_model.json"
    mp_filenames = [in_dir + os.sep + "walk_leftStance_quaternion_mm.mp",
                    in_dir + os.sep + "walk_rightStance_quaternion_mm.mp"]
    joint_names = ["foot_l", "foot_r"]
    n_steps = 5
    np.random.seed(100)
    step_length = 40.0
    use_trajectory = True
    use_local_coordinates = True
    generate_foot_step_constraints = True
    if use_trajectory:
        mv, positions = generate_motion_from_random_trajectory(skeleton_filename, mp_filenames, joint_names, n_steps, use_local_coordinates,step_length, generate_foot_step_constraints)
    else:
        mv, positions = generate_motion_from_random_constraints(skeleton_filename, mp_filenames, joint_names, n_steps, use_local_coordinates)
    mv.export(in_dir + os.sep + "out")
    data = {"points": positions}
    write_to_json_file(in_dir + os.sep + "outconstraints.data", data)

if __name__ == "__main__":
    main()
