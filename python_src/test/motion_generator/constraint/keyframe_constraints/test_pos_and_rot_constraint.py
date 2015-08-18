import sys
import os
ROOTDIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-5]) + os.sep
sys.path.append(ROOTDIR)
import numpy as np
from morphablegraphs.animation_data.bvh import BVHReader
from morphablegraphs.animation_data.skeleton import Skeleton
from morphablegraphs.motion_generator.constraint.keyframe_constraints.pos_and_rot_constraint import PositionAndRotationConstraint
from morphablegraphs.motion_generator.constraint.elementary_action_constraints_builder import ElementaryActionConstraintsBuilder
from morphablegraphs.motion_generator.constraint.motion_primitive_constraints_builder import MotionPrimitiveConstraintsBuilder
from morphablegraphs.motion_model.motion_primitive_node_group_builder import MotionPrimitiveNodeGroupBuilder
from morphablegraphs.motion_model.motion_primitive_graph import MotionPrimitiveGraph
from morphablegraphs.motion_generator.algorithm_configuration import AlgorithmConfigurationBuilder
from morphablegraphs.animation_data.motion_editing import convert_euler_frames_to_quaternion_frames
from morphablegraphs.utilities.io_helper_functions import load_json_file


def get_motion_primitive_graph(skeleton, elementary_action_name, morphable_model_directory):
    motion_primitive_graph_dummy = MotionPrimitiveGraph()
    motion_primitive_graph_dummy.skeleton = skeleton
    node_group_builder = MotionPrimitiveNodeGroupBuilder()
    node_group_builder.set_properties(False, False)

    node_group_builder.set_data_source(elementary_action_name, morphable_model_directory, subgraph_desc=None, graph_definition=None)
    walk_group = node_group_builder.build()
    motion_primitive_graph_dummy.nodes = walk_group.nodes
    motion_primitive_graph_dummy.node_groups = {elementary_action_name: walk_group}
    return motion_primitive_graph_dummy


def get_motion_primitive_constraints_for_first_step(mg_input_file, skeleton, morphable_model_directory):
    mg_input = load_json_file(mg_input_file)
    algorithm_config = AlgorithmConfigurationBuilder().build()
    elementary_action_name = "walk"
    motion_primitive_graph_dummy = get_motion_primitive_graph(skeleton, elementary_action_name, morphable_model_directory)
    elmentary_action_constraints_builder = ElementaryActionConstraintsBuilder(mg_input, motion_primitive_graph_dummy)
    elementary_action_constraints = elmentary_action_constraints_builder.get_next_elementary_action_constraints()
    motion_primitive_constraints_builder = MotionPrimitiveConstraintsBuilder()
    motion_primitive_constraints_builder.set_algorithm_config(algorithm_config)
    motion_primitive_constraints_builder.set_action_constraints(elementary_action_constraints)
    motion_primitive_name = "beginLeftStance"
    motion_primitive_constraints_builder.set_status(motion_primitive_name, last_arc_length=0, prev_frames=None, is_last_step=False)
    mp_constraints = motion_primitive_constraints_builder.build()
    return mp_constraints#constraint_desc PositionAndRotationConstraint(skeleton, constraint_desc, precision=1.0, weight_factor=1.0)


def test_pos_and_rot_constraint():
    bvh_file_path = ROOTDIR+os.sep.join(["..", "test_data", "motion_generator", "one_step_walk", "MGResult.bvh"])#walk_001_1_leftStance_43_86.bvh
    mg_input_file = ROOTDIR+os.sep.join(["..", "test_data", "motion_generator", "one_step_walk", "MGresult.json"])#ROOTDIR+os.sep.join(["..", "test_data", "motion_generator", "mg_input.json"])
    morphable_model_directory = ROOTDIR+os.sep.join(["..", "test_data", "motion_model", "elementary_action_walk_dir"])
    bvh_reader = BVHReader(bvh_file_path)
    skeleton = Skeleton(bvh_reader)
    quat_frames = convert_euler_frames_to_quaternion_frames(bvh_reader, bvh_reader.frames)
    motion_primitive_constraints = get_motion_primitive_constraints_for_first_step(mg_input_file, skeleton, morphable_model_directory)
    pos_and_rot_constraint = motion_primitive_constraints.constraints[0]
    error = pos_and_rot_constraint.evaluate_motion_sample(quat_frames)
    print error
    assert np.isclose(error, 3.84472345971)#574.957480905

test_pos_and_rot_constraint()