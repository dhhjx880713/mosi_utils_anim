__author__ = 'erhe01'
import sys
import os
ROOTDIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-4]) + os.sep
sys.path.append(ROOTDIR)
import numpy as np
from morphablegraphs.animation_data.bvh import BVHReader
from morphablegraphs.animation_data.skeleton import Skeleton
from morphablegraphs.motion_generator.constraint.elementary_action_constraints_builder import ElementaryActionConstraintsBuilder
from morphablegraphs.motion_generator.constraint.motion_primitive_constraints_builder import MotionPrimitiveConstraintsBuilder
from morphablegraphs.motion_model.motion_primitive_node_group_builder import MotionPrimitiveNodeGroupBuilder
from morphablegraphs.motion_model.motion_primitive_graph import MotionPrimitiveGraph
from morphablegraphs.motion_generator.algorithm_configuration import AlgorithmConfigurationBuilder
from morphablegraphs.utilities.io_helper_functions import load_json_file
from morphablegraphs.animation_data.motion_editing import convert_euler_frames_to_quaternion_frames

def get_motion_primitive_graph(skeleton, elementary_action_name, morphable_model_directory):
    motion_primitive_graph = MotionPrimitiveGraph()
    motion_primitive_graph.skeleton = skeleton
    node_group_builder = MotionPrimitiveNodeGroupBuilder()
    node_group_builder.set_properties(False, False)

    node_group_builder.set_data_source(elementary_action_name, morphable_model_directory, subgraph_desc=None, graph_definition=None)
    walk_group = node_group_builder.build()
    motion_primitive_graph.nodes = walk_group.nodes
    motion_primitive_graph.node_groups = {elementary_action_name: walk_group}
    for keys in motion_primitive_graph.node_groups.keys():
            motion_primitive_graph.node_groups[keys].update_attributes(update_stats=False)
    return motion_primitive_graph

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
    motion_primitive_name = "beginRightStance"
    motion_primitive_constraints_builder.set_status(motion_primitive_name, last_arc_length=0, prev_frames=None, is_last_step=False)
    mp_constraints = motion_primitive_constraints_builder.build()
    return mp_constraints



def test_motion_primitive_constraints():
    """ Tests the expected sum of  errors from the position and direction constraints that were extracted from the trajectory"""
    expected_error = 10.209201069
    bvh_file_path = ROOTDIR+os.sep.join(["..", "test_data", "motion_generator", "one_step_walk", "MGResult.bvh"])
    mg_input_file = ROOTDIR+os.sep.join(["..", "test_data", "motion_generator", "one_step_walk", "MGresult.json"])
    morphable_model_directory = ROOTDIR+os.sep.join(["..", "test_data", "motion_model", "elementary_action_walk_dir"])
    bvh_reader = BVHReader(bvh_file_path)
    skeleton = Skeleton(bvh_reader)
    quat_frames = convert_euler_frames_to_quaternion_frames(bvh_reader, bvh_reader.frames)
    motion_primitive_constraints = get_motion_primitive_constraints_for_first_step(mg_input_file, skeleton, morphable_model_directory)
    error_sum = 0
    for c in motion_primitive_constraints.constraints:
        error_sum += c.weight_factor * c.evaluate_motion_sample(quat_frames)
    print "error", error_sum, "expected error", expected_error
    assert np.isclose(error_sum, expected_error)
