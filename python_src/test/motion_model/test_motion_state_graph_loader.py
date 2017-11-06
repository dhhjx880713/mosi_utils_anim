__author__ = 'erhe01'
import sys
import os
ROOTDIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3]) + os.sep
sys.path.append(ROOTDIR)
from morphablegraphs.motion_model.motion_state_graph_loader import MotionStateGraphLoader


def test_motion_primitive_graph_builder():
    
    motion_primitive_graph_path = ROOTDIR+os.sep.join(["..", "test_data", "motion_model", "motion_primitive_graph_dir"])
    transition_model_directory = None
    load_transition_models = False
    motion_primitive_graph_builder = MotionStateGraphLoader()
    motion_primitive_graph_builder.set_data_source(motion_primitive_graph_path, transition_model_directory, load_transition_models)
    motion_primitive_graph = motion_primitive_graph_builder.build()
    motion_primitive_graph.print_information()
    assert ('walk', 'sidestepRight') in list(motion_primitive_graph.nodes.keys())
    assert ("walk", "leftStance") in list(motion_primitive_graph.nodes[("walk", "beginRightStance")].outgoing_edges.keys())
    assert motion_primitive_graph.nodes[("walk", "endLeftStance")].s_pca["n_components"] == 7
    assert motion_primitive_graph.nodes[("walk", "leftStance")].average_step_length > 0.0
    assert motion_primitive_graph.nodes[("walk", "rightStance")].n_standard_transitions == 1
test_motion_primitive_graph_builder()