# encoding: UTF-8
import json
import os

from morphablegraphs.motion_model.motion_state_graph_loader import MotionStateGraphLoader
from morphablegraphs.motion_generator.algorithm_configuration import AlgorithmConfigurationBuilder
from morphablegraphs.motion_generator.elementary_action_generator import ElementaryActionGenerator
import sys
os.chdir('../')

def load_josnfile(jsonfile):
    with open(jsonfile, 'r') as infile:
        jsondata = json.load(infile)
    return jsondata


class GraphDefinitionTester(object):

    def __init__(self, service_config_file):
        self.service_config = load_josnfile(service_config_file)
        self.graph_definition_data = load_josnfile(self.service_config["model_data"] + os.sep + 'graph_definition.json')
        self.motion_primitive_folder = self.service_config["model_data"] + os.sep + 'elementary_action_models'
        self._load_motion_primitive_data()
        self._parse_graph_definition()

    def _load_motion_primitive_data(self):
        self.space_partitioning_files = []
        self.motion_primitive_files = []
        self.metadata_files = []
        for subfolder in os.walk(self.motion_primitive_folder).next()[1]:
            for filename in os.walk(os.path.join(self.motion_primitive_folder, subfolder)).next()[2]:
                if '.pck' in filename:
                    self.space_partitioning_files.append(filename)
                if 'mm.json' in filename:
                    self.motion_primitive_files.append(filename)
                if 'meta' in filename:
                    self.metadata_files.append(subfolder + '_' + filename)

    def _initial_motion_state_graph(self):
        motion_state_graph_loader = MotionStateGraphLoader()
        motion_state_graph_loader.set_data_source(self.service_config["model_data"])
        motion_state_graph = motion_state_graph_loader.build()
        return motion_state_graph

    def _parse_graph_definition(self):
        self.motion_primitives = {}
        for key, values in self.graph_definition_data['transitions'].items():
            counter = 0
            for value in values:
                if "end" not in value.lower():
                    counter += 1
            self.motion_primitives[key] = {'n_transitions': counter,
                                           'transitions': values}

    def check_space_partitioning_files(self):
        for motion_primitive in list(self.motion_primitives.keys()):
            assert motion_primitive + '_quaternion_cluster_tree.pck' in self.space_partitioning_files, \
                (motion_primitive + ' has no space partitioning file! ')
        print("spalce partitioning files are completed! ")

    def check_motion_primitive_files(self):
        for motion_primitive in list(self.motion_primitives.keys()):
            if motion_primitive + '_quaternion_mm.json' not in self.motion_primitive_files:
                print((motion_primitive + '_quaternion_mm.json'))
                print("######################################")
                print((self.motion_primitive_files))
            # assert motion_primitive + '_quaternion_mm.json' in self.motion_primitive_files, \
            #     (motion_primitive + ' has no motion primitive file! ')
        print("motion primitive files are completed! ")

    def check_meta_information(self):
        """
        check the n_standard_transitions written in meta_information files match graph definition file or not
        :return:
        """
        for filename in self.metadata_files:
            segments = filename.split('_')
            subfolder = '_'.join(segments[:3])
            metafilename = '_'.join(segments[3:])
            with open(os.path.join(self.motion_primitive_folder, subfolder, metafilename), 'r') as infile:
                elementary_action = subfolder.split('_')[-1]
                meta_data = json.load(infile)
                for motion_primitive, value in meta_data["stats"].items():
                    n_transitions = value['n_standard_transitions']
                    mm_name = '_'.join([elementary_action, motion_primitive])
                    if n_transitions != self.motion_primitives[mm_name]['n_transitions']:
                        print("################################")
                        print((mm_name + ' has different n_transitions in graph_definition file and meta_information file'))
                        print((mm_name + ' has ' + str(n_transitions) + ' transitions in meta_information file'))
                        print((mm_name + ' has ' + str(self.motion_primitives[mm_name]['n_transitions']) + ' transitions in graph_definition file'))

    def generate_all_transitions(self):
        self.transition_pairs = []
        for key, values in self.motion_primitives.items():
            for transition in values['transitions']:
                new_pair = (key, transition)
                if new_pair not in self.transition_pairs:
                    self.transition_pairs.append(new_pair)

    def check_start_node_for_elementary_actions(self, motion_state_graph):
        """
        Check the start nodes for each elementary action are not empty, and the corresponding motion primitives are in
        data folder
        :param motion_state_graph:
        :return:
        """
        for elementary_action in list(motion_state_graph.node_groups.keys()):
            start_primitive_list = motion_state_graph.get_start_nodes(None, elementary_action)
            assert start_primitive_list != [], ('No start motion primitive for ' + elementary_action)
            for motion_primitive in start_primitive_list:
                motion_primitive_name = elementary_action + '_' + motion_primitive
                assert motion_primitive_name in list(self.motion_primitives.keys()), \
                    (motion_primitive_name + 'is not in motion data folder!')

    def check_graph_walk(self):
        # 1. every action's start and end nodes cannot be empty.
        # 2. there is no starndard nodes for one action, then start and end nodes can be the same
        # 3. end node cannot transfer to the node with the same name
        # 4. if there is only one motion primitive, it must be the end
        motion_state_graph = self._initial_motion_state_graph()
        self.generate_all_transitions()
        self.check_start_node_for_elementary_actions(motion_state_graph)
        for transition_pair in self.transition_pairs:
            for key, node in motion_state_graph.nodes.items():
                if transition_pair[0] == '_'.join(key):
                    type = node.node_type  # three types:  begin, end, standard
                    if type == 'begin':
                        assert transition_pair[0] in motion_state_graph.get_start_nodes(None, key[0]) # check the nodetype is correct or not
                        assert tuple(transition_pair[1].split('_')) in list(node.outgoing_edges.keys()), \
                            ('transition from ' + transition_pair[0] + ' to ' + transition_pair[1] + ' cannot be found in outedges')
                    elif type == 'end':
                        assert transition_pair[0].split('_')[1] in motion_state_graph.node_groups[key[0]].end_states, \
                            ('End state ' + transition_pair[0] + ' is not in ' + key[0] + ' end_states')
                        assert tuple(transition_pair[1].split('_')) in list(node.outgoing_edges.keys()), \
                            ('transition from ' + transition_pair[0] + ' to ' + transition_pair[1] + ' cannot be found in outedges')
                    elif type == 'standard':
                        assert tuple(transition_pair[1].split('_')) in list(node.outgoing_edges.keys()), \
                            ('transition from ' + transition_pair[0] + ' to ' + transition_pair[1] + ' cannot be found in outedges')




if __name__ == '__main__':
    service_file = r'config\service.config'
    tester = GraphDefinitionTester(service_file)
    tester._parse_graph_definition()
    tester.check_graph_walk()
    # tester.check_motion_primitive_files()
    # tester.check_space_partitioning_files()
    # tester.check_meta_information()