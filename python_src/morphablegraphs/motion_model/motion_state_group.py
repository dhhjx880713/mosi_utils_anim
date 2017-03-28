# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:57:42 2015

@author: erhe01
"""

from . import NODE_TYPE_START, NODE_TYPE_STANDARD, NODE_TYPE_END, NODE_TYPE_SINGLE,  NODE_TYPE_CYCLE_END
from elementary_action_meta_info import ElementaryActionMetaInfo
from ..utilities import write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_INFO, LOG_MODE_ERROR


class MotionStateGroup(ElementaryActionMetaInfo):
    """ Contains the motion primitives of an elementary action as nodes.
    """
    def __init__(self, elementary_action_name, elementary_action_directory, motion_state_graph):
        super(MotionStateGroup, self).__init__(elementary_action_name, elementary_action_directory)
        self.motion_state_graph = motion_state_graph
        self.nodes = dict()
        self.has_transition_models = False
        self.loaded_from_dict = elementary_action_directory is None

    def set_meta_information(self, meta_information=None):
        super(MotionStateGroup, self).set_meta_information(meta_information)
        self._set_node_attributes()

    def _set_node_attributes(self):
        write_message_to_log("elementary_action" + str(self.elementary_action_name), LOG_MODE_DEBUG)
        write_message_to_log("start states" + str(self.start_states), LOG_MODE_DEBUG)
        if len(self.nodes) == 1:
            node_key = self.nodes.keys()[0]
            self.nodes[node_key].node_type = NODE_TYPE_SINGLE
        else:
            for k in self.start_states:
                self.nodes[(self.elementary_action_name, k)].node_type = NODE_TYPE_START
            write_message_to_log("end states" + str(self.end_states), LOG_MODE_DEBUG)
            for k in self.end_states:
                 self.nodes[(self.elementary_action_name, k)].node_type = NODE_TYPE_END
            for k in self.cycle_states:
                 self.nodes[(self.elementary_action_name, k)].node_type = NODE_TYPE_CYCLE_END

    def _update_motion_state_stats(self, recalculate=False):
        """  Update stats of motion states for faster lookup.
        """
        changed_meta_info = False
        if recalculate:
            changed_meta_info = True
            self.meta_information["stats"] = {}
            for node_key in self.nodes.keys():
                self.nodes[node_key].update_motion_stats()
                self.meta_information["stats"][node_key[1]] = {"average_step_length": self.nodes[node_key].average_step_length,
                                                               "n_standard_transitions": self.nodes[node_key].n_standard_transitions}
                write_message_to_log("n standard transitions " + str(node_key) +" "+ str(self.nodes[node_key].n_standard_transitions), LOG_MODE_DEBUG)
            write_message_to_log("Updated meta information " + str(self.meta_information), LOG_MODE_DEBUG)
        else:
            if self.meta_information is None:
                self.meta_information = {}
            if "stats" not in self.meta_information.keys():
                self.meta_information["stats"] = {}
            for node_key in self.nodes.keys():
                if node_key[1] in self.meta_information["stats"].keys():
                    self.nodes[node_key].n_standard_transitions = self.meta_information["stats"][node_key[1]]["n_standard_transitions"]
                    self.nodes[node_key].average_step_length = self.meta_information["stats"][node_key[1]]["average_step_length"]
                else:
                    self.nodes[node_key].update_motion_stats()
                    self.meta_information["stats"][node_key[1]] = {"average_step_length": self.nodes[node_key].average_step_length,
                                                                   "n_standard_transitions": self.nodes[node_key].n_standard_transitions }
                    changed_meta_info = True
            write_message_to_log("Loaded stats from meta information file " + str(self.meta_information), LOG_MODE_DEBUG)
        if changed_meta_info and not self.loaded_from_dict:
            self.save_updated_meta_info()

    def generate_next_parameters(self, current_node_key, current_parameters, to_node_key, use_transition_model):
        """ Generate parameters for transitions.
        
        Parameters
        ----------
        * current_state: string
        \tName of the current motion primitive
        * current_parameters: np.ndarray
        \tParameters of the current state
        * to_node_key: tuple
        \t Identitfier of the action and motion primitive we want to transition to.
        \t Should have the format (action name, motionprimitive name)
        * use_transition_model: bool
        \t flag to set whether a prediction from the transition model should be made or not.
        """
        assert to_node_key[0] == self.elementary_action_name
        if self.has_transition_models and use_transition_model:
            print "use transition model", current_node_key, to_node_key
            next_parameters = self.nodes[current_node_key].predict_parameters(to_node_key, current_parameters)
        else:
            next_parameters = self.nodes[to_node_key].sample_low_dimensional_vector()
        return next_parameters

    def get_transition_type_for_action_from_trajectory(self, graph_walk, action_constraint, travelled_arc_length, arc_length_of_end):

        #test end condition for trajectory constraints
        if not action_constraint.check_end_condition(graph_walk.get_quat_frames(),\
                                travelled_arc_length, arc_length_of_end):

            #make standard transition to go on with trajectory following
            next_node_type = NODE_TYPE_STANDARD
        else:
            # threshold was overstepped. remove previous step before
            # trying to reach the goal using a last step
            next_node_type = NODE_TYPE_END

        write_message_to_log("Generate "+ str(next_node_type) + " transition from trajectory", LOG_MODE_DEBUG)
        return next_node_type

    def get_transition_type_for_action(self, graph_walk, action_constraint):
        prev_node = graph_walk.steps[-1].node_key
        n_standard_transitions = len(self.get_n_standard_transitions(prev_node))
        if n_standard_transitions > 0:
            next_node_type = NODE_TYPE_STANDARD
        else:
            next_node_type = NODE_TYPE_END
        write_message_to_log("Generate " + str(next_node_type) + " transition without trajectory " + str(n_standard_transitions) +" " + str(prev_node), LOG_MODE_DEBUG)
        if action_constraint.cycled_next and next_node_type == NODE_TYPE_END:
            next_node_type = NODE_TYPE_CYCLE_END
        return next_node_type

    def get_n_standard_transitions(self, prev_node):
        return [e for e in self.nodes[prev_node].outgoing_edges.keys()
                if self.nodes[prev_node].outgoing_edges[e].transition_type == NODE_TYPE_STANDARD]

    def get_random_transition(self, graph_walk, action_constraint, travelled_arc_length, arc_length_of_end):
        """ Get next state of the elementary action based on previous iteration.
        """
        prev_node = graph_walk.steps[-1].node_key
        if action_constraint.root_trajectory is None:
            next_node_type = self.get_transition_type_for_action(graph_walk, action_constraint)
        else:
            next_node_type = self.get_transition_type_for_action_from_trajectory(graph_walk, action_constraint, travelled_arc_length, arc_length_of_end)

        to_node_key = self.nodes[prev_node].generate_random_transition(next_node_type)
        if to_node_key is not None:
            return to_node_key, next_node_type
        else:
            return None, next_node_type

    def generate_random_walk(self, state_node, number_of_steps, use_transition_model=True):
        """ Generates a random graph walk to be converted into a BVH file

        Parameters
        ----------
        * start_state: string
        \tInitial state.
        * number_of_steps: integer
        \tNumber of transitions
        * use_transition_model: bool
        \tSets whether or not the transition model should be used in parameter prediction
        """
        current_node = state_node
        assert current_node in self.nodes.keys()
        graph_walk = []
        count = 0
        #print "start", current_node
        current_parameters = self.nodes[current_node].sample_low_dimensional_vector()
        entry = {"node_key": current_node, "parameters": current_parameters}
        graph_walk.append(entry)

        if self.nodes[current_node].n_standard_transitions > 0:
            while count < number_of_steps:
                to_node_key = self.nodes[current_node].generate_random_transition(NODE_TYPE_STANDARD)
                next_parameters = self.generate_next_parameters(current_node, current_parameters, to_node_key, use_transition_model)
                entry = {"node_key": to_node_key, "parameters": next_parameters}
                graph_walk.append(entry)
                current_parameters = next_parameters
                current_node = to_node_key
                count += 1
        #add end node
        to_node_key = self.nodes[current_node].generate_random_transition(NODE_TYPE_END)
        next_parameters = self.generate_next_parameters(current_node, current_parameters, to_node_key, use_transition_model)
        entry = {"node_key": to_node_key, "parameters": next_parameters}
        graph_walk.append(entry)
        return graph_walk

    def has_cycle_states(self):
        return len(self.cycle_states)> 0