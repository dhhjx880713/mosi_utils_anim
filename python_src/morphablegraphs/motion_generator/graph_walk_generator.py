# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:30:45 2015

Motion Graphs interface for further integration
Runs the complete Morphable Graphs Pipeline to generate a motion based on an
json input file. Runs the optimization sequentially and creates constraints
based on previous steps.

@author: Erik Herrmann, Han Du, Fabian Rupp, Markus Mauer
"""

from ..constraints import ElementaryActionConstraintsBuilder
from elementary_action_generator import ElementaryActionGenerator
from graph_walk import GraphWalk
from graph_walk_optimizer import GraphWalkOptimizer
from ..utilities import write_log, write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_ERROR, LOG_MODE_INFO


class GraphWalkGenerator(object):
    """ Provides a method to synthesize a graph walk based on a json input file
    
    Parameters
    ----------
    * motion_primitive_graph: String
        Contains paths to the motion data.
    * algorithm_config : dict
        Contains options for the algorithm.
    * service_config: dict
        Contains options for the connection with other services and the output.
    """
    def __init__(self, motion_primitive_graph, algorithm_config, service_config):
        self._algorithm_config = algorithm_config
        self.motion_primitive_graph = motion_primitive_graph
        if "create_ca_vis_data" in service_config.keys():
            self.create_ca_vis_data = service_config["create_ca_vis_data"]
        else:
            self.create_ca_vis_data = False
        self.action_generator = ElementaryActionGenerator(self.motion_primitive_graph, self._algorithm_config, service_config)
        self.graph_walk_optimizer = GraphWalkOptimizer(self.motion_primitive_graph, algorithm_config)
        self.action_constraints_builder = ElementaryActionConstraintsBuilder(self.motion_primitive_graph, self._algorithm_config)

    def set_algorithm_config(self, algorithm_config):
        """
        Parameters
        ----------
        * algorithm_config : dict
            Contains options for the algorithm.
        """
        self._algorithm_config = algorithm_config
        self.action_constraints_builder.set_algorithm_config(self._algorithm_config)
        self.action_generator.set_algorithm_config(self._algorithm_config)
        self.graph_walk_optimizer.set_algorithm_config(self._algorithm_config)

    def generate(self, mg_input_reader, prev_graph_walk=None):
        """ Converts constrains into a graph walk through the motion state graph

         Parameters
        ----------
        * mg_input_reader : MGInputFormatReader
        * prev_graph_walk: GraphWalk (optional)
        Returns
        -------
        * graph_walk: GraphWalk
            Contains the quaternion frames and annotations of the frames based on actions.
        """
        action_constraint_list = self.action_constraints_builder.build_list_from_input_file(mg_input_reader)
        if prev_graph_walk is None:
            graph_walk = GraphWalk(self.motion_primitive_graph, mg_input_reader, self._algorithm_config, self.action_constraints_builder.get_start_pose(), self.create_ca_vis_data)
        else:
            graph_walk = prev_graph_walk
            graph_walk.mg_input = mg_input_reader
            start_action_idx = graph_walk.get_number_of_actions()
            action_constraint_list = action_constraint_list[start_action_idx:]
        for action_constraints in action_constraint_list:

            if self._algorithm_config["debug_max_step"] > -1 and graph_walk.step_count > self._algorithm_config["debug_max_step"]:
                write_message_to_log("Abort motion synthesis. Reached maximum debug step number.", LOG_MODE_INFO)
                break

            success = self._add_elementary_action_to_graph_walk(action_constraints, graph_walk)

            if not success:
                write_message_to_log("Error: Aborting motion synthesis due to exception or high error due "
                                     "to unreachable constraints.", LOG_MODE_ERROR)
                return graph_walk
        return graph_walk

    def _add_elementary_action_to_graph_walk(self, action_constraints, graph_walk):
        write_message_to_log("Generate graph walk for" + action_constraints.action_name, LOG_MODE_DEBUG)
        self.action_generator.set_action_constraints(action_constraints)
        success = self.action_generator.append_action_to_graph_walk(graph_walk)
        if success:
            graph_walk = self.graph_walk_optimizer.optimize(graph_walk, self.action_generator.action_state, action_constraints)
            graph_walk.add_entry_to_action_list(action_constraints.action_name, self.action_generator.action_state.start_step, len(graph_walk.steps) - 1, action_constraints)
        return success

    def print_algorithm_config(self):
        for key in self._algorithm_config.keys():
            write_log(key, self._algorithm_config[key])
