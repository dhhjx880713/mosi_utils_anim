# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:30:45 2015

Motion Graphs interface for further integration
Runs the complete Morphable Graphs Pipeline to generate a motion based on an
json input file. Runs the optimization sequentially and creates constraints
based on previous steps.

@author: Erik Herrmann, Han Du, Fabian Rupp, Markus Mauer
"""

from constraints import ElementaryActionConstraintsBuilder
from elementary_action_generator import ElementaryActionGenerator
from graph_walk import GraphWalk
from graph_walk_optimizer import GraphWalkOptimizer
from ..utilities import write_log



class GraphWalkGenerator(object):
    """ Provides a method to synthesize a graph walk based on a json input file
    
    Parameters
    ----------
    * motion_primitive_graph: String
        Contains paths to the motion data.
    * algorithm_config : dict
        Contains options for the algorithm.
    """
    def __init__(self, motion_primitive_graph, algorithm_config):
        self._algorithm_config = algorithm_config
        self.motion_primitive_graph = motion_primitive_graph
        self.action_generator = ElementaryActionGenerator(self.motion_primitive_graph, self._algorithm_config)
        self.graph_walk_optimizer = GraphWalkOptimizer(self.motion_primitive_graph, algorithm_config)

    def set_algorithm_config(self, algorithm_config):
        """
        Parameters
        ----------
        * algorithm_config : dict
            Contains options for the algorithm.
        """
        self._algorithm_config = algorithm_config
        self.action_generator.set_algorithm_config(self._algorithm_config)

    def generate_graph_walk_from_constraints(self, input_file_reader):
        """ Converts a constrained graph walk to quaternion frames
         Parameters
        ----------
        * input_file_reader : MGInputFileReader
        Returns
        -------
        * graph_walk: GraphWalk
            Contains the quaternion frames and annotations of the frames based on actions.
        """

        if self._algorithm_config["verbose"]:
            self.print_algorithm_config()

        action_constraints_builder = ElementaryActionConstraintsBuilder(input_file_reader, self.motion_primitive_graph, self._algorithm_config)
        graph_walk = GraphWalk(self.motion_primitive_graph, action_constraints_builder.start_pose, self._algorithm_config)
        graph_walk.mg_input = action_constraints_builder.mg_input
        action_constraints = action_constraints_builder.get_next_elementary_action_constraints()
        while action_constraints is not None:
            if self._algorithm_config["debug_max_step"] > -1 and graph_walk.step_count > self._algorithm_config["debug_max_step"]:
                write_log("Aborting motion synthesis: Reached maximum debug step number")
                break
            success = self._add_elementary_action_to_graph_walk(action_constraints, graph_walk)
            if not success:
                write_log("Aborting motion synthes: Error from constraints has become too high. due to unreachable constraints.")
                return graph_walk
            action_constraints = action_constraints_builder.get_next_elementary_action_constraints()
        return graph_walk

    def _add_elementary_action_to_graph_walk(self, action_constraints, graph_walk):
        if self._algorithm_config["verbose"]:
            write_log("Generate graph walk for", action_constraints.action_name)

        self.action_generator.set_action_constraints(action_constraints)
        success = self.action_generator.append_action_to_graph_walk(graph_walk)
        if success:
            graph_walk = self.graph_walk_optimizer.optimize(graph_walk, self.action_generator, action_constraints)
            graph_walk.add_entry_to_action_list(action_constraints.action_name, self.action_generator.action_state.start_step, len(graph_walk.steps) - 1)
        return success

    def print_algorithm_config(self):
        for key in self._algorithm_config.keys():
            write_log(key, self._algorithm_config[key])
