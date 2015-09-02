# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:30:45 2015

Motion Graphs interface for further integration
Runs the complete Morphable Graphs Pipeline to generate a motion based on an
json input file. Runs the optimization sequentially and creates constraints
based on previous steps.

@author: Erik Herrmann, Han Du, Fabian Rupp, Markus Mauer
"""

import time
import numpy as np
from ..utilities.io_helper_functions import load_json_file
from ..motion_model.motion_primitive_graph_loader import MotionPrimitiveGraphLoader
from constraints.mg_input_file_reader import MGInputFileReader
from constraints.elementary_action_constraints_builder import ElementaryActionConstraintsBuilder
from elementary_action_graph_walk_generator import ElementaryActionGraphWalkGenerator
from algorithm_configuration import AlgorithmConfigurationBuilder
from graph_walk import GraphWalk

SKELETON_FILE = "skeleton.bvh"  # TODO replace with standard skeleton in data directory


class GraphWalkGenerator(object):
    """
    Creates a MotionPrimitiveGraph instance and provides a method to synthesize a
    motion based on a json input file
    
    Parameters
    ----------
    * algorithm_config : dict
        Contains options for the algorithm.
    * service_config: String
        Contains paths to the motion data.
    """
    def __init__(self, service_config, algorithm_config):
        self._service_config = service_config        
        self._algorithm_config = algorithm_config
        motion_primitive_graph_directory = self._service_config["model_data"]
        transition_directory = self._service_config["transition_data"]
        graph_builder = MotionPrimitiveGraphLoader()
        graph_builder.set_data_source(SKELETON_FILE, motion_primitive_graph_directory,
                                      transition_directory, self._algorithm_config["use_transition_model"])
        self.motion_primitive_graph = graph_builder.build()
        self.elementary_action_generator = ElementaryActionGraphWalkGenerator(self.motion_primitive_graph,
                                                                           self._algorithm_config)
        return

    def set_algorithm_config(self, algorithm_config):
        """
        Parameters
        ----------
        * algorithm_config : dict
            Contains options for the algorithm.
        """
        if algorithm_config is None:
            algorithm_config_builder = AlgorithmConfigurationBuilder()
            self._algorithm_config = algorithm_config_builder.get_configuration()
        else:
            self._algorithm_config = algorithm_config
        self.elementary_action_generator.set_algorithm_config(self._algorithm_config)

    def generate_graph_walk(self, mg_input, export=True):
        """
        Converts a json input file with a list of elementary actions and constraints 
        into a graph_walk saved to a BVH file.
        
        Parameters
        ----------        
        * mg_input_filename : string or dict
            Dict or Path to json file that contains a list of elementary actions with constraints.
        * export : bool
            If set to True the generated graph_walk is exported as BVH together
            with a JSON-annotation file.
            
        Returns
        -------
        * graph_walk : GraphWalk
           Contains a list of quaternion frames and their annotation based on actions.
        """

        if type(mg_input) != dict:
            mg_input = load_json_file(mg_input)
        start = time.clock()
        input_file_reader = MGInputFileReader(mg_input)
        elementary_action_constraints_builder = ElementaryActionConstraintsBuilder(input_file_reader, self.motion_primitive_graph)
        graph_walk = self._generate_graph_walk_from_constraints(elementary_action_constraints_builder)
        seconds = time.clock() - start
        self.print_runtime_statistics(graph_walk, seconds)
        # export the motion to a bvh file if export == True
        if export:
            output_filename = self._service_config["output_filename"]
            if output_filename == "" and "session" in mg_input.keys():
                output_filename = mg_input["session"]
                graph_walk.frame_annotation["sessionID"] = mg_input["session"]
            graph_walk.export_motion(self._service_config["output_dir"], output_filename, add_time_stamp=True, write_log=self._service_config["write_log"])
        return graph_walk

    def _generate_graph_walk_from_constraints(self, elementary_action_constraints_builder):
        """ Converts a constrained graph walk to quaternion frames
         Parameters
        ----------
        * elementary_action_constraints_builder : ElementaryActionConstraintsBuilder
        Returns
        -------
        * graph_walk: GraphWalk
            Contains the quaternion frames and annotations of the frames based on actions.
        """
        if self._algorithm_config["verbose"]:
            for key in self._algorithm_config.keys():
                print key, self._algorithm_config[key]
    
        graph_walk = GraphWalk(self.motion_primitive_graph,
                              elementary_action_constraints_builder.start_pose,
                              self._algorithm_config)
        graph_walk.mg_input = elementary_action_constraints_builder.get_mg_input_file()

        action_constraints = elementary_action_constraints_builder.get_next_elementary_action_constraints()
        while action_constraints is not None:
            if self._algorithm_config["debug_max_step"] > -1 and graph_walk.step_count > self._algorithm_config["debug_max_step"]:
                print "reached max step"
                break
              
            if self._algorithm_config["verbose"]:
                print "convert", action_constraints.action_name, "to graph walk"
    
            self.elementary_action_generator.set_action_constraints(action_constraints)
            success = self.elementary_action_generator.append_elementary_action_to_graph_walk(graph_walk)
                
            if not success:
                print "Arborting conversion"
                return graph_walk
            action_constraints = elementary_action_constraints_builder.get_next_elementary_action_constraints()
        return graph_walk

    def print_runtime_statistics(self, graph_walk, time_in_seconds):
        n_steps = len(graph_walk.steps)
        objective_evaluations = 0
        average_error = 0
        for step in graph_walk.steps:
            objective_evaluations += step.motion_primitive_constraints.evaluations
            average_error += step.motion_primitive_constraints.min_error
        average_error /= n_steps
        minutes = int(time_in_seconds/60)
        seconds = time_in_seconds % 60
        total_time_string = "finished synthesis in " + str(minutes) + " minutes " + str(seconds) + " seconds"
        evaluations_string = "total number of objective evaluations " + str(objective_evaluations)
        error_string = "average error for " + str(n_steps) + \
                       " motion primitives: " + str(average_error)
        print total_time_string
        print evaluations_string
        print error_string

