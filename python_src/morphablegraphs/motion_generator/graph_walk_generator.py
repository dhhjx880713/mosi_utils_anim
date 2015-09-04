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
from optimization.optimizer_builder import OptimizerBuilder
from constraints.time_constraints_builder import TimeConstraintsBuilder
from constraints.spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE
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
        self.time_error_minimizer = OptimizerBuilder(self._algorithm_config).build_time_error_minimizer()
        self.global_error_minimizer = OptimizerBuilder(self._algorithm_config).build_global_error_minimizer_residual()
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
        if self._algorithm_config["use_global_time_optimization"]:
            self._optimize_time_parameters_over_graph_walk(graph_walk)
        time_in_seconds = time.clock() - start
        minutes = int(time_in_seconds/60)
        seconds = time_in_seconds % 60
        print "finished synthesis in " + str(minutes) + " minutes " + str(seconds) + " seconds"
        graph_walk.print_statistics()
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

            if self._algorithm_config["use_global_spatial_optimization"] and action_constraints.contains_user_constraints:
                self._optimize_over_graph_walk(graph_walk, self.elementary_action_generator.state.start_step-5)

            action_constraints = elementary_action_constraints_builder.get_next_elementary_action_constraints()

        return graph_walk

    def _optimize_over_graph_walk(self, graph_walk, start_step=-1):
        #if start_step < 0:
        #    start_step = len(graph_walk.steps)-20
        start_step = max(start_step, 0)
        if self._algorithm_config["use_global_spatial_optimization"]:
            self._optimize_spatial_parameters_over_graph_walk(graph_walk, start_step)
        if self._algorithm_config["use_global_time_optimization"]:
            self._optimize_time_parameters_over_graph_walk(graph_walk)

    def _optimize_spatial_parameters_over_graph_walk(self, graph_walk, start_step=0):
        initial_guess = []
        for step in graph_walk.steps[start_step:]:
            step.motion_primitive_constraints.constraints = [constraint for constraint in step.motion_primitive_constraints.constraints
                                                             if constraint.constraint_type != SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE]
            initial_guess += step.parameters[:step.n_spatial_components].tolist()
        for step in graph_walk.steps[start_step:]:
            for constraint in step.motion_primitive_constraints.constraints:
                if constraint.desired_time is not None:
                    constraint.weight_factor = 1000.0
        if start_step == 0:
            prev_frames = None
        else:
            prev_frames = graph_walk.get_quat_frames()[:graph_walk.steps[start_step].start_frame]
        print "start global optimization", len(initial_guess)
        self.global_error_minimizer.set_objective_function_parameters((self.motion_primitive_graph, graph_walk.steps[start_step:],
                                self._algorithm_config["optimization_settings"]["error_scale_factor"],
                                self._algorithm_config["optimization_settings"]["quality_scale_factor"],
                                prev_frames))
        optimal_parameters = self.global_error_minimizer.run(initial_guess)
        graph_walk.update_spatial_parameters(optimal_parameters, start_step)

    def _optimize_time_parameters_over_graph_walk(self, graph_walk, start_step=0):

        time_constraints = TimeConstraintsBuilder(graph_walk, start_step).build()
        if time_constraints is not None:
            data = (self.motion_primitive_graph, graph_walk, time_constraints,
                    self._algorithm_config["optimization_settings"]["error_scale_factor"],
                    self._algorithm_config["optimization_settings"]["quality_scale_factor"])
            self.time_error_minimizer.set_objective_function_parameters(data)
            initial_guess = time_constraints.get_initial_guess(graph_walk)
            print "initial_guess", initial_guess, time_constraints.constraint_list
            optimal_parameters = self.time_error_minimizer.run(initial_guess)
            graph_walk.update_time_parameters(optimal_parameters, start_step)
            graph_walk.convert_to_motion(start_step)
