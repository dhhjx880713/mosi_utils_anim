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
from ..motion_model.motion_primitive_graph_builder import MotionPrimitiveGraphBuilder
from constraints.elementary_action_constraints_builder import ElementaryActionConstraintsBuilder
from elementary_action_sample_generator import ElementaryActionSampleGenerator
from . import global_counter_dict
from algorithm_configuration import AlgorithmConfigurationBuilder
from motion_sample import MotionSample

SKELETON_FILE = "skeleton.bvh"  # TODO replace with standard skeleton in data directory


class MotionSampleGenerator(object):
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
        graph_builder = MotionPrimitiveGraphBuilder()
        graph_builder.set_data_source(SKELETON_FILE, motion_primitive_graph_directory,
                                      transition_directory, self._algorithm_config["use_transition_model"])
        self.motion_primitive_graph = graph_builder.build()
        self.elementary_action_generator = ElementaryActionSampleGenerator(self.motion_primitive_graph,
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

    def generate_motion(self, mg_input, export=True):
        """
        Converts a json input file with a list of elementary actions and constraints 
        into a motion saved to a BVH file.
        
        Parameters
        ----------        
        * mg_input_filename : string or dict
            Dict or Path to json file that contains a list of elementary actions with constraints.
        * export : bool
            If set to True the generated motion is exported as BVH together 
            with a JSON-annotation file.
            
        Returns
        -------
        * motion : MotionGeneratorResult
           Contains a list of quaternion frames and their annotation based on actions.
        """
        
        global_counter_dict["evaluations"] = 0
        if type(mg_input) != dict:
            mg_input = load_json_file(mg_input)
        start = time.clock()
        elementary_action_constraints_builder = ElementaryActionConstraintsBuilder(mg_input, self.motion_primitive_graph)
        motion = self._generate_motion_from_constraints(elementary_action_constraints_builder)
        seconds = time.clock() - start
        self.print_runtime_statistics(seconds)
        # export the motion to a bvh file if export == True
        if export:
            output_filename = self._service_config["output_filename"]
            if output_filename == "" and "session" in mg_input.keys():
                output_filename = mg_input["session"]
                motion.frame_annotation["sessionID"] = mg_input["session"]
            motion.export(self._service_config["output_dir"], output_filename, add_time_stamp=True, write_log=self._service_config["write_log"])
        return motion

    def _generate_motion_from_constraints(self, elementary_action_constraints_builder):
        """ Converts a constrained graph walk to quaternion frames
         Parameters
        ----------
        * elementary_action_constraints_builder : ElementaryActionConstraintsBuilder
        Returns
        -------
        * motion: MotionSample
            Contains the quaternion frames and annotations of the frames based on actions.
        """
        if self._algorithm_config["verbose"]:
            for key in self._algorithm_config.keys():
                print key, self._algorithm_config[key]
    
        motion = MotionSample(self.motion_primitive_graph.skeleton,
                              elementary_action_constraints_builder.start_pose,
                              self._algorithm_config)
        motion.mg_input = elementary_action_constraints_builder.mg_input

        action_constraints = elementary_action_constraints_builder.get_next_elementary_action_constraints()
        while action_constraints is not None:
            if self._algorithm_config["debug_max_step"] > -1 and motion.step_count > self._algorithm_config["debug_max_step"]:
                print "reached max step"
                break
              
            if self._algorithm_config["verbose"]:
                print "convert", action_constraints.action_name, "to graph walk"
    
            self.elementary_action_generator.set_action_constraints(action_constraints)
            success = self.elementary_action_generator.append_elementary_action_to_motion(motion)
                
            if not success:
                print "Arborting conversion"
                return motion
            action_constraints = elementary_action_constraints_builder.get_next_elementary_action_constraints()
        return motion

    def print_runtime_statistics(self, time_in_seconds):
        minutes = int(time_in_seconds/60)
        seconds = time_in_seconds % 60
        total_time_string = "finished synthesis in " + str(minutes) + " minutes " + str(seconds) + " seconds"
        evaluations_string = "total number of objective evaluations " + str(global_counter_dict["evaluations"])
        error_string = "average error for " + str(len(global_counter_dict["motionPrimitveErrors"])) +" motion primitives: " + str(np.average(global_counter_dict["motionPrimitveErrors"],axis=0))
        print total_time_string
        print evaluations_string
        print error_string

