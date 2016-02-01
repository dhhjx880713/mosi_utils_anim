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
from ..utilities.io_helper_functions import load_json_file
from ..motion_model.motion_state_graph_loader import MotionStateGraphLoader
from constraints.mg_input_file_reader import MGInputFileReader
from constraints.elementary_action_constraints_builder import ElementaryActionConstraintsBuilder
from elementary_action_graph_walk_generator import ElementaryActionGraphWalkGenerator
from algorithm_configuration import AlgorithmConfigurationBuilder
from graph_walk import GraphWalk
from graph_walk_optimizer import GraphWalkOptimizer


class GraphWalkGenerator(GraphWalkOptimizer):
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

        super(GraphWalkGenerator, self).__init__(algorithm_config)
        motion_primitive_graph_path = self._service_config["model_data"]
        graph_loader = MotionStateGraphLoader()
        graph_loader.set_data_source(motion_primitive_graph_path, self._algorithm_config["use_transition_model"])
        self.motion_primitive_graph = graph_loader.build()
        self.elementary_action_generator = ElementaryActionGraphWalkGenerator(self.motion_primitive_graph,
                                                                              self._algorithm_config)
        self._global_spatial_optimization_steps = self._algorithm_config["global_spatial_optimization_settings"]["max_steps"]

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
        self._global_spatial_optimization_steps = self._algorithm_config["global_spatial_optimization_settings"]["max_steps"]

    def generate_graph_walk(self, mg_input, export=True, activate_joint_map=False, activate_coordinate_transform=False):
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
        * activate_joint_map: bool
            Maps left hand to left hand endsite and right hand to right hand endsite
        * activate_coordinate_transform: bool
            Converts input coordinates from CAD coordinate system to OpenGL coordinate system
            
        Returns
        -------
        * graph_walk : GraphWalk
           Contains a list of quaternion frames and their annotation based on actions.
        """

        if type(mg_input) != dict:
            mg_input = load_json_file(mg_input)
        start = time.clock()
        input_file_reader = MGInputFileReader(mg_input, activate_joint_map, activate_coordinate_transform)
        elementary_action_constraints_builder = ElementaryActionConstraintsBuilder(input_file_reader, self.motion_primitive_graph, self._algorithm_config)
        graph_walk = self._generate_graph_walk_from_constraints(elementary_action_constraints_builder)
        if self._algorithm_config["use_global_time_optimization"]:
            graph_walk = self._optimize_time_parameters_over_graph_walk(graph_walk)
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
            graph_walk.export_motion(self._service_config["output_dir"], output_filename, add_time_stamp=True, export_details=self._service_config["write_log"])
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
        graph_walk.mg_input = elementary_action_constraints_builder.mg_input
        graph_walk.hand_pose_generator = self.motion_primitive_graph.hand_pose_generator
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

            #print "has user constraints", action_constraints.contains_user_constraints
            if self._algorithm_config["use_global_spatial_optimization"] and action_constraints.contains_user_constraints:
                #print "spatial graph walk optimization"
                start_step = max(self.elementary_action_generator.state.start_step-self._global_spatial_optimization_steps, 0)
                graph_walk = self._optimize_spatial_parameters_over_graph_walk(graph_walk, start_step)

            if self._algorithm_config["optimize_collision_avoidance_constraints_extra"] and action_constraints.collision_avoidance_constraints is not None and len(action_constraints.collision_avoidance_constraints) > 0 :
                print "optimize collision avoidance parameters"
                graph_walk = self._optimize_for_collision_avoidance_constraints(graph_walk, action_constraints, self.elementary_action_generator.state.start_step)

            graph_walk.add_entry_to_action_list(action_constraints.action_name, self.elementary_action_generator.state.start_step, len(graph_walk.steps)-1)

            action_constraints = elementary_action_constraints_builder.get_next_elementary_action_constraints()

        return graph_walk

