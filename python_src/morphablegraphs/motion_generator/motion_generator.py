import time
import os
from datetime import datetime
import json
from ..motion_model import MotionStateGraphLoader
from ..constraints import MGInputFormatReader
from algorithm_configuration import AlgorithmConfigurationBuilder
from graph_walk_generator import GraphWalkGenerator
from graph_walk_optimizer import GraphWalkOptimizer
from inverse_kinematics import InverseKinematics
from ..utilities import load_json_file, write_log, clear_log, save_log, write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_INFO, LOG_MODE_ERROR, set_log_mode


class MotionGenerator(object):
    """
    Creates a MotionPrimitiveGraph instance and provides a method to synthesize a motion based on a json input file

    Parameters
    ----------
    * algorithm_config : dict
        Contains options for the algorithm.
    * service_config: dict
        Contains paths to the motion data and information about the input and output format.
    """
    def __init__(self, service_config, algorithm_config):
        self._service_config = service_config
        self._algorithm_config = algorithm_config
        graph_loader = MotionStateGraphLoader()
        graph_loader.set_data_source(self._service_config["model_data"], self._algorithm_config["use_transition_model"])
        self.motion_state_graph = graph_loader.build()
        self.graph_walk_generator = GraphWalkGenerator(self.motion_state_graph, algorithm_config, service_config)
        self.graph_walk_optimizer = GraphWalkOptimizer(self.motion_state_graph, algorithm_config)
        self.inverse_kinematics = None

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
        self.graph_walk_generator.set_algorithm_config(self._algorithm_config)
        self.graph_walk_optimizer.set_algorithm_config(self._algorithm_config)

    def generate_motion(self, mg_input, activate_joint_map=False, activate_coordinate_transform=False, complete_motion_vector=True):
        """
        Converts a json input file with a list of elementary actions and constraints
        into a graph_walk saved to a BVH file.

        Parameters
        ----------
        * mg_input : string or dict
            Dict or Path to json file that contains a list of elementary actions with constraints.
        * activate_joint_map: bool
            Maps left hand to left hand endsite and right hand to right hand endsite
        * activate_coordinate_transform: bool
            Converts input coordinates from CAD coordinate system to OpenGL coordinate system

        Returns
        -------
        * motion_vector : AnnotatedMotionVector
           Contains a list of quaternion frames and their annotation based on actions.
        """
        clear_log()
        write_message_to_log("Start motion synthesis", LOG_MODE_INFO)
        write_message_to_log("Use configuration " + str(self._algorithm_config), LOG_MODE_DEBUG)
        if type(mg_input) != dict:
            mg_input = load_json_file(mg_input)
        start_time = time.clock()
        mg_input_reader = MGInputFormatReader(mg_input, activate_joint_map, activate_coordinate_transform)

        graph_walk = self.graph_walk_generator.generate(mg_input_reader)

        time_in_seconds = time.clock() - start_time
        minutes = int(time_in_seconds/60)
        seconds = time_in_seconds % 60
        write_message_to_log("Finished graph walk generation in " + str(minutes) + " minutes " + str(seconds) + " seconds", LOG_MODE_INFO)
        if self._algorithm_config["use_global_time_optimization"]:
            graph_walk = self.graph_walk_optimizer.optimize_time_parameters_over_graph_walk(graph_walk)

        motion_vector = graph_walk.convert_to_annotated_motion()

        if self._algorithm_config["activate_inverse_kinematics"]:
            write_message_to_log("Modify using inverse kinematics", LOG_MODE_DEBUG)
            self.inverse_kinematics = InverseKinematics(self.motion_state_graph.skeleton, self._algorithm_config, motion_vector.frames[0])
            self.inverse_kinematics.modify_motion_vector(motion_vector)
            self.inverse_kinematics.fill_rotate_events(motion_vector)


        time_in_seconds = time.clock() - start_time
        minutes = int(time_in_seconds / 60)
        seconds = time_in_seconds % 60
        write_message_to_log("Finished synthesis in " + str(minutes) + " minutes " + str(seconds) + " seconds", LOG_MODE_INFO)

        if complete_motion_vector:

            motion_vector.frames = self.motion_state_graph.skeleton.complete_motion_vector_from_reference(motion_vector.frames)
            if motion_vector.frames is not None:
                if self.motion_state_graph.hand_pose_generator is not None:
                    write_message_to_log("Generate hand poses", LOG_MODE_DEBUG)
                    self.motion_state_graph.hand_pose_generator.generate_hand_poses(motion_vector)
                self._output_info(graph_walk, start_time)
        return motion_vector

    def _output_info(self, graph_walk, start_time):

            write_log(graph_walk.get_statistics_string())
            if self._service_config["write_log"]:
                time_stamp = unicode(datetime.now().strftime("%d%m%y_%H%M%S"))
                save_log(self._service_config["output_dir"] + os.sep + "mg_"+time_stamp + ".log")
                graph_walk.export_generated_constraints(self._service_config["output_dir"] + os.sep + "generated_constraints_"+time_stamp + ".json")

    def export_statistics(self, mg_input, graph_walk,  motion_vector, filename):
        if motion_vector.has_frames():
            output_filename = self._service_config["output_filename"]
            if output_filename == "" and "session" in mg_input.keys():
                output_filename = mg_input["session"]
                motion_vector.frame_annotation["sessionID"] = mg_input["session"]
                motion_vector.export(self._service_config["output_dir"], output_filename, add_time_stamp=True, export_details=self._service_config["write_log"])
                #self.export_statistics(graph_walk, self._service_config["output_dir"] + os.sep + output_filename + "_statistics" + time_stamp + ".json")

                statistics_string = graph_walk.get_statistics_string()
                constraints = graph_walk.get_generated_constraints()
                constraints_string = json.dumps(constraints)
                if filename is None:
                    time_stamp = unicode(datetime.now().strftime("%d%m%y_%H%M%S"))
                    filename = "graph_walk_statistics" + time_stamp + ".json"
                outfile = open(filename, "wb")
                outfile.write(statistics_string)
                outfile.write("\n"+constraints_string)
                outfile.close()
            else:
                write_message_to_log("Error: no motion data to export", LOG_MODE_ERROR)

    def get_skeleton(self):
        return self.motion_state_graph.skeleton



