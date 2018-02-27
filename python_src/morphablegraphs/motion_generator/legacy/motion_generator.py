import json
import os
import time
from datetime import datetime
from ...constraints import MGInputFormatReader
from ..algorithm_configuration import DEFAULT_ALGORITHM_CONFIG
from ..graph_walk_optimizer import GraphWalkOptimizer
from ...animation_data.motion_editing import LegacyInverseKinematics
from .graph_walk_generator import GraphWalkGenerator
from ...motion_model import MotionStateGraphLoader
from ...utilities import clear_log, save_log, write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_INFO, LOG_MODE_ERROR


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
            self._algorithm_config = DEFAULT_ALGORITHM_CONFIG
        else:
            self._algorithm_config = algorithm_config
        self.graph_walk_generator.set_algorithm_config(self._algorithm_config)
        self.graph_walk_optimizer.set_algorithm_config(self._algorithm_config)

    def generate_motion(self, mg_input, activate_joint_map=False, activate_coordinate_transform=False,
                        complete_motion_vector=True, speed=1.0):
        """
        Converts a json input file with a list of elementary actions and constraints
        into a motion vector

        Parameters
        ----------
        * mg_input :  dict
            Dict contains a list of elementary actions with constraints.
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

        start_time = time.clock()

        mg_input_reader = MGInputFormatReader(self.motion_state_graph, activate_joint_map, activate_coordinate_transform)

        if not mg_input_reader.read_from_dict(mg_input):
            write_message_to_log("Error: Could not process input constraints", LOG_MODE_ERROR)
            return None

        offset = mg_input_reader.center_constraints()

        graph_walk = self.graph_walk_generator.generate(mg_input_reader)

        self._print_time("graph walk generation", start_time)

        if self._algorithm_config["use_global_time_optimization"]:
            graph_walk = self.graph_walk_optimizer.optimize_time_parameters_over_graph_walk(graph_walk)

        motion_vector = graph_walk.convert_to_annotated_motion(speed)

        motion_vector = self._post_process_motion(motion_vector, complete_motion_vector, start_time)

        motion_vector.translate_root(offset)

        self._print_info(graph_walk)

        return motion_vector

    def _post_process_motion(self, motion_vector, complete_motion_vector, start_time):
        """
        Applies inverse kinematics constraints on a annotated motion vector and adds hand poses
         and DOFs that are not modelled.

        Parameters
        ----------
        * motion_vector : AnnotatedMotionVector
            Contains motion but also the constraints
        * complete_motion_vector: bool
            Sets DOFs that are not modelled by the motion model using default values.
        * start_time: int
            Start of motion synthesis for time measurement messages.

        Returns
        -------
        * motion_vector : AnnotatedMotionVector
           Contains a list of quaternion frames and their annotation based on actions.
        """
        if self._algorithm_config["activate_inverse_kinematics"]:
            write_message_to_log("Modify using inverse kinematics", LOG_MODE_DEBUG)
            self.inverse_kinematics = LegacyInverseKinematics(self.motion_state_graph.skeleton, self._algorithm_config)
            self.inverse_kinematics.modify_motion_vector(motion_vector)
            self.inverse_kinematics.fill_rotate_events(motion_vector)
        self._print_time("synthesis", start_time)
        if complete_motion_vector:
            motion_vector.frames = self.motion_state_graph.skeleton.add_fixed_joint_parameters_to_motion(
                motion_vector.frames)
            if motion_vector.frames is not None:
                if self.motion_state_graph.hand_pose_generator is not None:
                    write_message_to_log("Generate hand poses", LOG_MODE_DEBUG)
                    self.motion_state_graph.hand_pose_generator.generate_hand_poses(motion_vector)
        return motion_vector

    def get_skeleton(self):
        return self.motion_state_graph.skeleton

    def _print_info(self, graph_walk):
        write_message_to_log(graph_walk.get_statistics_string(), LOG_MODE_INFO)
        if self._service_config["write_log"]:
            time_stamp = str(datetime.now().strftime("%d%m%y_%H%M%S"))
            save_log(self._service_config["output_dir"] + os.sep + "mg_" + time_stamp + ".log")
            graph_walk.export_generated_constraints(
                self._service_config["output_dir"] + os.sep + "generated_constraints_" + time_stamp + ".json")

    def export_statistics(self, mg_input, graph_walk, motion_vector, filename):
        if motion_vector.has_frames():
            output_filename = self._service_config["output_filename"]
            if output_filename == "" and "session" in list(mg_input.keys()):
                output_filename = mg_input["session"]
                motion_vector.frame_annotation["sessionID"] = mg_input["session"]
                motion_vector.export(self._service_config["output_dir"], output_filename, add_time_stamp=True,
                                     export_details=self._service_config["write_log"])
                # self.export_statistics(graph_walk, self._service_config["output_dir"] + os.sep + output_filename + "_statistics" + time_stamp + ".json")

                statistics_string = graph_walk.get_statistics_string()
                constraints = graph_walk.get_generated_constraints()
                constraints_string = json.dumps(constraints)
                if filename is None:
                    time_stamp = str(datetime.now().strftime("%d%m%y_%H%M%S"))
                    filename = "graph_walk_statistics" + time_stamp + ".json"
                outfile = open(filename, "wb")
                outfile.write(statistics_string)
                outfile.write("\n" + constraints_string)
                outfile.close()
            else:
                write_message_to_log("Error: no motion data to export", LOG_MODE_ERROR)

    def _print_time(self, method_name, start_time):
        time_in_seconds = time.clock() - start_time
        write_message_to_log("Finished "+method_name+" in " + str(int(time_in_seconds / 60)) + " minutes "
                             + str(time_in_seconds % 60) + " seconds", LOG_MODE_INFO)


