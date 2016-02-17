import time
from ..motion_model import MotionStateGraphLoader
from constraints import MGInputFileReader
from algorithm_configuration import AlgorithmConfigurationBuilder
from graph_walk import GraphWalk
from graph_walk_generator import GraphWalkGenerator
from graph_walk_optimizer import GraphWalkOptimizer
from ..utilities import load_json_file


class MotionGenerator(object):
    """
    Creates a MotionPrimitiveGraph instance and provides a method to synthesize a motion based on a json input file

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
        graph_loader = MotionStateGraphLoader()
        graph_loader.set_data_source(self._service_config["model_data"], self._algorithm_config["use_transition_model"])
        self.motion_primitive_graph = graph_loader.build()
        self.graph_walk_generator = GraphWalkGenerator(self.motion_primitive_graph, algorithm_config)
        self.graph_walk_optimizer = GraphWalkOptimizer(self.motion_primitive_graph, algorithm_config)

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
        self.graph_walk_optimizer.set_algorithm_config(self._algorithm_config)
        self._global_spatial_optimization_steps = self._algorithm_config["global_spatial_optimization_settings"]["max_steps"]

    def generate_motion(self, mg_input, export=True, activate_joint_map=False, activate_coordinate_transform=False):
            """
            Converts a json input file with a list of elementary actions and constraints
            into a graph_walk saved to a BVH file.

            Parameters
            ----------
            * mg_input : string or dict
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
            * motion_vector : AnnotatedMotionVector
               Contains a list of quaternion frames and their annotation based on actions.
            """

            if type(mg_input) != dict:
                mg_input = load_json_file(mg_input)
            start = time.clock()
            input_file_reader = MGInputFileReader(mg_input, activate_joint_map, activate_coordinate_transform)
            graph_walk = self.graph_walk_generator.generate_graph_walk_from_constraints(input_file_reader)
            if self._algorithm_config["use_global_time_optimization"]:
                graph_walk = self.graph_walk_optimizer.optimize_time_parameters_over_graph_walk(graph_walk)


            time_in_seconds = time.clock() - start
            minutes = int(time_in_seconds/60)
            seconds = time_in_seconds % 60
            print "finished synthesis in " + str(minutes) + " minutes " + str(seconds) + " seconds"
            graph_walk.print_statistics()
            if export:  # export the motion to a bvh file if export == True
                output_filename = self._service_config["output_filename"]
                if output_filename == "" and "session" in mg_input.keys():
                    output_filename = mg_input["session"]
                    graph_walk.frame_annotation["sessionID"] = mg_input["session"]
                graph_walk.export_motion(self._service_config["output_dir"], output_filename, add_time_stamp=True, export_details=self._service_config["write_log"])
            #TODO return AnnotatedMotionVector
            return graph_walk
