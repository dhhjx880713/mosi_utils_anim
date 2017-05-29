
import time
from algorithm_configuration import AlgorithmConfigurationBuilder
from ..constraints.mg_input_format_reader import MGInputFormatReader
from ..constraints.elementary_action_constraints_builder import ElementaryActionConstraintsBuilder
from ..constraints.motion_primitive_constraints_builder import MotionPrimitiveConstraintsBuilder
from ea_state import ElementaryActionGeneratorState
from motion_primitive_generator import MotionPrimitiveGenerator
from graph_walk import GraphWalk, GraphWalkEntry
from graph_walk_planner import GraphWalkPlanner
from ..motion_model.motion_state_group import NODE_TYPE_END
from ..motion_model.motion_state_graph_loader import MotionStateGraphLoader
from graph_walk_optimizer import GraphWalkOptimizer
from ..utilities import load_json_file, write_log, clear_log, save_log, write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_INFO, LOG_MODE_ERROR, set_log_mode


class MotionGenerator2(object):
    def __init__(self, service_config, algorithm_config):
        self._service_config = service_config
        self._algorithm_config = algorithm_config
        graph_loader = MotionStateGraphLoader()
        graph_loader.set_data_source(self._service_config["model_data"], self._algorithm_config["use_transition_model"])
        self._motion_state_graph = graph_loader.build()
        self.graph_walk_planner = GraphWalkPlanner(self._motion_state_graph, algorithm_config)
        self.graph_walk = None
        self.action_constraints_builder = ElementaryActionConstraintsBuilder(self._motion_state_graph, algorithm_config)
        self.mp_constraints_builder = MotionPrimitiveConstraintsBuilder()
        self.mp_constraints_builder.set_algorithm_config(self._algorithm_config)
        self.end_step_length_factor = 1.0
        self.step_look_ahead_distance = 100
        self.graph_walk_optimizer = GraphWalkOptimizer(self._motion_state_graph, algorithm_config)
        self.inverse_kinematics = None

        self.set_algorithm_config(algorithm_config)

    def generate_motion(self, mg_input, activate_joint_map, activate_coordinate_transform,
            complete_motion_vector=True, speed=1.0):
        clear_log()
        write_message_to_log("Start motion synthesis", LOG_MODE_INFO)
        mg_input_reader = MGInputFormatReader(self._motion_state_graph, activate_joint_map,
                                              activate_coordinate_transform)

        if not mg_input_reader.read_from_dict(mg_input):
            write_message_to_log("Error: Could not process input constraints", LOG_MODE_ERROR)
            return None

        start_time = time.clock()

        offset = mg_input_reader.center_constraints()

        self.graph_walk = GraphWalk(self._motion_state_graph, mg_input_reader, self._algorithm_config)
        action_list = self.action_constraints_builder.build_list_from_input_file(mg_input_reader)
        for constraints in action_list:
            self._generate_action(constraints)

        time_in_seconds = time.clock() - start_time
        write_message_to_log("Finished " + "synthesis" + " in " + str(int(time_in_seconds / 60)) + " minutes "
                             + str(time_in_seconds % 60) + " seconds", LOG_MODE_INFO)

        motion_vector = self.graph_walk.convert_to_annotated_motion(speed)
        if complete_motion_vector:
            motion_vector.frames = self._motion_state_graph.skeleton.complete_motion_vector_from_reference(
                motion_vector.frames)

        motion_vector.translate_root(offset)

        return motion_vector

    def _generate_action(self, action_constraints):
        self.mp_generator = MotionPrimitiveGenerator(action_constraints, self._algorithm_config)
        self.mp_constraints_builder.set_action_constraints(action_constraints)
        action_state = ElementaryActionGeneratorState(self._algorithm_config)
        arc_length_of_end = self.get_end_step_arc_length(action_constraints)
        self.graph_walk_planner.set_state(self.graph_walk, self.mp_generator, action_state, action_constraints, arc_length_of_end)
        node_key = self.graph_walk_planner.get_best_start_node()
        self._generate_motion_primitive(action_constraints, node_key, action_state)
        while not action_state.is_end_state():
            self.graph_walk_planner.set_state(self.graph_walk, self.mp_generator, action_state, action_constraints, arc_length_of_end)
            node_key, next_node_type = self.graph_walk_planner.get_best_transition_node()
            self._generate_motion_primitive(action_constraints, node_key, action_state, next_node_type==NODE_TYPE_END)

        self.graph_walk.step_count += action_state.temp_step
        self.graph_walk.update_frame_annotation(action_constraints.action_name,
                                           action_state.action_start_frame, self.graph_walk.get_num_of_frames())

        write_message_to_log("Reached end of elementary action " + action_constraints.action_name, LOG_MODE_INFO)

    def _generate_motion_primitive(self, action_constraints, node_key, action_state, is_last_step=False):
        new_node_type = self._motion_state_graph.nodes[node_key].node_type
        self.mp_constraints_builder.set_status(node_key,
                                               action_state.travelled_arc_length,
                                               self.graph_walk,
                                               is_last_step)
        mp_constraints = self.mp_constraints_builder.build()

        mp_name = mp_constraints.motion_primitive_name
        prev_mp_name = ""
        prev_parameters = None
        if len(self.graph_walk.steps) > 0:
            prev_mp_name = self.graph_walk.steps[-1].node_key[1]
            prev_parameters = self.graph_walk.steps[-1].parameters

        new_parameters = self.mp_generator.generate_constrained_sample(mp_name, mp_constraints, prev_mp_name,
                                                  self.graph_walk.get_quat_frames(), prev_parameters)
        motion_spline = self._motion_state_graph.nodes[node_key].back_project(new_parameters, use_time_parameters=False)


        self.graph_walk.append_quat_frames(motion_spline.get_motion_vector())

        new_travelled_arc_length = 0
        if action_constraints.root_trajectory is not None:
            new_travelled_arc_length = self._update_travelled_arc_length(action_constraints, self.graph_walk.get_quat_frames(),
                                                                         action_state.travelled_arc_length)
        new_step = GraphWalkEntry(self._motion_state_graph, node_key, new_parameters,
                                  new_travelled_arc_length, action_state.step_start_frame,
                                  self.graph_walk.get_num_of_frames() - 1, mp_constraints)

        self.graph_walk.steps.append(new_step)

        action_state.transition(node_key, new_node_type, new_travelled_arc_length, self.graph_walk.get_num_of_frames())


    def get_end_step_arc_length(self, action_constraints):
        node_group = action_constraints.get_node_group()
        end_state = node_group.get_random_end_state()
        if end_state is not None:
            arc_length_of_end = self._motion_state_graph.nodes[
                                         end_state].average_step_length * self.end_step_length_factor
        else:
            arc_length_of_end = 0.0

        return arc_length_of_end

    def _update_travelled_arc_length(self, action_constraints, new_quat_frames,  prev_travelled_arc_length):
        """update travelled arc length based on new closest point on trajectory """

        max_arc_length = prev_travelled_arc_length + self.step_look_ahead_distance  # was originally set to 80
        closest_point, distance = action_constraints.root_trajectory.find_closest_point(
            new_quat_frames[-1][:3], prev_travelled_arc_length, max_arc_length)
        new_travelled_arc_length, eval_point = action_constraints.root_trajectory.get_absolute_arc_length_of_point(
            closest_point, min_arc_length=prev_travelled_arc_length)
        if new_travelled_arc_length == -1:
            new_travelled_arc_length = action_constraints.root_trajectory.full_arc_length
        return new_travelled_arc_length

    def get_skeleton(self):
        return self._motion_state_graph.skeleton

    def set_algorithm_config(self, algorithm_config):
        """
        Parameters
        ----------
        * algorithm_config : dict
            Contains options for the algorithm.
        """
        if algorithm_config is None:
            algorithm_config_builder = AlgorithmConfigurationBuilder()
            self._algorithm_config = algorithm_config_builder.build()
        else:
            self._algorithm_config = algorithm_config
        self.graph_walk_optimizer.set_algorithm_config(self._algorithm_config)
        if "trajectory_following_settings" in algorithm_config.keys():
            trajectory_following_settings = algorithm_config["trajectory_following_settings"]
            self.end_step_length_factor = trajectory_following_settings["end_step_length_factor"]
            self.step_look_ahead_distance = trajectory_following_settings["look_ahead_distance"]
