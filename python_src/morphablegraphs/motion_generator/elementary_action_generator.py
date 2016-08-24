__author__ = 'erhe01'

import numpy as np
from copy import copy
from ..utilities.exceptions import PathSearchError
from ..motion_model import NODE_TYPE_END
from motion_primitive_generator import MotionPrimitiveGenerator
from constraints.motion_primitive_constraints_builder import MotionPrimitiveConstraintsBuilder
from graph_walk import GraphWalkEntry
from constraints import CA_CONSTRAINTS_MODE_DIRECT_CONNECTION
from ..utilities import write_log
from ca_interface import CAInterface
from ea_state import ElementaryActionGeneratorState
from graph_walk_planner import GraphWalkPlanner


class ElementaryActionGenerator(object):

    def __init__(self, motion_primitive_graph, algorithm_config, service_config):
        self.motion_state_graph = motion_primitive_graph
        self._algorithm_config = algorithm_config
        self.ca_action_names = service_config["collision_avoidance_actions"]
        self.motion_primitive_constraints_builder = MotionPrimitiveConstraintsBuilder()
        self.motion_primitive_constraints_builder.set_algorithm_config(self._algorithm_config)
        self.action_state = ElementaryActionGeneratorState(self._algorithm_config)
        self.step_look_ahead_distance = algorithm_config["trajectory_following_settings"]["look_ahead_distance"]
        self.average_elementary_action_error_threshold = algorithm_config["average_elementary_action_error_threshold"]
        self.use_local_coordinates = self._algorithm_config["use_local_coordinates"]
        self.end_step_length_factor = algorithm_config["trajectory_following_settings"]["end_step_length_factor"]
        self.max_distance_to_path = algorithm_config["trajectory_following_settings"]["max_distance_to_path"]
        self.activate_direction_ca_connection = algorithm_config["collision_avoidance_constraints_mode"] == CA_CONSTRAINTS_MODE_DIRECT_CONNECTION
        self.path_planner = GraphWalkPlanner(self.motion_state_graph, algorithm_config)

        if service_config["collision_avoidance_service_url"] is not None:
            print "created ca interface", service_config["collision_avoidance_service_url"]
            self.ca_interface = CAInterface(self, service_config)
        else:
            self.ca_interface = None

    def set_algorithm_config(self, algorithm_config):
        self._algorithm_config = algorithm_config
        self.action_state.debug_max_step = algorithm_config["debug_max_step"]
        self.step_look_ahead_distance = algorithm_config["trajectory_following_settings"]["look_ahead_distance"]
        self.average_elementary_action_error_threshold = algorithm_config["average_elementary_action_error_threshold"]
        self.use_local_coordinates = self._algorithm_config["use_local_coordinates"]
        self.end_step_length_factor = algorithm_config["trajectory_following_settings"]["end_step_length_factor"]
        self.max_distance_to_path = algorithm_config["trajectory_following_settings"]["max_distance_to_path"]
        self.activate_direction_ca_connection = algorithm_config["collision_avoidance_constraints_mode"] == CA_CONSTRAINTS_MODE_DIRECT_CONNECTION
        self.motion_primitive_constraints_builder.set_algorithm_config(self._algorithm_config)

    def set_action_constraints(self, action_constraints):
        self.action_constraints = action_constraints
        self.motion_primitive_constraints_builder.set_action_constraints(self.action_constraints)
        self.motion_primitive_generator = MotionPrimitiveGenerator(self.action_constraints, self._algorithm_config)
        self.node_group = self.action_constraints.get_node_group()
        end_state = self.node_group.get_random_end_state()
        if end_state is not None:
            self.arc_length_of_end = self.motion_state_graph.nodes[end_state].average_step_length * self.end_step_length_factor
        else:
            self.arc_length_of_end = 0.0

    def _select_next_motion_primitive_node(self, graph_walk):
        """extract from graph based on previous step and heuristic """
        self.path_planner.set_state(graph_walk, self.motion_primitive_generator, self.action_state, self.action_constraints, self.arc_length_of_end)

        if self.action_state.current_node is None:  # is start state
            if self.action_constraints.root_trajectory is not None: # use trajectory to determine best start
                next_node = self.path_planner.get_best_start_node()
            else:
                next_node = self.motion_state_graph.get_random_action_transition(graph_walk, self.action_constraints.action_name, self.action_constraints.cycled_previous)

            next_node_type = self.motion_state_graph.nodes[next_node].node_type
            if next_node is None:
                write_log("Error: Could not find a transition of type action_transition from ", self.action_state.prev_action_name, self.action_state.prev_mp_name, " to state", self.action_state.current_node)

        else:  # is intermediate start state
            next_node, next_node_type = self.path_planner.get_best_transition_node()
        return next_node, next_node_type

    def _update_travelled_arc_length(self, new_quat_frames, prev_graph_walk, prev_travelled_arc_length):
        """update travelled arc length based on new closest point on trajectory """
        #if len(prev_graph_walk) > 0:
        #    min_arc_length = prev_graph_walk[-1].arc_length
        #else:
        #    min_arc_length = 0.0
        max_arc_length = prev_travelled_arc_length + self.step_look_ahead_distance  # was originally set to 80
        closest_point, distance = self.action_constraints.root_trajectory.find_closest_point(new_quat_frames[-1][:3],  prev_travelled_arc_length, max_arc_length)
        new_travelled_arc_length, eval_point = self.action_constraints.root_trajectory.get_absolute_arc_length_of_point(closest_point, min_arc_length=prev_travelled_arc_length)
        if new_travelled_arc_length == -1:
            new_travelled_arc_length = self.action_constraints.root_trajectory.full_arc_length
        return new_travelled_arc_length

    def _gen_motion_primitive_constraints(self, next_node, next_node_type, graph_walk):
        try:
            is_last_step = (next_node_type == NODE_TYPE_END)
            self.motion_primitive_constraints_builder.set_status(next_node,
                                                                 self.action_state.travelled_arc_length,
                                                                 graph_walk,
                                                                 is_last_step)
            return self.motion_primitive_constraints_builder.build()

        except PathSearchError as e:
            write_log("Error: Moved beyond last point of spline using parameters", str(e.search_parameters))
            return None

    def append_action_to_graph_walk(self, graph_walk):
        """Convert an entry in the elementary action list to a list of quaternion frames.
        Note only one trajectory constraint per elementary action is currently supported
        and it should be for the Hip joint.

        If there is a trajectory constraint the algorithm will try to follow it otherwise
        a random graph walk is generated based on predefined transitions in the graph
        If there is a keyframe constraint it is assigned to the motion primitives
        in the graph walk

        Parameters
        ---------
        * graph_walk: GraphWalk
            Result object contains the intermediary result of the motion generation process. The animation keyframes of
            the elementary action will be appended to the frames in this object
        Returns
        -------
        * success: Bool
            True if successful and False, if an error occurred during the constraints generation
        """
        if self.action_constraints.root_trajectory is not None:
            max_arc_length = self.action_constraints.root_trajectory.full_arc_length
        else:
            max_arc_length = np.inf
        self.action_state.initialize_from_previous_graph_walk(graph_walk, max_arc_length, self.action_constraints.cycled_next)
        write_log("Start synthesis of elementary action", self.action_constraints.action_name)
        errors = [0]
        distances = []
        while not self.action_state.is_end_state():
            error = self._transition_to_next_action_state(graph_walk)
            if error is None:
                return False  # the generation of the state was not successful
            errors.append(error)
            if self.action_constraints.root_trajectory is not None:
                d = self._distance_to_path(graph_walk)
                distances.append(d)
                write_log("distance to path",d)
                if d >= self.max_distance_to_path:
                    write_log("Warning: Distance to path has become larger than", self.max_distance_to_path)
                    return False

        graph_walk.step_count += self.action_state.temp_step
        graph_walk.update_frame_annotation(self.action_constraints.action_name, self.action_state.action_start_frame, graph_walk.get_num_of_frames())
        avg_error = np.average(errors)
        write_log("Reached end of elementary action", self.action_constraints.action_name, "with an average error of", avg_error)
        return avg_error < self.average_elementary_action_error_threshold

    def _transition_to_next_action_state(self, graph_walk):

        new_node, new_node_type = self._select_next_motion_primitive_node(graph_walk)
        if new_node is None:
            write_log("Error: Failed to find a transition")
            return None
        write_log("Transition to state", new_node)

        mp_constraints = self._gen_motion_primitive_constraints(new_node, new_node_type, graph_walk)

        #f self.action_constraints.cycled_next:
        #    print "is cycled action"
        #    new_node_type = NODE_TYPE_CYCLE_END
        #else:
        #    print "is not cycled"

        if mp_constraints is None:
            write_log("Error: Failed to generate constraints")
            return None
        new_motion_spline, new_parameters = self.motion_primitive_generator.generate_constrained_motion_spline(mp_constraints, graph_walk)

        if self.activate_direction_ca_connection and self.ca_interface is not None and self.action_constraints.action_name in self.ca_action_names:
            ca_constraints = self.ca_interface.get_constraints(self.action_constraints.group_id, new_node, new_motion_spline, graph_walk)
            print "ca constraints",ca_constraints
            if ca_constraints is not None and len(ca_constraints) > 0:
                mp_constraints.constraints += ca_constraints
                new_motion_spline, new_parameters = self.motion_primitive_generator.generate_constrained_motion_spline(mp_constraints, graph_walk)

        new_arc_length = self._create_graph_walk_entry(new_node, new_motion_spline, new_parameters, mp_constraints, graph_walk)
        self.action_state.transition(new_node, new_node_type, new_arc_length, graph_walk.get_num_of_frames())

        return mp_constraints.min_error

    def _create_graph_walk_entry(self, new_node, new_motion_spline, new_parameters, mp_constraints, graph_walk):
        """ Concatenate frames to motion and apply smoothing """
        prev_steps = graph_walk.steps
        graph_walk.append_quat_frames(new_motion_spline.get_motion_vector())

        if self.action_constraints.root_trajectory is not None:
            new_travelled_arc_length = self._update_travelled_arc_length(graph_walk.get_quat_frames(), prev_steps, self.action_state.travelled_arc_length)
        else:
            new_travelled_arc_length = 0
        #new_travelled_arc_length = mp_constraints.goal_arc_length

        new_step = GraphWalkEntry(self.motion_state_graph, new_node, new_parameters,
                                  new_travelled_arc_length, self.action_state.step_start_frame,
                                  graph_walk.get_num_of_frames() - 1, mp_constraints)
        graph_walk.steps.append(new_step)
        return new_travelled_arc_length

    def _distance_to_path(self, graph_walk):
        step_goal = copy(graph_walk.steps[-1].motion_primitive_constraints.step_goal)
        step_goal[1] = 0.0
        root = copy(graph_walk.motion_vector.frames[-1][:3])
        root[1] = 0.0
        d = np.linalg.norm(step_goal - root)
        return d

