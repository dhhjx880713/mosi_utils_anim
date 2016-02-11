__author__ = 'erhe01'

import numpy as np
from ..utilities.exceptions import PathSearchError
from ..motion_model import NODE_TYPE_END, NODE_TYPE_SINGLE
from motion_primitive_generator import MotionPrimitiveGenerator
from constraints.motion_primitive_constraints_builder import MotionPrimitiveConstraintsBuilder
from graph_walk import GraphWalkEntry
from constraints.motion_primitive_constraints import MotionPrimitiveConstraints
from constraints.spatial_constraints.keyframe_constraints.direction_constraint import DirectionConstraint
from constraints.spatial_constraints.keyframe_constraints.global_transform_constraint import GlobalTransformConstraint


class ElementaryActionGeneratorState(object):
        def __init__(self, algorithm_config):
            self.start_step = -1
            self.prev_action_name = None
            self.prev_mp_name = None
            self.action_start_frame = -1
            self.current_node = None
            self.current_node_type = ""
            self.temp_step = 0
            self.travelled_arc_length = 0.0
            self.debug_max_step = algorithm_config["debug_max_step"]
            self.step_start_frame = 0

        def initialize_from_previous_graph_walk(self, graph_walk):
            self.start_step = graph_walk.step_count
            if self.start_step > 0:
                self.prev_action_name = graph_walk.steps[-1]
                self.prev_mp_name = graph_walk.steps[-1]
            else:
                self.prev_action_name = None
                self.prev_mp_name = None
            self.action_start_frame = graph_walk.get_num_of_frames()
            self.current_node = None
            self.current_node_type = ""
            self.temp_step = 0
            self.travelled_arc_length = 0.0

        def is_end_state(self):
            reached_debug_max_step = self.start_step + self.temp_step > self.debug_max_step and self.debug_max_step > -1
            return self.current_node_type == NODE_TYPE_END or self.current_node_type == NODE_TYPE_SINGLE or reached_debug_max_step

        def transition(self, new_node, new_node_type, new_travelled_arc_length, new_step_start_frame):
            self.current_node = new_node
            self.current_node_type = new_node_type
            self.travelled_arc_length = new_travelled_arc_length
            self.step_start_frame = new_step_start_frame
            self.temp_step += 1


class ElementaryActionGenerator(object):
    def __init__(self, motion_primitive_graph, algorithm_config):
        self.motion_primitive_graph = motion_primitive_graph
        self._algorithm_config = algorithm_config
        self.motion_primitive_constraints_builder = MotionPrimitiveConstraintsBuilder()
        self.motion_primitive_constraints_builder.set_algorithm_config(self._algorithm_config)
        self.action_state = ElementaryActionGeneratorState(self._algorithm_config)
        self.start_node_selection_look_ahead_distance = algorithm_config["trajectory_following_settings"]["look_ahead_distance"]
        self.average_elementary_action_error_threshold = algorithm_config["average_elementary_action_error_threshold"]

    def set_algorithm_config(self, algorithm_config):
        self._algorithm_config = algorithm_config
        self.action_state.debug_max_step = algorithm_config["debug_max_step"]
        self.start_node_selection_look_ahead_distance = algorithm_config["trajectory_following_settings"]["look_ahead_distance"]
        self.average_elementary_action_error_threshold = algorithm_config["average_elementary_action_error_threshold"]
        self.motion_primitive_constraints_builder.set_algorithm_config(self._algorithm_config)

    def set_action_constraints(self, action_constraints):
        self.action_constraints = action_constraints
        self.motion_primitive_constraints_builder.set_action_constraints(
            self.action_constraints)
        self.motion_primitive_generator = MotionPrimitiveGenerator(self.action_constraints, self._algorithm_config)
        self.node_group = self.action_constraints.get_node_group()
        self.arc_length_of_end = self.motion_primitive_graph.nodes[
            self.node_group.get_random_end_state()].average_step_length

    def get_best_start_node(self, graph_walk, action_name):
        next_nodes = self.motion_primitive_graph.get_start_nodes(graph_walk, action_name)
        n_nodes = len(next_nodes)
        if n_nodes > 1:
            goal_arc_length = self.action_state.travelled_arc_length + self.start_node_selection_look_ahead_distance
            goal_position = self.action_constraints.root_trajectory.query_point_by_absolute_arc_length(goal_arc_length)
            constraint_desc = {"joint": "Hips", "canonical_keyframe": -1, "position": goal_position, "n_canonical_frames": 0,
                               "semanticAnnotation":  {"keyframeLabel": "end", "generated": True}}
            pos_constraint = GlobalTransformConstraint(self.motion_primitive_graph.skeleton, constraint_desc, 1.0, 1.0)
            mp_constraints = MotionPrimitiveConstraints()
            mp_constraints.start_pose = graph_walk.motion_vector.start_pose
            mp_constraints.constraints.append(pos_constraint)
            if graph_walk.get_num_of_frames() > 0:
                prev_frames = graph_walk.get_quat_frames()
            else:
                prev_frames = None

            errors = np.empty(n_nodes)
            index = 0
            for node_name in next_nodes:
                motion_primitive_node = self.motion_primitive_graph.nodes[(action_name, node_name)]
                self.motion_primitive_generator._search_for_best_fit_sample_in_cluster_tree(motion_primitive_node,
                                                                                        mp_constraints,
                                                                                        prev_frames)
                print "evaluated start option",node_name, mp_constraints.min_error
                errors[index] = mp_constraints.min_error
                index += 1
            min_idx = np.argmin(errors)
            next_node = next_nodes[min_idx]
            print "next node is", next_node, "with an error of", errors[min_idx], "towards",goal_position
            return (action_name, next_node)
        else:
            return (action_name, next_nodes[0])

    def _select_next_motion_primitive_node(self, graph_walk):
        """extract from graph based on previous last step + heuristic """

        if self.action_state.current_node is None:
            if self.action_constraints.root_trajectory is not None:
                next_node = self.get_best_start_node(graph_walk, self.action_constraints.action_name)
            else:
                next_node = self.motion_primitive_graph.get_random_action_transition(graph_walk, self.action_constraints.action_name)

            next_node_type = self.motion_primitive_graph.nodes[next_node].node_type
            if next_node is None:
                print "Error: Could not find a transition of type action_transition from ", self.action_state.prev_action_name, self.action_state.prev_mp_name, " to state", self.action_state.current_node
        elif len(self.motion_primitive_graph.nodes[self.action_state.current_node].outgoing_edges) > 0:
            next_node, next_node_type = self.node_group.get_random_transition(graph_walk, self.action_constraints, self.action_state.travelled_arc_length, self.arc_length_of_end)
            if next_node is None:
                print "Error: Could not find a transition of type", next_node_type, "from state", self.action_state.current_node
        else:
            print "Error: Could not find a transition from state", self.action_state.current_node
            next_node = self.motion_primitive_graph.node_groups[self.action_constraints.action_name].get_random_start_state()
            next_node_type = self.motion_primitive_graph.nodes[next_node].node_type

        return next_node, next_node_type

    def _update_travelled_arc_length(
            self, new_quat_frames, prev_graph_walk, prev_travelled_arc_length):
        """update travelled arc length based on new closest point on trajectory """
        if len(prev_graph_walk) > 0:
            min_arc_length = prev_graph_walk[-1].arc_length
        else:
            min_arc_length = 0.0

        closest_point, distance = self.action_constraints.root_trajectory.find_closest_point(new_quat_frames[-1][:3],  min_arc_length, -1)
        new_travelled_arc_length, eval_point = self.action_constraints.root_trajectory.get_absolute_arc_length_of_point(
            closest_point, min_arc_length=prev_travelled_arc_length)
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
            print "moved beyond end point using parameters",
            str(e.search_parameters)
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
        self.action_state.initialize_from_previous_graph_walk(graph_walk)
        print "start converting elementary action", self.action_constraints.action_name
        errors = [0]
        while not self.action_state.is_end_state():
            try:
                error = self._transition_to_next_action_state(graph_walk)
                errors.append(error)
            except ValueError, e:
                print e.message
                return False

        graph_walk.step_count += self.action_state.temp_step
        graph_walk.update_frame_annotation(self.action_constraints.action_name, self.action_state.action_start_frame, graph_walk.get_num_of_frames())
        avg_error = np.average(errors)
        print "reached end of elementary action", self.action_constraints.action_name, "with an average error of",avg_error
        return avg_error < self.average_elementary_action_error_threshold

    def _transition_to_next_action_state(self, graph_walk):

        new_node, new_node_type = self._select_next_motion_primitive_node(graph_walk)
        if new_node is None:
            raise ValueError("Failed to find a transition")
        print "transitioned to state", new_node

        mp_constraints = self._gen_motion_primitive_constraints(new_node, new_node_type, graph_walk)
        if mp_constraints is None:
            raise ValueError("Failed to generator constraints")
        new_motion_spline, new_parameters = self.motion_primitive_generator.generate_constrained_motion_spline(mp_constraints, graph_walk)

        new_arc_length = self._create_graph_walk_entry(new_node, new_motion_spline, mp_constraints, graph_walk)

        self.action_state.transition(new_node, new_node_type, new_arc_length, graph_walk.get_num_of_frames())

        return mp_constraints.min_error

    def _create_graph_walk_entry(self, new_node, new_motion_spline, constraints, graph_walk):
        """ Concatenate frames to motion and apply smoothing """
        prev_steps = graph_walk.steps
        graph_walk.append_quat_frames(new_motion_spline.get_motion_vector())

        if self.action_constraints.root_trajectory is not None:
            new_travelled_arc_length = self._update_travelled_arc_length(graph_walk.get_quat_frames(), prev_steps, self.action_state.travelled_arc_length)
        else:
            new_travelled_arc_length = 0
        #new_travelled_arc_length = mp_constraints.goal_arc_length

        new_step = GraphWalkEntry(self.motion_primitive_graph, new_node,
                                  new_motion_spline.low_dimensional_parameters,
                                  new_travelled_arc_length, self.action_state.step_start_frame,
                                  graph_walk.get_num_of_frames() - 1, constraints)
        graph_walk.steps.append(new_step)
        return new_travelled_arc_length


