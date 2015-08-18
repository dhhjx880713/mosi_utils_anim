__author__ = 'erhe01'

from ..utilities.exceptions import PathSearchError
from ..motion_model import NODE_TYPE_START, NODE_TYPE_END
from motion_primitive_sample_generator import MotionPrimitiveSampleGenerator
from constraint.motion_primitive_constraints_builder import MotionPrimitiveConstraintsBuilder
from constraint.time_constraints_builder import TimeConstraintsBuilder
from numerical_minimizer import NumericalMinimizer
from motion_sample import GraphWalkEntry
from objective_functions import obj_time_error_sum


class ElementaryActionSampleGeneratorState(object):
        def __init__(self):
            self.start_step = -1
            self.prev_action_name = None
            self.prev_mp_name = None
            self.start_frame = -1
            self.current_node = None
            self.current_node_type = ""
            self.temp_step = 0
            self.travelled_arc_length = 0.0

        def reset(self, motion):
            self.start_step = motion.step_count
            if self.start_step > 0:
                self.prev_action_name = motion.graph_walk[-1]
                self.prev_mp_name = motion.graph_walk[-1]
            else:
                self.prev_action_name = None
                self.prev_mp_name = None

            self.start_frame = motion.n_frames
            self.current_node = None # todo change all loop variables to class variables
            self.current_node_type = ""
            self.temp_step = 0
            self.travelled_arc_length = 0.0

class ElementaryActionSampleGenerator(object):

    def __init__(self, morphable_graph, algorithm_config):
        self.morphable_graph = morphable_graph
        self._algorithm_config = algorithm_config
        self.motion_primitive_constraints_builder = MotionPrimitiveConstraintsBuilder()
        self.motion_primitive_constraints_builder.set_algorithm_config(
            self._algorithm_config)
        self.numerical_minimizer = NumericalMinimizer(self._algorithm_config)
        self.numerical_minimizer.set_objective_function(obj_time_error_sum)
        self.state = ElementaryActionSampleGeneratorState()
        return

    def set_algorithm_config(self, algorithm_config):
        self._algorithm_config = algorithm_config
        self.motion_primitive_constraints_builder.set_algorithm_config(
            self._algorithm_config)

    def set_constraints(self, action_constraints):
        self.action_constraints = action_constraints
        self.motion_primitive_constraints_builder.set_action_constraints(
            self.action_constraints)

        self.motion_primitive_generator = MotionPrimitiveSampleGenerator(
            self.action_constraints, self._algorithm_config)
        self.node_group = self.action_constraints.get_node_group()
        self.arc_length_of_end = self.morphable_graph.nodes[
            self.node_group.get_random_end_state()].average_step_length

    def _select_next_motion_primitive_node_key(self, motion):
        """extract from graph based on previous last step + heuristic """

        if self.state.current_node is None:
            next_node = self.morphable_graph.get_random_action_transition(motion, self.action_constraints.action_name)
            next_node_type = NODE_TYPE_START
            if next_node is None:
                print "Error: Could not find a transition of type action_transition from ",\
                    self.state.prev_action_name, self.state.prev_mp_name, " to state", self.state.current_node
        elif len(self.morphable_graph.nodes[self.state.current_node].outgoing_edges) > 0:
            next_node, next_node_type = self.node_group.get_random_transition(
                motion, self.action_constraints, self.state.travelled_arc_length, self.arc_length_of_end)
            if next_node is None:
                print "Error: Could not find a transition of type", next_node_type, "from state", self.state.current_node
        else:
            print "Error: Could not find a transition from state", self.state.current_node

        return next_node, next_node_type

    def _update_travelled_arc_length(
            self, new_quat_frames, prev_motion, prev_travelled_arc_length):
        """update travelled arc length based on new closest point on trajectory """
        if len(prev_motion.graph_walk) > 0:
            min_arc_length = prev_motion.graph_walk[-1].arc_length
        else:
            min_arc_length = 0.0
        closest_point, distance = self.action_constraints.trajectory.find_closest_point(
            new_quat_frames[-1][:3], min_arc_length=min_arc_length)
        new_travelled_arc_length, eval_point = self.action_constraints.trajectory.get_absolute_arc_length_of_point(
            closest_point, min_arc_length=prev_travelled_arc_length)
        if new_travelled_arc_length == -1:
            new_travelled_arc_length = self.action_constraints.trajectory.full_arc_length
        return new_travelled_arc_length

    def _update_annotated_motion(
            self, current_state, quat_frames, motion_primitive_constraints, motion):
        """ Concatenate frames to motion and apply smoothing """
        canonical_keyframe_labels = self.node_group.get_canonical_keyframe_labels(
            current_state[1])
        start_frame = motion.n_frames
        motion.append_quat_frames(quat_frames)
        last_frame = motion.n_frames - 1
        motion.update_action_list(
            motion_primitive_constraints.constraints,
            self.action_constraints.keyframe_annotations,
            canonical_keyframe_labels,
            start_frame,
            last_frame)

    def _get_motion_primitive_constraints_from_action_constraints(
            self, current_state, current_motion_primitive_type, prev_motion, travelled_arc_length):
        try:
            is_last_step = (current_motion_primitive_type == NODE_TYPE_END)
            print current_state
            self.motion_primitive_constraints_builder.set_status(
                current_state[1], travelled_arc_length, prev_motion.quat_frames, is_last_step)
            return self.motion_primitive_constraints_builder.build()

        except PathSearchError as e:
            print "moved beyond end point using parameters",
            str(e.search_parameters)
            return None

    def append_elementary_action_to_motion(self, motion):
        """Convert an entry in the elementary action list to a list of quaternion frames.
        Note only one trajectory constraint per elementary action is currently supported
        and it should be for the Hip joint.

        If there is a trajectory constraint it is used otherwise a random graph walk is used
        if there is a keyframe constraint it is assigned to the motion primitives
        in the graph walk

        Paramaters
        ---------
        * motion: MotionGeneratorResult
            Result object contains the intermediary result of the motion generation process. The animation keyframes of
            the elementary action will be appended to the frames in this object
        Returns
        -------
        * success: Bool
            True if successful and False, if an error occurred during the constraints generation
        """
        self.state.reset(motion)
        print "start converting elementary action", self.action_constraints.action_name
        while self.state.current_node_type != NODE_TYPE_END:

            if self._algorithm_config["debug_max_step"] > -1 and motion.step_count + \
                    self.state.temp_step > self._algorithm_config["debug_max_step"]:
                print "reached max step"
                break
            self.state.current_node, self.state.current_node_type = self._select_next_motion_primitive_node_key(motion)
            if self.state.current_node is None:
                break
            print "transitioned to state", self.state.current_node
            motion_primitive_constraints = self._get_motion_primitive_constraints_from_action_constraints(
                self.state.current_node, self.state.current_node_type, motion, self.state.travelled_arc_length)
            if motion_primitive_constraints is None:
                return False

            motion_primitive_sample = self.motion_primitive_generator.generate_motion_primitive_sample_from_constraints(
                motion_primitive_constraints, motion)

            self._update_annotated_motion(
                self.state.current_node,
                motion_primitive_sample.get_motion_vector(False),
                motion_primitive_constraints,
                motion)
            if self.action_constraints.trajectory is not None:
                self.state.travelled_arc_length = self._update_travelled_arc_length(
                    motion.quat_frames, motion, self.state.travelled_arc_length)
            motion.graph_walk.append(
                GraphWalkEntry(
                    self.state.current_node,
                    motion_primitive_sample.low_dimensional_parameters,
                    self.state.travelled_arc_length))
            self.state.temp_step += 1

        motion.step_count += self.state.temp_step
        motion.update_frame_annotation(
            self.action_constraints.action_name,
            self.state.start_frame,
            motion.n_frames)
        if self._algorithm_config["use_global_optimization"]:
            self._optimize_over_graph_walk(motion.graph_walk)

        print "reached end of elementary action", self.action_constraints.action_name
        return True

    def _optimize_over_graph_walk(self, motion):
        time_constraints = TimeConstraintsBuilder(
            self.action_constraints, motion, self.state.start_step).build()
        data = (self.morphable_graph, motion, time_constraints)
        self.numerical_minimizer.set_objective_function_parameters(data)
        self.action_constraints
