__author__ = 'erhe01'


from utilities.exceptions import SynthesisError, PathSearchError
from motion_model import NODE_TYPE_START, NODE_TYPE_STANDARD, NODE_TYPE_END
from motion_primitive_generator import MotionPrimitiveGenerator
from constraint.motion_primitive_constraints_builder import MotionPrimitiveConstraintsBuilder
from motion_generator_result import GraphWalkEntry


class ElementaryActionGenerator(object):

    def __init__(self, morphable_graph, algorithm_config):
        self.morphable_graph = morphable_graph
        self._algorithm_config = algorithm_config
        self.motion_primitive_constraints_builder = MotionPrimitiveConstraintsBuilder()
        self.motion_primitive_constraints_builder.set_algorithm_config(
            self._algorithm_config)
        return

    def set_algorithm_config(self, algorithm_config):
        self._algorithm_config = algorithm_config
        self.motion_primitive_constraints_builder.set_algorithm_config(
            self._algorithm_config)

    def set_constraints(self, action_constraints):
        self.action_constraints = action_constraints
        self.motion_primitive_constraints_builder.set_action_constraints(
            self.action_constraints)

        self.motion_primitive_generator = MotionPrimitiveGenerator(
            self.action_constraints, self._algorithm_config)
        self.node_group = self.action_constraints.get_node_group()
        self.arc_length_of_end = self.morphable_graph.nodes[
            self.node_group.get_random_end_state()].average_step_length

    def _select_next_motion_primitive_node_key(
            self, state, motion, prev_action_name, prev_mp_name, travelled_arc_length):
        """extract from graph based on previous last step + heuristic """
        if state is None:
            next_state = self.morphable_graph.get_random_action_transition(
                motion, self.action_constraints.action_name)
            next_motion_primitive_type = NODE_TYPE_START
            if next_state is None:
                print "Error: Could not find a transition of type action_transition from ",\
                    prev_action_name, prev_mp_name, " to state", state
        elif len(self.morphable_graph.nodes[state].outgoing_edges) > 0:
            next_state, next_motion_primitive_type = self.node_group.get_random_transition(
                motion, self.action_constraints, travelled_arc_length, self.arc_length_of_end)
            if next_state is None:
                print "Error: Could not find a transition of type", next_motion_primitive_type, "from state", state
        else:
            print "Error: Could not find a transition from state", state

        return next_state, next_motion_primitive_type

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
            self, current_state, quat_frames, parameters, motion_primitive_constraints, motion):
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

    def _update_graph_walk(self, motion, current_state,
                           parameters, travelled_arc_length):
        graph_walk_entry = GraphWalkEntry(
            self.action_constraints.action_name,
            current_state[1],
            parameters,
            travelled_arc_length)
        motion.graph_walk.append(graph_walk_entry)

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

        if motion.step_count > 0:
            prev_action_name = motion.graph_walk[-1]
            prev_mp_name = motion.graph_walk[-1]
        else:
            prev_action_name = None
            prev_mp_name = None

        start_frame = motion.n_frames
        current_state = None
        current_motion_primitive_type = ""
        temp_step = 0
        travelled_arc_length = 0.0
        print "start converting elementary action", self.action_constraints.action_name
        while current_motion_primitive_type != NODE_TYPE_END:

            if self._algorithm_config["debug_max_step"] > -1 and motion.step_count + \
                    temp_step > self._algorithm_config["debug_max_step"]:
                print "reached max step"
                break
            current_state, current_motion_primitive_type = self._select_next_motion_primitive_node_key(
                current_state, motion, prev_action_name, prev_mp_name, travelled_arc_length)
            if current_state is None:
                break
            print "transitioned to state", current_state
            motion_primitive_constraints = self._get_motion_primitive_constraints_from_action_constraints(
                current_state, current_motion_primitive_type, motion, travelled_arc_length)
            if motion_primitive_constraints is None:
                return False

            tmp_quat_frames, parameters = self.motion_primitive_generator.generate_motion_primitive_from_constraints(
                motion_primitive_constraints, motion)

            self._update_annotated_motion(
                current_state,
                tmp_quat_frames,
                parameters,
                motion_primitive_constraints,
                motion)
            if self.action_constraints.trajectory is not None:
                travelled_arc_length = self._update_travelled_arc_length(
                    motion.quat_frames, motion, travelled_arc_length)
            self._update_graph_walk(
                motion,
                current_state,
                parameters,
                travelled_arc_length)
            temp_step += 1

        motion.step_count += temp_step
        motion.update_frame_annotation(
            self.action_constraints.action_name,
            start_frame,
            motion.n_frames)
        print "reached end of elementary action", self.action_constraints.action_name
        return True
#        if self._algorithm_config["active_global_optimization"]:
#            optimize_globally(motion.graph_walk, start_step, action_constraints)
