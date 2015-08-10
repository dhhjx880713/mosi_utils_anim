__author__ = 'erhe01'


from utilities.exceptions import SynthesisError, PathSearchError
from motion_model import NODE_TYPE_START, NODE_TYPE_STANDARD, NODE_TYPE_END
from motion_primitive_generator import MotionPrimitiveGenerator
from constraint.motion_primitive_constraints_builder import MotionPrimitiveConstraintsBuilder
from motion_generator_result import GraphWalkEntry

class ElementaryActionGenerator(object):
    def __init__(self, morphable_graph, algorithm_config):
        self.morphable_graph = morphable_graph
        self._algorithm_config =  algorithm_config
        self.motion_primitive_constraints_builder = MotionPrimitiveConstraintsBuilder()
        self.motion_primitive_constraints_builder.set_algorithm_config(self._algorithm_config)
        return

    def set_algorithm_config(self, algorithm_config):
        self._algorithm_config = algorithm_config
        self.motion_primitive_constraints_builder.set_algorithm_config(self._algorithm_config)

    def set_constraints(self,action_constraints):
        self.action_constraints = action_constraints
        self.motion_primitive_constraints_builder.set_action_constraints(self.action_constraints)

        self.motion_primitive_generator = MotionPrimitiveGenerator(self.action_constraints, self._algorithm_config)
        #skeleton = action_constraints.get_skeleton()
        self.node_group = self.action_constraints.get_node_group()
        self.arc_length_of_end = self.morphable_graph.nodes[self.node_group.get_random_end_state()].average_step_length

    def append_elementary_action_to_motion(self, motion):
        """Convert an entry in the elementary action list to a list of quaternion frames.
        Note only one trajectory constraint per elementary action is currently supported
        and it should be for the Hip joint.

        If there is a trajectory constraint it is used otherwise a random graph walk is used
        if there is a keyframe constraint it is assigned to the motion primitves
        in the graph walk

        Paramaters
        ---------
        * elementary_action : string
          the identifier of the elementary action

        * constraint_list : list of dict
         the constraints element from the elementary action list entry

        * morphable_graph : MorphableGraph
        \t An instance of the MorphableGraph.
        * start_pose : dict
         Contains orientation and position as lists with three elements

        * keyframe_annotations : dict of dicts
          Contains a dict of events/actions associated with certain keyframes

        Returns
        -------
        * motion: MotionGeneratorResult
        """

        if motion.step_count >0:
             prev_action_name = motion.graph_walk[-1]
             prev_mp_name = motion.graph_walk[-1]
        else:
             prev_action_name = None
             prev_mp_name = None

        start_frame = motion.n_frames
        #create sequence of list motion primitives,arc length and number of frames for backstepping
        current_state = None
        current_motion_primitive_type = ""
        temp_step = 0
        travelled_arc_length = 0.0
        print "start converting elementary action",self.action_constraints.action_name
        while current_motion_primitive_type != NODE_TYPE_END:

            if self._algorithm_config["debug_max_step"]  > -1 and motion.step_count + temp_step > self._algorithm_config["debug_max_step"]:
                print "reached max step"
                break
            #######################################################################
            # Get motion primitive = extract from graph based on previous last step + heuristic
            if current_state is None:
                 current_state = self.morphable_graph.get_random_action_transition(motion, self.action_constraints.action_name)
                 current_motion_primitive_type = NODE_TYPE_START
                 if current_state is None:

                     print "Error: Could not find a transition of type action_transition from ",prev_action_name,prev_mp_name ," to state",current_state
                     break
            elif len(self.morphable_graph.nodes[current_state].outgoing_edges) > 0:
                prev_state = current_state
                current_state, current_motion_primitive_type = self.node_group.get_random_transition(motion, self.action_constraints, travelled_arc_length, self.arc_length_of_end)
                if current_state is None:
                     print "Error: Could not find a transition of type",current_motion_primitive_type,"from state",prev_state
                     break
            else:
                print "Error: Could not find a transition from state",current_state
                break

            print "transitioned to state",current_state
            #######################################################################
            #Generate constraints from action_constraints

            try:
                is_last_step = (current_motion_primitive_type == NODE_TYPE_END)
                print current_state
                self.motion_primitive_constraints_builder.set_status(current_state[1], travelled_arc_length, motion.quat_frames, is_last_step)
                motion_primitive_constraints = self.motion_primitive_constraints_builder.build()

            except PathSearchError as e:
                    print "moved beyond end point using parameters",
                    str(e.search_parameters)
                    return False


            # get optimal parameters, Back-project to frames in joint angle space,
            # Concatenate frames to motion and apply smoothing
            tmp_quat_frames, parameters = self.motion_primitive_generator.generate_motion_primitive_from_constraints(motion_primitive_constraints, motion)

            #update annotated motion
            canonical_keyframe_labels = self.node_group.get_canonical_keyframe_labels(current_state[1])
            start_frame = motion.n_frames
            motion.append_quat_frames(tmp_quat_frames)
            last_frame = motion.n_frames-1
            motion.update_action_list(motion_primitive_constraints.constraints, self.action_constraints.keyframe_annotations, canonical_keyframe_labels, start_frame, last_frame)

            #update arc length based on new closest point
            if self.action_constraints.trajectory is not None:
                if len(motion.graph_walk) > 0:
                    min_arc_length = motion.graph_walk[-1].arc_length
                else:
                    min_arc_length = 0.0
                closest_point,distance = self.action_constraints.trajectory.find_closest_point(motion.quat_frames[-1][:3],min_arc_length=min_arc_length)
                travelled_arc_length,eval_point = self.action_constraints.trajectory.get_absolute_arc_length_of_point(closest_point,min_arc_length=travelled_arc_length)
                if travelled_arc_length == -1 :
                    travelled_arc_length = self.action_constraints.trajectory.full_arc_length

            #update graph walk of motion
            graph_walk_entry = GraphWalkEntry(self.action_constraints.action_name,current_state[1], parameters, travelled_arc_length)
            motion.graph_walk.append(graph_walk_entry)

            temp_step += 1

        motion.step_count += temp_step
        motion.update_frame_annotation(self.action_constraints.action_name, start_frame, motion.n_frames)

        print "reached end of elementary action", self.action_constraints.action_name
#        if self._algorithm_config["active_global_optimization"]:
#            optimize_globally(motion.graph_walk, start_step, action_constraints)
    #    if trajectory is not None:
    #        print "info", trajectory.full_arc_length, \
    #               travelled_arc_length,arc_length_of_end, \
    #               np.linalg.norm(trajectory.get_last_control_point() - quat_frames[-1][:3]), \
    #               check_end_condition(morphable_subgraph,quat_frames,trajectory,\
    #                                        travelled_arc_length,arc_length_of_end)


        return True

