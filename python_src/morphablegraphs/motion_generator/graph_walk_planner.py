import numpy as np
from copy import copy, deepcopy
import random
from constraints.motion_primitive_constraints import MotionPrimitiveConstraints
from constraints.spatial_constraints.keyframe_constraints.global_transform_constraint import GlobalTransformConstraint
from constraints.spatial_constraints.keyframe_constraints.direction_2d_constraint import Direction2DConstraint
from ..animation_data.motion_editing import create_transformation_matrix
from ..utilities import write_log
from .graph_walk import GraphWalk, GraphWalkEntry
from ea_state import ElementaryActionGeneratorState


class PlannerState(object):
    def __init__(self, current_node, graph_walk, travelled_arc_length):
        self.graph_walk = graph_walk
        self.travelled_arc_length = travelled_arc_length
        self.current_node = current_node

    def create_copy(self):
        current_node = copy(self.current_node)
        travelled_arc_length = self.travelled_arc_length
        graph_walk = GraphWalk(self.graph_walk.motion_state_graph, self.graph_walk.mg_input, self.graph_walk._algorithm_config)
        graph_walk.motion_vector.frames = [deepcopy(self.graph_walk.motion_vector.frames[-1])]
        return PlannerState(current_node, graph_walk, travelled_arc_length)


class GraphWalkPlanner(object):
    def __init__(self, motion_state_graph,  algorithm_config):
        self.motion_state_graph = motion_state_graph
        self.step_look_ahead_distance = algorithm_config["trajectory_following_settings"]["look_ahead_distance"]
        self.use_local_coordinates = algorithm_config["use_local_coordinates"]
        self.mp_generator = None
        self.state = None
        self.action_constraints = None
        self.arc_length_of_end = 0.0
        self.node_group = None
        self.trajectory = None
        self._n_steps_looking_ahead = 1

    def set_state(self, graph_walk, mp_generator, action_state, action_constraints, arc_length_of_end):
        self.mp_generator = mp_generator
        self.state = PlannerState(action_state.current_node, graph_walk, action_state.travelled_arc_length)
        self.action_constraints = action_constraints
        self.trajectory = action_constraints.root_trajectory
        self.arc_length_of_end = arc_length_of_end
        self.node_group = self.action_constraints.get_node_group()

    def get_best_start_node(self):
        start_nodes = self.motion_state_graph.get_start_nodes(self.state.graph_walk, self.action_constraints.action_name)
        n_nodes = len(start_nodes)
        if n_nodes > 1:
            options = [(self.action_constraints.action_name, next_node) for next_node in start_nodes]
            return self.select_next_step(self.state, options, add_orientation=False)
        else:
            return self.action_constraints.action_name, start_nodes[0]

    def get_transition_options(self, state):
        if self.trajectory is not None:
            next_node_type = self.node_group.get_transition_type_for_action_from_trajectory(state.graph_walk,
                                                                                            self.action_constraints,
                                                                                            state.travelled_arc_length,
                                                                                            self.arc_length_of_end)
        else:
            next_node_type = self.node_group.get_transition_type_for_action(state.graph_walk, self.action_constraints)
        edges = self.motion_state_graph.nodes[self.state.current_node].outgoing_edges
        options = [edge_key for edge_key in edges.keys() if edges[edge_key].transition_type == next_node_type]
        #print "options",next_node_type, options
        return options, next_node_type

    def get_best_transition_node(self):
        options, next_node_type = self.get_transition_options(self.state)
        n_transitions = len(options)
        if n_transitions == 1:
            next_node = options[0]
        elif n_transitions > 1:
            if self.trajectory is not None:
                next_node = self.select_next_step(self.state, options, add_orientation=False)
            else:  # use random transition if there is no path to follow
                random_index = random.randrange(0, n_transitions, 1)
                next_node = options[random_index]
        else:
            write_log("Error: Could not find a transition from state", self.state.current_node,
                      len(self.motion_state_graph.nodes[self.state.current_node].outgoing_edges))
            next_node = self.node_group.get_random_start_state()
            next_node_type = self.motion_state_graph.nodes[next_node].node_type
        if next_node is None:
            write_log("Error: Could not find a transition of type", next_node_type, "from state",  self.state.current_node)
        return next_node, next_node_type

    def _add_constraint_with_orientation(self, constraint_desc, goal_arc_length, mp_constraints):
        goal_position, tangent_line = self.trajectory.get_tangent_at_arc_length(goal_arc_length)
        constraint_desc["position"] = goal_position.tolist()
        pos_constraint = GlobalTransformConstraint(self.motion_state_graph.skeleton, constraint_desc, 1.0, 1.0)
        mp_constraints.constraints.append(pos_constraint)
        dir_constraint_desc = {"joint": "Hips", "canonical_keyframe": -1, "dir_vector": tangent_line,
                               "semanticAnnotation": {"keyframeLabel": "end", "generated": True}}
        # TODO add weight to configuration
        dir_constraint = Direction2DConstraint(self.motion_state_graph.skeleton, dir_constraint_desc, 1.0, 1.0)
        mp_constraints.constraints.append(dir_constraint)

    def _add_constraint(self, constraint_desc, goal_arc_length, mp_constraints):
        constraint_desc["position"] = self.trajectory.query_point_by_absolute_arc_length(goal_arc_length).tolist()
        pos_constraint = GlobalTransformConstraint(self.motion_state_graph.skeleton, constraint_desc, 1.0, 1.0)
        mp_constraints.constraints.append(pos_constraint)

    def _generate_node_evaluation_constraints(self, state, add_orientation=False):
        goal_arc_length = state.travelled_arc_length + self.step_look_ahead_distance
        mp_constraints = MotionPrimitiveConstraints()
        mp_constraints.skeleton = self.motion_state_graph.skeleton
        mp_constraints.aligning_transform = create_transformation_matrix(state.graph_walk.motion_vector.start_pose["position"], state.graph_walk.motion_vector.start_pose["orientation"])
        mp_constraints.start_pose = state.graph_walk.motion_vector.start_pose
        constraint_desc = {"joint": "Hips", "canonical_keyframe": -1, "n_canonical_frames": 0,
                           "semanticAnnotation": {"keyframeLabel": "end", "generated": True}}
        if add_orientation:
            self._add_constraint_with_orientation(constraint_desc, goal_arc_length, mp_constraints)
        else:
            self._add_constraint(constraint_desc, goal_arc_length, mp_constraints)

        if self.use_local_coordinates and False:
            mp_constraints = mp_constraints.transform_constraints_to_local_cos()

        return mp_constraints

    def select_next_step(self, state, options, add_orientation=False):
        #next_node = self._look_one_step_ahead(state, options, add_orientation)
        mp_constraints = self._generate_node_evaluation_constraints(state, add_orientation)
        if state.current_node is None or True:
            errors, s_vectors = self._evaluate_options(state, mp_constraints, options)
        else:
            errors, s_vectors = self._evaluate_options_looking_ahead(state, mp_constraints, options, add_orientation)
        min_idx = np.argmin(errors)
        next_node = options[min_idx]
        write_log("####################################Next node is", next_node)#, "with an error of", errors[min_idx]
        return next_node

    def _evaluate_option(self, node_name, mp_constraints, prev_frames):
        motion_primitive_node = self.motion_state_graph.nodes[node_name]
        canonical_keyframe = motion_primitive_node.get_n_canonical_frames() - 1
        for c in mp_constraints.constraints:
            c.canonical_keyframe = canonical_keyframe
        s_vector = self.mp_generator._get_best_fit_sample_using_cluster_tree(motion_primitive_node, mp_constraints,
                                                                             prev_frames, 1)

        write_log("Evaluated option", node_name, mp_constraints.min_error)
        return s_vector, mp_constraints.min_error

    def _evaluate_options(self, state, mp_constraints, options):
        errors = np.empty(len(options))
        s_vectors = []
        index = 0
        for node_name in options:
            #print "option", node_name
            s_vector, error = self._evaluate_option(node_name, mp_constraints, state.graph_walk.motion_vector.frames)
            errors[index] = error
            s_vectors.append(s_vector)
            index += 1
        return errors, s_vectors

    def _evaluate_options_looking_ahead(self, state, mp_constraints, options, add_orientation=False):
        errors = np.empty(len(options))
        next_node = options[0]
        #TODO add state fork
        index = 0
        for node_name in options:
            print "evaluate",node_name
            node_state = state.create_copy()
            step_count = 0

            while step_count < self._n_steps_looking_ahead:
                s_vector, error = self._evaluate_option(node_name, mp_constraints, state.graph_walk.motion_vector.frames)
                #write_log("Evaluated option", node_name, mp_constraints.min_error,"at level", n_steps)
                errors[index] += error
                self._update_path(node_state, node_name, s_vector)
                print "advance along",node_name
                #if node_state.current_node is not None:
                #    new_options, next_node_type = self.get_transition_options(node_state)
                    #errors[index] += self._look_ahead_deeper(node_state, new_options, self._n_steps_looking_ahead, add_orientation)
                step_count += 1

            index += 1
        min_idx = np.argmin(errors)
        next_node = options[min_idx]
        return next_node

    def _look_ahead_deeper(self, state, options, max_depth, add_orientation=False):
        print "#####################################Look deeper from", state.current_node
        mp_constraints = self._generate_node_evaluation_constraints(state, add_orientation)
        errors, s_vectors = self._evaluate_options(state, mp_constraints, options)
        min_idx = np.argmin(errors)
        error = errors[min_idx]
        self._update_path(state, options[min_idx], s_vectors[min_idx])
        if max_depth > 0:
            new_options, next_node_type = self.get_transition_options(state)
            error += self._look_ahead_deeper(state, new_options, max_depth-1, add_orientation)
        return error

    def _update_path(self, state, next_node, s_vector):
        motion_spline = self.motion_state_graph.nodes[next_node].back_project(s_vector, use_time_parameters=False)
        state.graph_walk.append_quat_frames(motion_spline.get_motion_vector())
        max_arc_length = state.travelled_arc_length + self.step_look_ahead_distance  # was originally set to 80
        closest_point, distance = self.trajectory.find_closest_point(state.graph_walk.motion_vector.frames[-1][:3],state.travelled_arc_length,max_arc_length)
        new_travelled_arc_length, eval_point = self.trajectory.get_absolute_arc_length_of_point(closest_point, min_arc_length=state.travelled_arc_length)
        if new_travelled_arc_length == -1:
            new_travelled_arc_length = self.trajectory.full_arc_length
        state.travelled_arc_length = new_travelled_arc_length
        new_step = GraphWalkEntry(self.motion_state_graph, next_node, s_vector, new_travelled_arc_length, 0, 0, None)
        state.graph_walk.steps.append(new_step)
        state.current_node = next_node
