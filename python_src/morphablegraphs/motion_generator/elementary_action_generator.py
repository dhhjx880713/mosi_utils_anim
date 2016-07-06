__author__ = 'erhe01'

import numpy as np
from copy import copy, deepcopy
import json
import urllib2
import random
from ..utilities.exceptions import PathSearchError
from ..motion_model import NODE_TYPE_END, NODE_TYPE_SINGLE, NODE_TYPE_CYCLE_END
from ..animation_data.motion_editing import create_transformation_matrix
from motion_primitive_generator import MotionPrimitiveGenerator
from constraints.motion_primitive_constraints_builder import MotionPrimitiveConstraintsBuilder
from graph_walk import GraphWalkEntry
from constraints.motion_primitive_constraints import MotionPrimitiveConstraints
from constraints.spatial_constraints.keyframe_constraints.global_transform_constraint import GlobalTransformConstraint
from constraints.spatial_constraints.keyframe_constraints.direction_2d_constraint import Direction2DConstraint
from constraints.spatial_constraints.keyframe_constraints.global_transform_ca_constraint import GlobalTransformCAConstraint
from constraints import CA_CONSTRAINTS_MODE_DIRECT_CONNECTION
from ..utilities import write_log, get_bvh_writer
from ..animation_data.motion_editing import align_quaternion_frames, fast_quat_frames_transformation, transform_quaternion_frames, euler_angles_to_rotation_matrix


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
            self.max_arc_length = np.inf
            self.action_cycled_next = False

        def initialize_from_previous_graph_walk(self, graph_walk, max_arc_length, action_cycled_next):
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
            self.max_arc_length = max_arc_length
            self.action_cycled_next = action_cycled_next

        def is_end_state(self):
            return self.is_last_node() or self.reached_debug_max_step() or self.reached_max_arc_length()

        def reached_debug_max_step(self):
            return self.start_step + self.temp_step > self.debug_max_step and self.debug_max_step > -1

        def reached_max_arc_length(self):
            return self.travelled_arc_length >= self.max_arc_length

        def is_last_node(self):
            return self.current_node_type == NODE_TYPE_END or \
                   self.current_node_type == NODE_TYPE_SINGLE or\
                   (self.current_node is not None and self.action_cycled_next)

        def transition(self, new_node, new_node_type, new_travelled_arc_length, new_step_start_frame):
            self.current_node = new_node
            self.current_node_type = new_node_type
            self.travelled_arc_length = new_travelled_arc_length
            self.step_start_frame = new_step_start_frame
            self.temp_step += 1


class ElementaryActionGenerator(object):
    def __init__(self, motion_primitive_graph, algorithm_config, service_config):
        self.motion_state_graph = motion_primitive_graph
        self._algorithm_config = algorithm_config
        self.motion_primitive_constraints_builder = MotionPrimitiveConstraintsBuilder()
        self.motion_primitive_constraints_builder.set_algorithm_config(self._algorithm_config)
        self.action_state = ElementaryActionGeneratorState(self._algorithm_config)
        self.step_look_ahead_distance = algorithm_config["trajectory_following_settings"]["look_ahead_distance"]
        self.average_elementary_action_error_threshold = algorithm_config["average_elementary_action_error_threshold"]
        self.use_local_coordinates = self._algorithm_config["use_local_coordinates"]
        self.end_step_length_factor = algorithm_config["trajectory_following_settings"]["end_step_length_factor"]
        self.max_distance_to_path = algorithm_config["trajectory_following_settings"]["max_distance_to_path"]
        self.activate_direction_ca_connection = algorithm_config["collision_avoidance_constraints_mode"] == CA_CONSTRAINTS_MODE_DIRECT_CONNECTION
        self.activate_coordinate_transform_for_ca = service_config["activate_coordinate_transform"]
        self.ca_service_url = service_config["collision_avoidance_service_url"]
        self.coordinate_transform_matrix = np.array([[1,0,0,0],
                                                    [0,0,-1,0],
                                                    [0,1,0,0],
                                                    [0,0,0,1]])


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

    def generate_node_evaluation_constraints(self, graph_walk, add_orientation=False):
        goal_arc_length = self.action_state.travelled_arc_length + self.step_look_ahead_distance
        mp_constraints = MotionPrimitiveConstraints()
        mp_constraints.skeleton = self.action_constraints.get_skeleton()
        mp_constraints.aligning_transform = create_transformation_matrix(graph_walk.motion_vector.start_pose["position"],
                                                                         graph_walk.motion_vector.start_pose["orientation"])
        mp_constraints.start_pose = graph_walk.motion_vector.start_pose
        constraint_desc = {"joint": "Hips", "canonical_keyframe": -1,  "n_canonical_frames": 0,
                               "semanticAnnotation":  {"keyframeLabel": "end", "generated": True}}
        if add_orientation:
            goal_position, tangent_line = self.action_constraints.root_trajectory.get_tangent_at_arc_length(goal_arc_length)
            constraint_desc["position"] = goal_position.tolist()
            pos_constraint = GlobalTransformConstraint(self.motion_state_graph.skeleton, constraint_desc, 1.0, 1.0)
            mp_constraints.constraints.append(pos_constraint)
            dir_constraint_desc = {"joint": "Hips", "canonical_keyframe": -1, "dir_vector":tangent_line,
                                   "semanticAnnotation":  {"keyframeLabel": "end", "generated": True}}
            dir_constraint = Direction2DConstraint(self.motion_state_graph.skeleton, dir_constraint_desc, 1.0, 1.0)#TODO add weight to configuration
            mp_constraints.constraints.append(dir_constraint)
        else:
            constraint_desc["position"] = self.action_constraints.root_trajectory.query_point_by_absolute_arc_length(goal_arc_length).tolist()

            pos_constraint = GlobalTransformConstraint(self.motion_state_graph.skeleton, constraint_desc, 1.0, 1.0)
            mp_constraints.constraints.append(pos_constraint)
        return mp_constraints

    def _evaluate_multiple_path_following_options(self, graph_walk, options, add_orientation=False):
        mp_constraints = self.generate_node_evaluation_constraints(graph_walk, add_orientation)

        prev_frames = None
        if self.use_local_coordinates and False:
            mp_constraints = mp_constraints.transform_constraints_to_local_cos()
        elif graph_walk.get_num_of_frames() > 0:
            prev_frames = graph_walk.get_quat_frames()

        errors = np.empty(len(options))
        index = 0
        for node_name in options:
            motion_primitive_node = self.motion_state_graph.nodes[node_name]
            canonical_keyframe = motion_primitive_node.get_n_canonical_frames() - 1
            for c in mp_constraints.constraints:
                c.canonical_keyframe = canonical_keyframe
            self.motion_primitive_generator._get_best_fit_sample_using_cluster_tree(motion_primitive_node,
                                                                                    mp_constraints, prev_frames, 1)
            write_log("Evaluated option", node_name, mp_constraints.min_error)
            errors[index] = mp_constraints.min_error
            index += 1
        min_idx = np.argmin(errors)
        next_node = options[min_idx]
        write_log("Next node is", next_node, "with an error of", errors[min_idx])
        return next_node

    def get_best_start_node(self, graph_walk, action_name):
        start_nodes = self.motion_state_graph.get_start_nodes(graph_walk, action_name)
        n_nodes = len(start_nodes)
        if n_nodes > 1:
            options = [(action_name, next_node) for next_node in start_nodes]
            return self._evaluate_multiple_path_following_options(graph_walk, options, add_orientation=False)
        else:
            return action_name, start_nodes[0]

    def _get_best_transition_node(self, graph_walk):
        if self.action_constraints.root_trajectory is not None:
            next_node_type = self.node_group.get_transition_type_for_action_from_trajectory(graph_walk, self.action_constraints,
                                                             self.action_state.travelled_arc_length,
                                                             self.arc_length_of_end)
        else:
            next_node_type = self.node_group.get_transition_type_for_action(graph_walk, self.action_constraints)
        edges = self.motion_state_graph.nodes[self.action_state.current_node].outgoing_edges
        options = [edge_key for edge_key in edges.keys() if edges[edge_key].transition_type == next_node_type]
        n_transitions = len(options)
        if n_transitions == 1:
            next_node = options[0]
        elif n_transitions > 1:
            if self.action_constraints.root_trajectory is not None:
                next_node = self._evaluate_multiple_path_following_options(graph_walk, options, add_orientation=False)
            else:  # use random transition if there is no path to follow
                random_index = random.randrange(0, n_transitions, 1)
                next_node = options[random_index]
        else:
            write_log("Error: Could not find a transition from state", self.action_state.current_node, len(self.motion_state_graph.nodes[self.action_state.current_node].outgoing_edges))
            next_node = self.motion_state_graph.node_groups[self.action_constraints.action_name].get_random_start_state()
            next_node_type = self.motion_state_graph.nodes[next_node].node_type
        if next_node is None:
           write_log("Error: Could not find a transition of type", next_node_type, "from state", self.action_state.current_node)
        return next_node, next_node_type

    def _select_next_motion_primitive_node(self, graph_walk):
        """extract from graph based on previous step and heuristic """
        if self.action_state.current_node is None:  # is start state
            if self.action_constraints.root_trajectory is not None: # use trajectory to determine best start
                next_node = self.get_best_start_node(graph_walk, self.action_constraints.action_name)
            else:
                next_node = self.motion_state_graph.get_random_action_transition(graph_walk, self.action_constraints.action_name, self.action_constraints.cycled_previous)

            next_node_type = self.motion_state_graph.nodes[next_node].node_type
            if next_node is None:
                write_log("Error: Could not find a transition of type action_transition from ", self.action_state.prev_action_name, self.action_state.prev_mp_name, " to state", self.action_state.current_node)

        else:  # is intermediate start state
            next_node, next_node_type = self._get_best_transition_node(graph_walk)
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
        if self.activate_direction_ca_connection and self.ca_service_url is not None:
            ca_constraints = self._get_collision_avoidance_constraints(new_node, new_motion_spline, graph_walk)
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

    def _get_collision_avoidance_constraints(self, new_node, new_motion_spline, graph_walk):
        """ Generate constraints using the rest interface of the collision avoidance module directly.
            #TODO move to wrapper
        """
        aligned_motion_spline, global_transformation = self._get_aligned_motion_spline(new_motion_spline, graph_walk.get_quat_frames())
        if self.activate_coordinate_transform_for_ca:
            global_transformation = np.dot(global_transformation, self.coordinate_transform_matrix)
        frames = aligned_motion_spline.get_motion_vector()
        global_bvh_string = get_bvh_writer(self.motion_state_graph.skeleton, frames).generate_bvh_string()
        ca_input = {"elementary_action_name": new_node[0],
                    "motion_primitive_name": new_node[1],
                    "global_transform": global_transformation.tolist(),
                    "global_bvh_frames": global_bvh_string}

        ca_output = self._call_ca_rest_interface(ca_input)
        if ca_output is not None:
            return self._create_ca_constraints(new_node, ca_output, graph_walk)
        else:
            return None

    def _get_aligned_motion_spline(self, new_motion_spline, prev_frames):
        aligned_motion_spline = deepcopy(new_motion_spline)
        if prev_frames is not None:
            angle, offset = fast_quat_frames_transformation(prev_frames, new_motion_spline.coeffs)
            aligned_motion_spline.coeffs = transform_quaternion_frames(aligned_motion_spline.coeffs,
                                                                       [0, angle, 0], offset)
            global_transformation = euler_angles_to_rotation_matrix([0, angle, 0])
            global_transformation[:3, 3] = offset
        elif self.action_constraints.start_pose is not None:
            aligned_motion_spline.coeffs = transform_quaternion_frames(aligned_motion_spline.coeffs,  self.action_constraints.start_pose["orientation"],
                                                                   self.action_constraints.start_pose["position"])
            global_transformation = euler_angles_to_rotation_matrix(self.action_constraints.start_pose["orientation"])
            global_transformation[:3, 3] = self.action_constraints.start_pose["position"]
        else:
            global_transformation = np.eye(4,4)
        return aligned_motion_spline, global_transformation

    def _call_ca_rest_interface(self, ca_input):
        """ call ca rest interface using a json payload
        """
        if self.ca_service_url is not None:
            write_log("Call CA interface",self.ca_service_url,"for",ca_input["elementary_action_name"],ca_input["motion_primitive_name"])
            request = urllib2.Request("http://"+self.ca_service_url, json.dumps(ca_input))
            try:
                handler = urllib2.urlopen(request)
                ca_output_string = handler.read()
                ca_result = json.loads(ca_output_string)
                return ca_result
            except urllib2.HTTPError, e:
               write_log(e.code)
            except urllib2.URLError, e:
               write_log(e.args)
        return None

    def _create_ca_constraints(self, new_node, ca_output, graph_walk):
        ca_constraints = []
        n_canonical_frames = int(self.motion_state_graph.nodes[new_node].get_n_canonical_frames())
        for joint_name in ca_output.keys():
            for ca_constraint_desc in ca_output[joint_name]:
                if "position" in ca_constraint_desc.keys() and len(ca_constraint_desc["position"]) == 3:
                    if self.activate_coordinate_transform_for_ca:
                        position = np.array([ca_constraint_desc["position"][0],ca_constraint_desc["position"][2],-ca_constraint_desc["position"][1]])
                    else:
                        position = np.array(ca_constraint_desc["position"])
                    ca_constraint = GlobalTransformCAConstraint(self.motion_state_graph.skeleton,
                                                                {"joint": joint_name, "canonical_keyframe": -1,
                                                                 "n_canonical_frames": n_canonical_frames,
                                                                 "position": position,
                                                                 "semanticAnnotation":  {"generated": True, "keyframeLabel": None},
                                                                 "ca_constraint": True},
                                                                1.0, 1.0, len(graph_walk.steps))
                    print "CREATE CA constraint", joint_name, ca_constraint.position
                    ca_constraints.append(ca_constraint)
        return ca_constraints
