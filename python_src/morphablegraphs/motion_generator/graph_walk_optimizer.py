__author__ = 'erhe01'

from copy import deepcopy
from optimization.optimizer_builder import OptimizerBuilder
from constraints.time_constraints_builder import TimeConstraintsBuilder
from constraints.spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE, SPATIAL_CONSTRAINT_TYPE_TRAJECTORY, SPATIAL_CONSTRAINT_TYPE_TRAJECTORY_SET, SPATIAL_CONSTRAINT_TYPE_KEYFRAME_DIR_2D, SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION
from constraints.motion_primitive_constraints import MotionPrimitiveConstraints
from ..utilities.log import write_log

GRAPH_WALK_OPTIMIZATION_TWO_HANDS = "none"
GRAPH_WALK_OPTIMIZATION_ALL = "all"
GRAPH_WALK_OPTIMIZATION_TWO_HANDS = "two_hands"
GRAPH_WALK_OPTIMIZATION_END_POINT = "trajectory_end"
CONSTRAINT_FILTER_LIST = [SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE, SPATIAL_CONSTRAINT_TYPE_TRAJECTORY, SPATIAL_CONSTRAINT_TYPE_TRAJECTORY_SET]


class GraphWalkOptimizer(object):
    def __init__(self, motion_primitive_graph, algorithm_config):
        self.motion_primitive_graph = motion_primitive_graph
        self.time_error_minimizer = OptimizerBuilder(algorithm_config).build_time_error_minimizer()
        self.global_error_minimizer = OptimizerBuilder(algorithm_config).build_global_error_minimizer_residual()
        self.collision_avoidance_error_minimizer = OptimizerBuilder(algorithm_config).build_spatial_error_minimizer()
        self.set_algorithm_config(algorithm_config)

    def set_algorithm_config(self, algorithm_config):
        self._algorithm_config = algorithm_config
        self.spatial_mode = algorithm_config["global_spatial_optimization_mode"]
        self.optimize_collision_avoidance_constraints_extra = algorithm_config["optimize_collision_avoidance_constraints_extra"]
        self._global_spatial_optimization_steps = algorithm_config["global_spatial_optimization_settings"]["max_steps"]
        self._position_weight_factor = algorithm_config["global_spatial_optimization_settings"]["position_weight"]
        self._orientation_weight_factor = algorithm_config["global_spatial_optimization_settings"]["orientation_weight"]

    def _is_optimization_required(self, action_constraints):
        return self.spatial_mode == GRAPH_WALK_OPTIMIZATION_ALL and action_constraints.contains_user_constraints or \
               self.spatial_mode == GRAPH_WALK_OPTIMIZATION_TWO_HANDS and action_constraints.contains_two_hands_constraints

    def optimize(self, graph_walk, action_generator, action_constraints):
         #print "has user constraints", action_constraints.contains_user_constraints
        if self._is_optimization_required(action_constraints):
            start_step = max(action_generator.action_state.start_step - self._global_spatial_optimization_steps, 0)
            write_log("start spatial graph walk optimization at", start_step, "looking back", self._global_spatial_optimization_steps, "steps")
            graph_walk = self.optimize_spatial_parameters_over_graph_walk(graph_walk, start_step)

        elif self.spatial_mode == GRAPH_WALK_OPTIMIZATION_END_POINT and action_constraints.root_trajectory is not None:
            start_step = max(len(graph_walk.steps) - self._global_spatial_optimization_steps, 0)
            write_log("start spatial graph walk optimization at", start_step, "looking back", self._global_spatial_optimization_steps, "steps")
            graph_walk = self.optimize_spatial_parameters_over_graph_walk(graph_walk, start_step)

        if self.optimize_collision_avoidance_constraints_extra and action_constraints.collision_avoidance_constraints is not None and len(action_constraints.collision_avoidance_constraints) > 0 :
            write_log("optimize collision avoidance parameters")
            graph_walk = self.optimize_for_collision_avoidance_constraints(graph_walk, action_constraints, action_generator.action_state.start_step)
        return graph_walk

    #def _optimize_over_graph_walk(self, graph_walk, start_step=-1):
    #    start_step = max(start_step, 0)
    #    if self._algorithm_config["use_global_spatial_optimization"]:
    #        self.optimize_spatial_parameters_over_graph_walk(graph_walk, start_step)
    #    if self._algorithm_config["use_global_time_optimization"]:
    #        self.optimize_time_parameters_over_graph_walk(graph_walk)

    def optimize_spatial_parameters_over_graph_walk(self, graph_walk, start_step=0):
        initial_guess = graph_walk.get_global_spatial_parameter_vector(start_step)
        constraint_count = self._filter_constraints(graph_walk, start_step)
        self._adapt_constraint_weights(graph_walk, start_step)
        if constraint_count > 0:
            if start_step == 0:
                prev_frames = None
            else:
                prev_frames = graph_walk.get_quat_frames()[:graph_walk.steps[start_step].start_frame]
            #print "start global optimization", len(initial_guess), constraint_count
            self.global_error_minimizer.set_objective_function_parameters((self.motion_primitive_graph, graph_walk.steps[start_step:],
                                    self._algorithm_config["global_spatial_optimization_settings"]["error_scale_factor"],
                                    self._algorithm_config["global_spatial_optimization_settings"]["quality_scale_factor"],
                                    prev_frames))
            optimal_parameters = self.global_error_minimizer.run(initial_guess)
            graph_walk.update_spatial_parameters(optimal_parameters, start_step)
            #keyframe_error = graph_walk.get_average_keyframe_constraint_error()
            graph_walk.update_temp_motion_vector(start_step, use_time_parameters=False)
            #print keyframe_error
        else:
            print "no user defined constraints"
        return graph_walk

    def _filter_constraints(self, graph_walk, start_step):
        constraint_count = 0
        for step in graph_walk.steps[start_step:]: #TODO add pose constraint for pick and place
            reduced_constraints = []
            for constraint in step.motion_primitive_constraints.constraints:
                if constraint.constraint_type not in CONSTRAINT_FILTER_LIST:
                     reduced_constraints.append(constraint)
            step.motion_primitive_constraints.constraints = reduced_constraints
            #initial_guess += step.parameters[:step.n_spatial_components].tolist()
            constraint_count += len(step.motion_primitive_constraints.constraints)
        return constraint_count

    def _adapt_constraint_weights(self, graph_walk, start_step):
        if self.spatial_mode == GRAPH_WALK_OPTIMIZATION_ALL or self.spatial_mode == GRAPH_WALK_OPTIMIZATION_TWO_HANDS:
            for step in graph_walk.steps[start_step:]:
                for constraint in step.motion_primitive_constraints.constraints:
                    if not "generated" in constraint.semantic_annotation.keys():
                        constraint.weight_factor = self._position_weight_factor
        else: # self.spatial_mode == GRAPH_WALK_OPTIMIZATION_END_POINT
             for constraint in graph_walk.steps[-1].motion_primitive_constraints.constraints:
                 if constraint.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION:
                     constraint.weight_factor = self._position_weight_factor
                 elif constraint.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_DIR_2D:
                     constraint.weight_factor = self._orientation_weight_factor

    def optimize_time_parameters_over_graph_walk(self, graph_walk, start_step=0):

        time_constraints = TimeConstraintsBuilder(graph_walk, start_step).build()
        if time_constraints is not None:
            data = (self.motion_primitive_graph, graph_walk, time_constraints,
                    self._algorithm_config["global_time_optimization_settings"]["error_scale_factor"],
                    self._algorithm_config["global_time_optimization_settings"]["quality_scale_factor"])
            self.time_error_minimizer.set_objective_function_parameters(data)
            initial_guess = graph_walk.get_global_time_parameter_vector(start_step)
            print "initial_guess", initial_guess, time_constraints.constraint_list
            optimal_parameters = self.time_error_minimizer.run(initial_guess)
            graph_walk.update_time_parameters(optimal_parameters, start_step)
            graph_walk.update_temp_motion_vector(start_step, 0)

        return graph_walk

    def optimize_for_collision_avoidance_constraints(self, graph_walk, action_constraints, start_step=0):
        #return graph_walk
        #original_frames = deepcopy(graph_walk.get_quat_frames())
        reduced_motion_vector = deepcopy(graph_walk.motion_vector)
        reduced_motion_vector.reduce_frames(graph_walk.steps[start_step].start_frame)
        print "start frame", graph_walk.steps[start_step].start_frame
        step_index = start_step
        n_steps = len(graph_walk.steps)
        print reduced_motion_vector.n_frames, graph_walk.get_num_of_frames(), reduced_motion_vector.n_frames - graph_walk.get_num_of_frames()
        while step_index < n_steps:
            node = self.motion_primitive_graph.nodes[graph_walk.steps[step_index].node_key]
            print graph_walk.steps[step_index].node_key, node.n_canonical_frames, graph_walk.steps[step_index].start_frame
            motion_primitive_constraints = MotionPrimitiveConstraints()
            active_constraint = False
            for trajectory in action_constraints.collision_avoidance_constraints:
                if reduced_motion_vector.frames is not None:
                    trajectory.set_min_arc_length_from_previous_frames(reduced_motion_vector.frames)
                else:
                    trajectory.min_arc_length = 0.0
                ##if trajectory.range_start < trajectory.min_arc_length+50 and trajectory.min_arc_length < trajectory.range_end:
                trajectory.set_number_of_canonical_frames(node.n_canonical_frames)
                #discrete_trajectory = trajectory.create_discrete_trajectory(original_frames[step.start_frame:step.end_frame])
                motion_primitive_constraints.constraints.append(trajectory)
                active_constraint = True
            if active_constraint:
                data = (node, motion_primitive_constraints, reduced_motion_vector.frames,
                        self._algorithm_config["local_optimization_settings"]["error_scale_factor"],
                        self._algorithm_config["local_optimization_settings"]["quality_scale_factor"])
                self.collision_avoidance_error_minimizer.set_objective_function_parameters(data)
                graph_walk.steps[step_index].parameters = self.collision_avoidance_error_minimizer.run(graph_walk.steps[step_index].parameters)
            motion_primitive_sample = node.back_project(graph_walk.steps[step_index].parameters, use_time_parameters=False)
            reduced_motion_vector.append_quat_frames(motion_primitive_sample.get_motion_vector())
            step_index += 1
        #print reduced_motion_vector.n_frames, graph_walk.get_num_of_frames()
        print step_index, len(graph_walk.steps)
        assert (len(graph_walk.motion_vector.frames)) == len(reduced_motion_vector.frames), (str(len(graph_walk.motion_vector.frames))) + "," + str(len(reduced_motion_vector.frames))
        graph_walk.motion_vector = reduced_motion_vector

        return graph_walk
