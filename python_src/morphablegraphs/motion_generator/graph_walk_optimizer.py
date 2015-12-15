__author__ = 'erhe01'

from copy import deepcopy
from optimization.optimizer_builder import OptimizerBuilder
from constraints.time_constraints_builder import TimeConstraintsBuilder
from constraints.spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE, SPATIAL_CONSTRAINT_TYPE_TRAJECTORY
from constraints.motion_primitive_constraints import MotionPrimitiveConstraints

class GraphWalkOptimizer(object):
    def __init__(self, algorithm_config):
        self.time_error_minimizer = OptimizerBuilder(algorithm_config).build_time_error_minimizer()
        self.global_error_minimizer = OptimizerBuilder(algorithm_config).build_global_error_minimizer_residual()
        self.collision_avoidance_error_minimizer = OptimizerBuilder(algorithm_config).build_spatial_error_minimizer()

    def _optimize_over_graph_walk(self, graph_walk, start_step=-1):
        #if start_step < 0:
        #    start_step = len(graph_walk.steps)-20
        start_step = max(start_step, 0)
        if self._algorithm_config["use_global_spatial_optimization"]:
            self._optimize_spatial_parameters_over_graph_walk(graph_walk, start_step)
        if self._algorithm_config["use_global_time_optimization"]:
            self._optimize_time_parameters_over_graph_walk(graph_walk)

    def _optimize_spatial_parameters_over_graph_walk(self, graph_walk, start_step=0):
        initial_guess = graph_walk.get_global_spatial_parameter_vector(start_step)
        constraint_count = 0
        for step in graph_walk.steps[start_step:]: #TODO add pose constraint for pick and place
            reduced_constraints = []
            for constraint in step.motion_primitive_constraints.constraints:
                if constraint.constraint_type != SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE and constraint.constraint_type != SPATIAL_CONSTRAINT_TYPE_TRAJECTORY:
                     reduced_constraints.append(constraint)
            step.motion_primitive_constraints.constraints = reduced_constraints
            #initial_guess += step.parameters[:step.n_spatial_components].tolist()
            constraint_count += len(step.motion_primitive_constraints.constraints)
        for step in graph_walk.steps[start_step:]:
            for constraint in step.motion_primitive_constraints.constraints:
                if not "generated" in constraint.semantic_annotation.keys():
                    constraint.weight_factor = 1000.0
                    print "change weight factor"
        if constraint_count > 0:
            if start_step == 0:
                prev_frames = None
            else:
                prev_frames = graph_walk.get_quat_frames()[:graph_walk.steps[start_step].start_frame]
            print "start global optimization", len(initial_guess), constraint_count
            self.global_error_minimizer.set_objective_function_parameters((self.motion_primitive_graph, graph_walk.steps[start_step:],
                                    self._algorithm_config["global_spatial_optimization_settings"]["error_scale_factor"],
                                    self._algorithm_config["global_spatial_optimization_settings"]["quality_scale_factor"],
                                    prev_frames))
            optimal_parameters = self.global_error_minimizer.run(initial_guess)
            graph_walk.update_spatial_parameters(optimal_parameters, start_step)
            #keyframe_error = graph_walk.get_average_keyframe_constraint_error()
            graph_walk.convert_to_motion(0, complete_motion_vector=False, create_frame_annotation=False)
            #graph_walk.export_motion("test", "test.bvh", True)
            #print keyframe_error
        else:
            print "no user defined constraints"
        return graph_walk

    def _optimize_time_parameters_over_graph_walk(self, graph_walk, start_step=0):

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
            graph_walk.convert_to_motion(start_step, 0, complete_motion_vector=False)

        return graph_walk

    def _optimize_for_collision_avoidance_constraints(self, graph_walk, action_constraints, start_step=0):
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
                if reduced_motion_vector.quat_frames is not None:
                    trajectory.set_min_arc_length_from_previous_frames(reduced_motion_vector.quat_frames)
                else:
                    trajectory.min_arc_length = 0.0
                ##if trajectory.range_start < trajectory.min_arc_length+50 and trajectory.min_arc_length < trajectory.range_end:
                trajectory.set_number_of_canonical_frames(node.n_canonical_frames)
                #discrete_trajectory = trajectory.create_discrete_trajectory(original_frames[step.start_frame:step.end_frame])
                motion_primitive_constraints.constraints.append(trajectory)
                active_constraint = True
            if active_constraint:
                data = (node, motion_primitive_constraints, reduced_motion_vector.quat_frames,
                        self._algorithm_config["local_optimization_settings"]["error_scale_factor"],
                        self._algorithm_config["local_optimization_settings"]["quality_scale_factor"])
                self.collision_avoidance_error_minimizer.set_objective_function_parameters(data)
                graph_walk.steps[step_index].parameters = self.collision_avoidance_error_minimizer.run(graph_walk.steps[step_index].parameters)
            motion_primitive_sample = node.back_project(graph_walk.steps[step_index].parameters, use_time_parameters=False)
            reduced_motion_vector.append_quat_frames(motion_primitive_sample.get_motion_vector())
            step_index += 1
        #print reduced_motion_vector.n_frames, graph_walk.get_num_of_frames()
        print step_index, len(graph_walk.steps)
        assert (len(graph_walk.motion_vector.quat_frames)) == len(reduced_motion_vector.quat_frames), (str(len(graph_walk.motion_vector.quat_frames))) + "," + str(len(reduced_motion_vector.quat_frames))
        graph_walk.motion_vector = reduced_motion_vector

        return graph_walk
