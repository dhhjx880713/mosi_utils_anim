import numpy as np
from copy import copy
from constraints.motion_primitive_constraints import MotionPrimitiveConstraints
from constraints.spatial_constraints.keyframe_constraints.global_transform_constraint import GlobalTransformConstraint
from constraints.spatial_constraints.keyframe_constraints.direction_2d_constraint import Direction2DConstraint
from ..animation_data.motion_editing import create_transformation_matrix
from ..utilities import write_log


class TrajectoryFollowingPlanner(object):
    def __init__(self, motion_state_graph,  algorithm_config):
        self.motion_state_graph = motion_state_graph
        self.step_look_ahead_distance = algorithm_config["trajectory_following_settings"]["look_ahead_distance"]
        self.use_local_coordinates = algorithm_config["use_local_coordinates"]
        self.mp_generator = None
        self.action_state = None
        self.graph_walk = None
        self.travelled_arc_length = 0.0
        self.trajectory = None

    def set_state(self, mp_generator, action_state, action_constraints, graph_walk):
        self.mp_generator = mp_generator
        self.action_state = action_state
        self.travelled_arc_length = action_state.travelled_arc_length
        self.trajectory = action_constraints.root_trajectory
        self.graph_walk = graph_walk
        self.motion_vector = copy(self.graph_walk.motion_vector)

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

    def _generate_node_evaluation_constraints(self, motion_vector, add_orientation=False):
        goal_arc_length = self.travelled_arc_length + self.step_look_ahead_distance
        mp_constraints = MotionPrimitiveConstraints()
        mp_constraints.skeleton = self.motion_state_graph.skeleton
        mp_constraints.aligning_transform = create_transformation_matrix(motion_vector.start_pose["position"], motion_vector.start_pose["orientation"])
        mp_constraints.start_pose = motion_vector.start_pose
        constraint_desc = {"joint": "Hips", "canonical_keyframe": -1, "n_canonical_frames": 0,
                           "semanticAnnotation": {"keyframeLabel": "end", "generated": True}}
        if add_orientation:
            self._add_constraint_with_orientation(constraint_desc, goal_arc_length, mp_constraints)
        else:
            self._add_constraint(constraint_desc, goal_arc_length, mp_constraints)
        return mp_constraints

    def select_next_step(self, options, add_orientation=False):
        if self.graph_walk is None:
            return None
        next_node = self._look_one_step_ahead(options, add_orientation)
        write_log("Next node is", next_node)#, "with an error of", errors[min_idx]
        return next_node

    def _look_one_step_ahead(self, options, add_orientation):
        motion_vector = self.motion_vector

        mp_constraints = self._generate_node_evaluation_constraints(motion_vector, add_orientation)

        if self.use_local_coordinates and False:
            mp_constraints = mp_constraints.transform_constraints_to_local_cos()

        errors, s_vectors = self._evaluate_options(mp_constraints, options, motion_vector.frames)
        min_idx = np.argmin(errors)
        next_node = options[min_idx]
        return next_node

    def _evaluate_options(self, mp_constraints, options, prev_frames):
        errors = np.empty(len(options))
        s_vectors = []
        index = 0
        for node_name in options:
            motion_primitive_node = self.motion_state_graph.nodes[node_name]
            canonical_keyframe = motion_primitive_node.get_n_canonical_frames() - 1
            for c in mp_constraints.constraints:
                c.canonical_keyframe = canonical_keyframe
            s_vector = self.mp_generator._get_best_fit_sample_using_cluster_tree(motion_primitive_node, mp_constraints, prev_frames, 1)
            s_vectors.append(s_vector)
            write_log("Evaluated option", node_name, mp_constraints.min_error)
            errors[index] = mp_constraints.min_error
            index += 1
        return errors, s_vectors