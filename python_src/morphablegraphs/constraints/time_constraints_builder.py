__author__ = 'erhe01'

from .time_constraints import TimeConstraints
from .spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION
from ..utilities import write_message_to_log, LOG_MODE_DEBUG


class TimeConstraintsBuilder(object):
    def __init__(self, graph_walk, start_step, end_step):
        self.start_step = start_step
        self.end_step = min(end_step+1, len(graph_walk.steps))
        index_range = list(range(self.start_step, self.end_step))
        self.time_constraint_list = []
        self.n_time_constraints = 0
        self._extract_time_constraints_from_graph_walk(graph_walk.steps, index_range)

    def _extract_time_constraints_from_graph_walk_entry(self, constrained_step_count, graph_walk_entry):
        """Extract time constraints on any keyframe constraints used during this graph walk step
        :param step_index: int
        :param graph_walk_entry: GraphWalkEntry
        :return:
        """
        if graph_walk_entry.motion_primitive_constraints is not None:
            for constraint in graph_walk_entry.motion_primitive_constraints.constraints:
                if constraint.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION and constraint.desired_time is not None:
                    time_constraint = constrained_step_count, constraint.canonical_keyframe, constraint.desired_time
                    self.time_constraint_list.append(time_constraint)
                    self.n_time_constraints += 1

    def _extract_time_constraints_from_graph_walk(self, graph_walk, index_range):
        self.n_time_constraints = 0
        self.time_constraint_list = []
        constrained_step_count = 0
        for step_index in index_range:
            self._extract_time_constraints_from_graph_walk_entry(constrained_step_count, graph_walk[step_index])
            constrained_step_count += 1

    def build(self, motion_primitive_graph, graph_walk):
        if self.n_time_constraints > 0:
            write_message_to_log("Found " + str(self.n_time_constraints) + " time constraints", LOG_MODE_DEBUG)
            return TimeConstraints(motion_primitive_graph, graph_walk, self.start_step, self.end_step, self.time_constraint_list)
        else:
            write_message_to_log("Did not find time constraints", LOG_MODE_DEBUG)
            return None

