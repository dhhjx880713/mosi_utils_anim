__author__ = 'erhe01'

from time_constraints import TimeConstraints
from spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION


class TimeConstraintsBuilder(object):
    def __init__(self, action_constraints, motion, start_step):
        self.action_constraints = action_constraints
        self.motion = motion
        self.start_step = start_step
        if start_step > 0:
            self.start_keyframe = motion.graph_walk[start_step-1].end_frame
        else:
            self.start_keyframe = 0
        index_range = range(self.start_step, len(motion.graph_walk))
        self.time_constraint_list = []
        self.n_time_constraints = 0
        self._extract_time_constraints_from_graph_walk(motion.graph_walk, index_range)

    def _extract_time_constraints_from_graph_walk_entry(self, step_index, graph_walk_entry):
        """Extract time constraints on any keyframe constraints used during this graph walk step
        :param step_index: int
        :param graph_walk_entry: GraphWalkEntry
        :return:
        """
        if graph_walk_entry.motion_primitive_constraints is not None:
            for constraint in graph_walk_entry.motion_primitive_constraints.constraints:
                if constraint.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION and constraint.desired_time is not None:
                    time_constraint = step_index, constraint.canonical_keyframe, constraint.desired_time
                    self.time_constraint_list.append(time_constraint)
                    self.n_time_constraints += 1

    def _extract_time_constraints_from_graph_walk(self, graph_walk, index_range):
        self.n_time_constraints = 0
        self.time_constraint_list = []
        for step_index in index_range:
            self._extract_time_constraints_from_graph_walk_entry(step_index, graph_walk[step_index])

    def build(self):
        if self.n_time_constraints > 0:
            print "Found", self.n_time_constraints, "time constraints"
            return TimeConstraints(self.start_step, self.start_keyframe, self.time_constraint_list)
        else:
            print "Did not find time constraints"
            return None

