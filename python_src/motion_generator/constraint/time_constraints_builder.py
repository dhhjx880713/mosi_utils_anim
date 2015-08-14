__author__ = 'erhe01'

from time_constraints import TimeConstraints


class TimeConstraintsBuilder(object):
    def __init__(self, action_constraints, motion, start_step):
        self.action_constraints = action_constraints
        self.motion = motion
        self.start_step = start_step
        if start_step > 0:
            self.start_key_frame = motion.frame_annotation['elementaryActionSequence'][start_step-1]["endFrame"]
        else:
            self.start_key_frame = 0

        return

    def _extract_time_constraints_from_keyframe_constraints(self):
        self.constraint_list = []
        #TODO

    def build(self):
        return TimeConstraints(self.start_step, self.start_keyframe, self.constraint_list)
