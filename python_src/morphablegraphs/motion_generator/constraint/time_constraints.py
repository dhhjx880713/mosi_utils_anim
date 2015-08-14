__author__ = 'erhe01'


class TimeConstraints(object):
    def __init__(self, start_step, start_keyframe):
        self.start_step = start_step
        self.start_keyframe = start_keyframe
        self.constraint_list = []


    def evaluate_graph_walk(self, s, morphable_graph, motion):

        time_functions = self._get_time_functions_from_graph_walk(s, morphable_graph, motion)
        #get difference to desired time for each constraint
        frame_time = morphable_graph.skeleton.frame_time
        error_sum = 0
        for time_constraint in self.constraint_list:
            error_sum += self.calculate_constraint_error(time_functions, time_constraint, frame_time)
        return error_sum

    def _get_time_functions_from_graph_walk(self, s, morphable_graph, motion):

        #get time functions for all steps
        time_functions = []
        offset = 0
        for step in motion.graph_walk[self.start_step:]:
            gamma = s[offset:step.n_time_components]
            time_function = morphable_graph.nodes[step.node_key]._inverse_temporal_pca(gamma)
            time_functions.append(time_function)
            offset += step.n_time_components

    def calculate_constraint_error(self, time_functions, time_constraint, frame_time):
        constrained_step_index, constrained_keyframe_index, desired_time = time_constraint

        n_frames = self.start_keyframe #w hen it starts the first step start_keyframe would be 0
        temp_step_index = 0
        for time_function in time_functions: # look back n_steps
            if temp_step_index < constrained_step_index:
                #simply add the number of frames
                n_frames += len(time_function)
                temp_step_index += 1
            else:
                #inverse lookup the warped frame that maps to the labelled canonical keyframe with the time constraint
                mapped_key_frame = min(time_function, key=lambda x: abs(x-constrained_keyframe_index))
                n_frames += mapped_key_frame
                total_seconds = n_frames * frame_time
                return abs(desired_time-total_seconds)

        return 10000