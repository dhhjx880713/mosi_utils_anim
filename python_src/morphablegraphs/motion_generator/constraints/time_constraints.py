__author__ = 'erhe01'
import numpy as np

class TimeConstraints(object):
    def __init__(self, start_step, start_keyframe, constraint_list):
        self.start_step = start_step
        self.start_keyframe = start_keyframe
        self.constraint_list = constraint_list

    def evaluate_graph_walk(self, s, motion_primitive_graph, motion):
        print "evaluate", s
        time_functions = self._get_time_functions_from_graph_walk(s, motion_primitive_graph, motion)
        #get difference to desired time for each constraint
        frame_time = motion_primitive_graph.skeleton.frame_time
        error_sum = 0
        for time_constraint in self.constraint_list:
            error_sum += self.calculate_constraint_error(time_functions, time_constraint, frame_time)
        return error_sum

    def _get_time_functions_from_graph_walk(self, s, motion_primitive_graph, graph_walk):
        """get time functions for all steps
        :param s:
        :param motion_primitive_graph:
        :param motion:
        :return:
        """
        time_functions = []
        offset = 0
        for step in graph_walk.steps[self.start_step:]:
            gamma = s[offset:offset+step.n_time_components]
            time_function = motion_primitive_graph.nodes[step.node_key]._inverse_temporal_pca(gamma)
            time_functions.append(time_function)
            offset += step.n_time_components
        return time_functions

    def calculate_constraint_error(self, time_functions, time_constraint, frame_time):
        constrained_step_index, constrained_keyframe_index, desired_time = time_constraint
        n_frames = self.start_keyframe #when it starts the first step start_keyframe would be 0
        temp_step_index = 0
        for time_function in time_functions: # look back n_steps
            print "time func", temp_step_index, constrained_step_index
            if temp_step_index < constrained_step_index:# go to the graph walk entry we want to constrain
                #simply add the number of frames
                n_frames += len(time_function)
                temp_step_index += 1
            else:
                #inverse lookup the warped frame that maps to the labeled canonical keyframe with the time constraint
                closest_keyframe = min(time_function, key=lambda x: abs(x-constrained_keyframe_index))
                mapped_keyframe = np.where(time_function==closest_keyframe)[0][0]
                n_frames += mapped_keyframe
                total_seconds = n_frames * frame_time
                error = abs(desired_time-total_seconds)# + negative_log_
                #print time_function
                print "time error", error, total_seconds, desired_time, mapped_keyframe, constrained_keyframe_index, n_frames
                return error
        return 10000

    def get_initial_guess(self, graph_walk):
        parameters = []
        for step in graph_walk.steps[self.start_step:]:
            parameters += step.parameters[step.n_spatial_components:].tolist()
        return parameters

