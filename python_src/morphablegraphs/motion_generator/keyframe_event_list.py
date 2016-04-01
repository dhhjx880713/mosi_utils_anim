import numpy as np
from copy import copy
from ..utilities import write_to_json_file

UNCONSTRAINED_EVENTS_TRANSFER_POINT = "transfer_point"


class KeyframeEventList(object):
    def __init__(self, ):
        self.frame_annotation = dict()
        self.frame_annotation['elementaryActionSequence'] = []
        self.keyframe_events_dict = dict()

    def update_events(self, graph_walk, start_step):
        self._create_event_dict(graph_walk)
        self._create_frame_annotation(graph_walk, start_step)
        self._add_event_list_to_frame_annotation(graph_walk)

    def update_frame_annotation(self, action_name, start_frame, end_frame):
        """Adds a dictionary to self.frame_annotation marking start and end
            frame of an action.
        """
        action_frame_annotation = dict()
        action_frame_annotation["startFrame"] = start_frame
        action_frame_annotation["elementaryAction"] = action_name
        action_frame_annotation["endFrame"] = end_frame
        self.frame_annotation['elementaryActionSequence'].append(action_frame_annotation)

    def _create_event_dict(self, graph_walk):
        self._create_events_from_keyframe_constraints(graph_walk)
        self._add_unconstrained_events_from_annotation(graph_walk)

    def _create_frame_annotation(self, graph_walk, start_step=0):
        self.frame_annotation['elementaryActionSequence'] = self.frame_annotation['elementaryActionSequence'][:start_step]
        if start_step == 0:
            start_frame = 0
        else:
            step = graph_walk.steps[start_step-1]
            start_frame = step.end_frame
        end_frame = start_frame
        prev_step = None
        for step in graph_walk.steps[start_step:]:
            action_name = step.node_key[0]
            #time_function = graph_walk.motion_state_graph.nodes[step.node_key].back_project_time_function(step.parameters)
            if prev_step is not None and action_name != prev_step.node_key[0]:
                #add entry for previous elementary action
                print "add", prev_step.node_key[0]
                self.update_frame_annotation(prev_step.node_key[0], start_frame, end_frame-1)
                start_frame = end_frame
            end_frame += step.end_frame - step.start_frame + 1
            prev_step = step
        if prev_step is not None:
            print "add", prev_step.node_key[0]
            self.update_frame_annotation(prev_step.node_key[0], start_frame, end_frame-1)

    #def _warp_keyframe_index(self, time_function, key_frame_index):
    #    """ Inverse lookup of warped keyframe
    #    :param time_function:
    #    :param key_frame_index:
    #    :return:
    #    """
    #    closest_keyframe = min(time_function, key=lambda x: abs(x - key_frame_index))
    #    warped_keyframe = np.where(time_function == closest_keyframe)[0][0]
    #    return warped_keyframe

    def _extract_keyframe_index(self, keyframe_event, time_function, frame_offset):
        canonical_keyframe = int(keyframe_event["canonical_keyframe"])
        if time_function is not None:
            #warped_keyframe = self._warp_keyframe_index(time_function, canonical_keyframe)
            event_keyframe_index = frame_offset + int(time_function[canonical_keyframe]) + 1  # add +1 to map the frame correctly TODO: test and verify for all cases
        else:
            event_keyframe_index = frame_offset + canonical_keyframe
        return event_keyframe_index

    def _extract_event_list(self, keyframe_event, event_keyframe_index):
        #extract events from event list
        n_events = len(keyframe_event["event_list"])
        if n_events == 1:
            events = keyframe_event["event_list"]
        else:
            events = self._merge_multiple_keyframe_events(keyframe_event["event_list"], n_events)

        ##merge events with events of previous iterations
        if event_keyframe_index in self.keyframe_events_dict:
            events = events + self.keyframe_events_dict[event_keyframe_index]
        events = self._merge_multiple_keyframe_events(events, len(events))
        return events

    def _create_events_from_keyframe_constraints(self, graph_walk):
        """ Traverse elementary actions and motion primitives
        :return:
        """
        self.keyframe_events_dict = dict()
        frame_offset = 0
        for step in graph_walk.steps:
            time_function = None
            if graph_walk.use_time_parameters:
                time_function = graph_walk.motion_state_graph.nodes[step.node_key].back_project_time_function(step.parameters)
            for keyframe_event in step.motion_primitive_constraints.keyframe_event_list.values():
                event_keyframe_index = self._extract_keyframe_index(keyframe_event, time_function, frame_offset)
                self.keyframe_events_dict[event_keyframe_index] = self._extract_event_list(keyframe_event, event_keyframe_index)
            frame_offset += step.end_frame - step.start_frame + 1

    def _add_event_list_to_frame_annotation(self, graph_walk):
        """
        self.keyframe_events_dict[keyframe] m
        :return:
        """
        #print "keyframe event dict", self.keyframe_events_dict
        keyframe_event_list = []
        for keyframe in self.keyframe_events_dict.keys():
            #rint "keyframe event dict", self.keyframe_events_dict[keyframe]
            for event_desc in self.keyframe_events_dict[keyframe]:
                print "event description", event_desc
                event = dict()
                if graph_walk.mg_input is not None and graph_walk.mg_input.activate_joint_mapping:
                    if isinstance(event_desc["parameters"]["joint"], basestring):
                        event["jointName"] = graph_walk.mg_input.inverse_map_joint(event_desc["parameters"]["joint"])
                    else:
                        print "apply joint mapping"
                        event["jointName"] = map(graph_walk.mg_input.inverse_map_joint, event_desc["parameters"]["joint"])
                else:
                    event["jointName"] = event_desc["parameters"]["joint"]
                event["jointName"] = self._handle_both_hands_event(event, graph_walk.mg_input.activate_joint_mapping)
                event_type = event_desc["event"]
                target = event_desc["parameters"]["target"]
                event[event_type] = target
                event["frameNumber"] = int(keyframe)
                keyframe_event_list.append(event)
        self.frame_annotation["events"] = keyframe_event_list

    def _merge_multiple_keyframe_events(self, events, num_events):
        """Merge events if there are more than one event defined for the same keyframe.
        """
        event_list = [(events[i]["event"], events[i]) for i in xrange(num_events)]
        temp_event_dict = dict()
        for name, event in event_list:
            if name not in temp_event_dict.keys():
               temp_event_dict[name] = event
            else:
                if "joint" in temp_event_dict[name]["parameters"].keys():
                    existing_entry = copy(temp_event_dict[name]["parameters"]["joint"])
                    if isinstance(existing_entry, basestring) and event["parameters"]["joint"] != existing_entry:
                        temp_event_dict[name]["parameters"]["joint"] = [existing_entry, event["parameters"]["joint"]]
                    elif event["parameters"]["joint"] not in existing_entry:
                        temp_event_dict[name]["parameters"]["joint"].append(event["parameters"]["joint"])
                    print "event dict merged", temp_event_dict[name]
                else:
                    print "event dict merge did not happen", temp_event_dict[name]
        return temp_event_dict.values()

    def export_to_file(self, prefix):
        write_to_json_file(prefix + "_annotations"+".json", self.frame_annotation)
        #print "keyframe event dict", self.keyframe_events_dict, filename
        write_to_json_file(prefix + "_actions"+".json", self.keyframe_events_dict)

    def _add_unconstrained_events_from_annotation(self, graph_walk):
        """The method assumes the start and end frames of each step were already warped by calling convert_to_motion
        """
        if graph_walk.mg_input is not None:
            for action_index, action_entry in enumerate(graph_walk.elementary_action_list):
                keyframe_annotations = graph_walk.mg_input.keyframe_annotations[action_index]
                for key in keyframe_annotations.keys():
                    if key == UNCONSTRAINED_EVENTS_TRANSFER_POINT:
                        self._add_transition_event(graph_walk, keyframe_annotations, action_entry)

    def _add_transition_event(self, graph_walk, keyframe_annotations, action_entry):
        """ Look for the frame with the closest distance and add a transition event for it
        """
        if len(keyframe_annotations[UNCONSTRAINED_EVENTS_TRANSFER_POINT]["annotations"]) == 2:

            #print "create transfer event"
            joint_name_a = keyframe_annotations[UNCONSTRAINED_EVENTS_TRANSFER_POINT]["annotations"][0]["parameters"]["joint"]
            joint_name_b = keyframe_annotations[UNCONSTRAINED_EVENTS_TRANSFER_POINT]["annotations"][1]["parameters"]["joint"]
            attach_joint = joint_name_a
            for event_parameters in keyframe_annotations[UNCONSTRAINED_EVENTS_TRANSFER_POINT]["annotations"]:
                if event_parameters["event"] == "attach":
                    attach_joint = event_parameters["parameters"]["joint"]

            if isinstance(joint_name_a, basestring):
                keyframe_range_start = graph_walk.steps[action_entry.start_step].start_frame
                keyframe_range_end = min(graph_walk.steps[action_entry.end_step].end_frame+1, graph_walk.motion_vector.n_frames)
                least_distance = np.inf
                closest_keyframe = graph_walk.steps[action_entry.start_step].start_frame
                for frame_index in xrange(keyframe_range_start, keyframe_range_end):
                    position_a = graph_walk.full_skeleton.nodes[joint_name_a].get_global_position(graph_walk.motion_vector.frames[frame_index])
                    position_b = graph_walk.full_skeleton.nodes[joint_name_b].get_global_position(graph_walk.motion_vector.frames[frame_index])
                    distance = np.linalg.norm(position_a - position_b)
                    if distance < least_distance:
                        least_distance = distance
                        closest_keyframe = frame_index
                target_object = keyframe_annotations[UNCONSTRAINED_EVENTS_TRANSFER_POINT]["annotations"][0]["parameters"]["target"]
                self.keyframe_events_dict[closest_keyframe] = [ {"event":"transfer", "parameters": {"joint" : [attach_joint], "target": target_object}}]
                print "added transfer event", closest_keyframe

    def _handle_both_hands_event(self, event, activate_joint_mapping=False):
        if isinstance(event["jointName"], list):
            if activate_joint_mapping:
                if "RightHand" in event["jointName"] and "LeftHand" in event["jointName"]:
                    return "BothHands"
                else:
                    return str(event["jointName"])
            else:
                if "RightToolEndSite" in event["jointName"] and "LeftToolEndSite" in event["jointName"]:
                    return "BothHands"
                else:
                    return str(event["jointName"])
        else:
            return str(event["jointName"])