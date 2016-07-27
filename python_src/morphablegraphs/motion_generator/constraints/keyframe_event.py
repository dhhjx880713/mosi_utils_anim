from copy import copy

class KeyframeEvent(object):
    def __init__(self, label, canonical_keyframe, event_list):
        self.label = label
        self.canonical_keyframe = int(canonical_keyframe)
        self.event_list = event_list

    def to_dict(self):
        return {"canonical_keyframe": self.canonical_keyframe, "event_list": self.event_list}

    def extract_keyframe_index(self, time_function, frame_offset):
        if time_function is not None:
            return frame_offset + int(time_function[self.canonical_keyframe]) + 1  # add +1 to map the frame correctly TODO: test and verify for all cases
        else:
            return frame_offset + self.canonical_keyframe

    def merge_event_list(self, prev_events=None):
        """merge events with events of previous iterations"""
        if prev_events is not None:
            self.event_list = self.event_list + prev_events.event_list
        n_events = len(self.event_list)
        if n_events > 1:
            self.event_list = self._merge_multiple_keyframe_events(self.event_list, n_events)

    def merge_event_list2(self, prev_events=None):
        """merge events with events of previous iterations"""
        events = self.event_list
        if prev_events is not None:
            events = events + prev_events
        n_events = len(events)
        if n_events == 1:
            return events
        else:
            return self._merge_multiple_keyframe_events(events, n_events)

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