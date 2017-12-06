__author__ = 'erhe01'
import os
import random
from ..utilities.io_helper_functions import write_to_json_file
from . import META_INFORMATION_FILE_NAME

LAST_FRAME = "lastFrame"  # TODO set standard for keyframe values
NEGATIVE_ONE = "-1"

KEYFRAME_LABEL_END = "end"
KEYFRAME_LABEL_START = "start"
KEYFRAME_LABEL_MIDDLE = "middle"


class ElementaryActionMetaInfo(object):
    def __init__(self, elementary_action_name, elementary_action_directory):
        self.elementary_action_name = elementary_action_name
        self.elementary_action_directory = elementary_action_directory
        self.label_to_motion_primitive_map = dict()
        self.start_states = list()
        self.cycle_states = list()
        self.n_start_states = 0
        self.end_states = list()
        self.n_end_states = 0
        self.mp_annotations = dict()
        self.meta_information = None
        self.motion_primitive_annotation_regions = dict()

    def set_meta_information(self, meta_information=None):
        """
        Identify start and end states from meta information.
        """
        if meta_information is None:
            return
        self.meta_information = meta_information
        for key in ["annotations", "start_states", "end_states"]:
            assert key in list(self.meta_information.keys())
        self.start_states = self.meta_information["start_states"]
        if "cycle_states" in list(self.meta_information.keys()):
            self.cycle_states = self.meta_information["cycle_states"]
        self.n_start_states = len(self.start_states)
        self.end_states = self.meta_information["end_states"]
        self.n_end_states = len(self.end_states)
        self.mp_annotations = self.meta_information["annotations"]
        self._create_annotation_label_to_motion_primitive_map()
        if "annotation_regions" in list(self.meta_information.keys()):
            self.motion_primitive_annotation_regions = self.meta_information["annotation_regions"]

    def _create_annotation_label_to_motion_primitive_map(self):
        """Create a map from semantic label to motion primitive
        """
        for motion_primitive in list(self.mp_annotations.keys()):
            if motion_primitive != "all_primitives":
                annotations = self.mp_annotations[motion_primitive]
                for label in list(annotations.keys()):
                    if label not in list(self.label_to_motion_primitive_map.keys()):
                        self.label_to_motion_primitive_map[label] = []
                    self.label_to_motion_primitive_map[label] += [motion_primitive]

    def get_random_start_state(self):
        """ Returns the name of a random start state. """
        if self.n_start_states > 0:
            random_index = random.randrange(0, self.n_start_states, 1)
            return self.elementary_action_name, self.start_states[random_index]

    def get_start_states(self):
        """
        Return all start states
        :return:
        """
        return self.start_states

    def get_random_end_state(self):
        """ Returns the name of a random start state."""
        if self.n_end_states > 0:
            random_index = random.randrange(0, self.n_end_states, 1)
            return self.elementary_action_name, self.end_states[random_index]

    def _convert_tuples_to_strings(self, in_dict):
        copy_dict = {}
        for key in list(in_dict.keys()):
            if isinstance(key, tuple):
                try:
                    copy_dict[key[1]] = in_dict[key]
                except Exception as exception:
                    print(exception.message)
                    continue
            else:
                copy_dict[key] = in_dict[key]
        return copy_dict

    def save_updated_meta_info(self):
        """ Save updated meta data to a json file
        """
        if self.meta_information is not None and self.elementary_action_directory is not None:
            path = self.elementary_action_directory + os.sep + META_INFORMATION_FILE_NAME
            write_to_json_file(path, self._convert_tuples_to_strings(self.meta_information))
        return

    def get_canonical_keyframe_labels(self, motion_primitive_name):
        if motion_primitive_name in list(self.mp_annotations.keys()):
            keyframe_labels = self.mp_annotations[motion_primitive_name]
        else:
            keyframe_labels = {}
        return keyframe_labels

    def get_keyframe_from_annotation(self, mp_name, label, n_canonical_frames):
        keyframe = None
        if label == KEYFRAME_LABEL_END:
            keyframe = n_canonical_frames-1
        elif label == KEYFRAME_LABEL_START:#"start"
            keyframe = 0
        elif label == KEYFRAME_LABEL_MIDDLE:#"middle"
            keyframe = n_canonical_frames/2
        else:
            print("search for label ", label, list(self.mp_annotations[mp_name].keys()))
            if mp_name in list(self.mp_annotations.keys()) and \
                            label in list(self.mp_annotations[mp_name].keys()):
                    keyframe = self.mp_annotations[mp_name][label]
                    if keyframe in [NEGATIVE_ONE, LAST_FRAME]:
                        keyframe = n_canonical_frames-1
                    elif keyframe == KEYFRAME_LABEL_MIDDLE:
                        keyframe = n_canonical_frames/2
            else:
                print("Error: Could not map keyframe label", label, list(self.mp_annotations.keys()))
        if keyframe is not None:
            keyframe = int(keyframe)
        return keyframe
