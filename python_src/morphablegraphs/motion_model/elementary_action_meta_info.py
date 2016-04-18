__author__ = 'erhe01'
import os
import random
from ..utilities.io_helper_functions import write_to_json_file
from . import META_INFORMATION_FILE_NAME


class ElementaryActionMetaInfo(object):
    def __init__(self, elementary_action_name, elementary_action_directory):
        self.elementary_action_name = elementary_action_name
        self.elementary_action_directory = elementary_action_directory
        self.label_to_motion_primitive_map = dict()
        self.start_states = list()
        self.n_start_states = 0
        self.end_states = list()
        self.n_end_states = 0
        self.motion_primitive_annotations = dict()
        self.meta_information = None
        self.semantic_annotation_map = dict()

    def set_meta_information(self, meta_information=None):
        """
        Identify start and end states from meta information.
        """
        if meta_information is not None:
            self.meta_information = meta_information
            for key in ["annotations", "start_states", "end_states"]:
                assert key in self.meta_information.keys()
            self.start_states = self.meta_information["start_states"]
            self.n_start_states = len(self.start_states)
            self.end_states = self.meta_information["end_states"]
            self.n_end_states = len(self.end_states)
            self.motion_primitive_annotations = self.meta_information["annotations"]
            if "semantic_annotation_map" in self.meta_information.keys():
                self.semantic_annotation_map = self.meta_information["semantic_annotation_map"]
            self._create_annotation_label_to_motion_primitive_map()

    def _create_annotation_label_to_motion_primitive_map(self):
        """Create a map from semantic label to motion primitive
        """
        for motion_primitive in self.motion_primitive_annotations.keys():
            if motion_primitive != "all_primitives":
                annotations = self.motion_primitive_annotations[motion_primitive]
                for label in annotations.keys():
                    self.label_to_motion_primitive_map[label] = motion_primitive

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

    def _convert_keys_to_strings(self, mydict):
        copy_dict = {}
        for key in mydict.keys():
            if isinstance(key, tuple):
                try:
                    copy_dict[key[1]] = mydict[key]
                except Exception as exception:
                    print exception.message
                    continue
            else:
                copy_dict[key] = mydict[key]
        return copy_dict

    def save_updated_meta_info(self):
        """ Save updated meta data to a json file
        """
        if self.meta_information is not None and self.elementary_action_directory is not None:
            path = self.elementary_action_directory + os.sep + META_INFORMATION_FILE_NAME
            write_to_json_file(path, self._convert_keys_to_strings(self.meta_information))
        return

    def get_canonical_keyframe_labels(self, motion_primitive_name):
        if motion_primitive_name in self.motion_primitive_annotations.keys():
            keyframe_labels = self.motion_primitive_annotations[motion_primitive_name]
        else:
            keyframe_labels = {}
        return keyframe_labels
