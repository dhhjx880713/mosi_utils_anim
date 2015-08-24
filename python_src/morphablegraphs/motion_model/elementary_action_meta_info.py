__author__ = 'erhe01'
import os
import random
from ..utilities.io_helper_functions import write_to_json_file
from . import META_INFORMATION_FILE_NAME


class ElementaryActionMetaInfo(object):
    def __init__(self, elementary_action_name, elementary_action_directory):
        self.elementary_action_name = elementary_action_name
        self.elementary_action_directory = elementary_action_directory
        self.annotation_map = dict()
        self.start_states = list()
        self.end_states = list()
        self.motion_primitive_annotations = dict()
        self.meta_information = None

    def set_meta_information(self, meta_information=None):
        """
        Identify start and end states from meta information.
        """
        self.meta_information = meta_information
        for key in ["annotations", "start_states", "end_states"]:
            assert key in self.meta_information.keys()
        self.start_states = self.meta_information["start_states"]
        self.end_states = self.meta_information["end_states"]
        self.motion_primitive_annotations = self.meta_information["annotations"]
        self._create_semantic_annotation()

    def _create_semantic_annotation(self):
        """Create a map from semantic label to motion primitive
        """
        for motion_primitive in self.meta_information["annotations"].keys():
            if motion_primitive != "all_primitives":
                motion_primitve_annotations = self.meta_information["annotations"][motion_primitive]
                for label in motion_primitve_annotations.keys():
                    self.annotation_map[label] = motion_primitive

    def get_random_start_state(self):
        """ Returns the name of a random start state. """
        random_index = random.randrange(0, len(self.start_states), 1)
        start_state = (self.elementary_action_name, self.start_states[random_index])
        return start_state

    def get_random_end_state(self):
        """ Returns the name of a random start state."""
        random_index = random.randrange(0, len(self.end_states), 1)
        start_state = (self.elementary_action_name, self.end_states[random_index])
        return start_state

    def _convert_keys_to_strings(self, mydict):
        copy_dict = {}
        for key in mydict.keys():
            if isinstance(key) is tuple:
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
