import os
from datetime import datetime
from ..animation_data import MotionVector, ROTATION_TYPE_QUATERNION
from ..utilities import write_to_json_file, write_to_logfile


class AnnotatedMotionVector(MotionVector):
    def __init__(self, algorithm_config=None, rotation_type=ROTATION_TYPE_QUATERNION):
        super(AnnotatedMotionVector, self).__init__(algorithm_config, rotation_type)
        self.frame_annotation = dict()
        self.frame_annotation['elementaryActionSequence'] = []
        self.step_count = 0
        self.mg_input = None
        self._algorithm_config = algorithm_config
        self.keyframe_events_dict = dict()
        self.skeleton = None

    def export(self, output_dir, output_filename, add_time_stamp=False, export_details=False):
        """ Saves the resulting animation frames, the annotation and actions to files.
        Also exports the input file again to the output directory, where it is
        used as input for the constraints visualization by the animation server.
        """

        MotionVector.export(self, self.skeleton, output_dir, output_filename, add_time_stamp)

        if self.mg_input is not None:
            write_to_json_file(output_dir + os.sep + output_filename + ".json", self.mg_input.mg_input_file)
        self._export_event_dict(output_dir + os.sep + output_filename + "_actions"+".json")
        write_to_json_file(output_dir + os.sep + output_filename + "_annotations"+".json", self.frame_annotation)


    def _export_event_dict(self, filename):
        #print "keyframe event dict", self.keyframe_events_dict, filename
        write_to_json_file(filename, self.keyframe_events_dict)