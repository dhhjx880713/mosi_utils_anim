import os
from datetime import datetime
from ..animation_data import MotionVector, ROTATION_TYPE_QUATERNION, Skeleton, BVHReader
from ..utilities import write_to_json_file, write_to_logfile
from ..utilities.io_helper_functions import get_bvh_writer


class AnnotatedMotionVector(MotionVector):
    def __init__(self, algorithm_config=None, rotation_type=ROTATION_TYPE_QUATERNION):
        super(AnnotatedMotionVector, self).__init__(algorithm_config, rotation_type)
        self.keyframe_event_list = None
        self.mg_input = None
        self.skeleton = None
        self.graph_walk = None
        self.ik_constraints = {}

    def export(self, output_dir, output_filename, add_time_stamp=False, export_details=False):
        """ Saves the resulting animation frames, the annotation and actions to files.
        Also exports the input file again to the output directory, where it is
        used as input for the constraints visualization by the animation server.
        """

        MotionVector.export(self, self.skeleton, output_dir, output_filename, add_time_stamp)

        if self.mg_input is not None:
            write_to_json_file(output_dir + os.sep + output_filename + ".json", self.mg_input.mg_input_file)
        if self.keyframe_event_list is not None:
            self.keyframe_event_list.export_to_file(output_dir + os.sep + output_filename)

    def load_from_file(self, file_name, filter_joints=True):
        bvh = BVHReader(file_name)
        self.skeleton = Skeleton(bvh)
        self.from_bvh_reader(bvh, filter_joints=filter_joints)

    def generate_bvh_string(self):
        bvh_writer = get_bvh_writer(self.skeleton, self.frames)
        return bvh_writer.generate_bvh_string()
