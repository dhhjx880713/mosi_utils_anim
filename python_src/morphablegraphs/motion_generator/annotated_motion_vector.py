import os
import numpy as np
from ..animation_data import MotionVector, ROTATION_TYPE_QUATERNION, SkeletonBuilder, BVHReader
from ..utilities import write_to_json_file
from ..utilities.io_helper_functions import get_bvh_writer


class AnnotatedMotionVector(MotionVector):
    def __init__(self, skeleton=None, algorithm_config=None, rotation_type=ROTATION_TYPE_QUATERNION):
        super(AnnotatedMotionVector, self).__init__(skeleton, algorithm_config, rotation_type)
        self.keyframe_event_list = None
        self.mg_input = None
        self.graph_walk = None
        self.grounding_constraints = None
        self.ground_contacts = None
        self.ik_constraints = dict()

    def export(self, output_dir, output_filename, add_time_stamp=False, export_details=False):
        """ Saves the resulting animation frames, the annotation and actions to files.
        Also exports the input file again to the output directory, where it is
        used as input for the constraints visualization by the animation server.
        """

        MotionVector.export(self, self.skeleton, output_dir + os.sep + output_filename, add_time_stamp)
        self.export_annotation(output_dir, output_filename)

    def export_annotation(self,output_dir, output_filename):
        if self.mg_input is not None:
            write_to_json_file(output_dir + os.sep + output_filename + ".json", self.mg_input.mg_input_file)
        if self.keyframe_event_list is not None:
            self.keyframe_event_list.export_to_file(output_dir + os.sep + output_filename)

    def load_from_file(self, file_name, filter_joints=True):
        bvh = BVHReader(file_name)
        self.skeleton = SkeletonBuilder().from_bvh_reader(bvh, filter_joints=filter_joints)

    def generate_bvh_string(self):
        bvh_writer = get_bvh_writer(self.skeleton, self.frames)
        return bvh_writer.generate_bvh_string()

    def to_unity_format(self, scale=1.0):
        """ Converts the frames into a custom json format for use in a Unity client"""
        animated_joints = [j for j, n in list(self.skeleton.nodes.items()) if
                           "EndSite" not in j and len(n.children) > 0]  # self.animated_joints
        unity_frames = []

        for node in list(self.skeleton.nodes.values()):
            node.quaternion_index = node.index

        for frame in self.frames:
            unity_frame = self._convert_frame_to_unity_format(frame, animated_joints, scale)
            unity_frames.append(unity_frame)

        result_object = dict()
        result_object["frames"] = unity_frames
        result_object["frameTime"] = self.frame_time
        result_object["jointSequence"] = animated_joints
        return result_object

    def _convert_frame_to_unity_format(self, frame, animated_joints, scale=1.0):
        """ Converts the frame into a custom json format and converts the transformations
            to the left-handed coordinate system of Unity.
            src: http://answers.unity3d.com/questions/503407/need-to-convert-to-right-handed-coordinates.html
        """
        unity_frame = {"rotations": [], "rootTranslation": None}
        for node_name in list(self.skeleton.nodes.keys()):
            if node_name in animated_joints:
                node = self.skeleton.nodes[node_name]
                if node_name == self.skeleton.root:
                    t = frame[:3] * scale
                    unity_frame["rootTranslation"] = {"x": -t[0], "y": t[1], "z": t[2]}

                if node_name in self.skeleton.animated_joints:  # use rotation from frame
                    # TODO fix: the animated_joints is ordered differently than the nodes list for the latest model
                    index = self.skeleton.animated_joints.index(node_name)
                    offset = index * 4 + 3
                    r = frame[offset:offset + 4]
                    unity_frame["rotations"].append({"x": -r[1], "y": r[2], "z": r[3], "w": -r[0]})
                else:  # use fixed joint rotation
                    r = node.rotation
                    unity_frame["rotations"].append({"x": -r[1], "y": r[2], "z": r[3], "w": -r[0]})
        return unity_frame
