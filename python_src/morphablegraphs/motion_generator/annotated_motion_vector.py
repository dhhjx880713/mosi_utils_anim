import os
from ..animation_data import MotionVector, ROTATION_TYPE_QUATERNION, Skeleton, BVHReader
from ..utilities import write_to_json_file
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
        self.export_annotation(output_dir, output_filename)

    def export_annotation(self,output_dir, output_filename):
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

    def to_unity_format(self, scale=1.0):
        unity_frames = []
        for node in self.skeleton.nodes.values():
            node.quaternion_index = node.index
        for frame in self.frames:
            unity_frame = {"translations": [], "rotations": [], "rootTranslation": None}
            offset = 3
            for node_name in self.skeleton.nodes.keys():#TODO check order of skeleton.nodes dict
                if node_name in self.skeleton.animated_joints:
                    #print node_name
                    node = self.skeleton.nodes[node_name]
                    if node_name == self.skeleton.root:
                        t = (node.get_global_position(frame)-node.offset)*scale
                        unity_frame["rootTranslation"] = {"x": t[0], "y": t[1], "z": t[2]}

                    t = np.array(node.offset) * scale
                    unity_frame["translations"].append({"x": t[0], "y": t[1], "z": t[2]})
                    r = frame[offset:offset+4]
                    
                    unity_frame["rotations"].append({"x": r[1], "y": r[2], "z": r[3], "w": r[0]})
                    offset += 4
            unity_frames.append(unity_frame)

        result_object = dict()
        result_object["frames"] = unity_frames
        result_object["frameTime"] = self.frame_time
        result_object["jointSequence"] = self.skeleton.animated_joints
        return result_object

