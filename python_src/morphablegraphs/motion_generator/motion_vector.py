__author__ = 'erhe01'


from ..animation_data.motion_editing import fast_quat_frames_alignment,\
                                          transform_quaternion_frames
from ..utilities.io_helper_functions import export_quat_frames_to_bvh_file


class MotionVector(object):
    """
    Contains quaternion frames,
    """
    def __init__(self, algorithm_config=None):
        self.n_frames = 0
        self.quat_frames = None
        self.start_pose = None
        if algorithm_config is not None:
            self.apply_spatial_smoothing = algorithm_config["smoothing_settings"]["spatial_smoothing"]
            self.smoothing_window = algorithm_config["smoothing_settings"]["spatial_smoothing_window"]
        else:
            self.apply_spatial_smoothing = False
            self.smoothing_window = 0

    def append_quat_frames(self, new_frames):
        """Align quaternion frames to previous frames

        Parameters
        ----------
        * new_frames: list
            A list of quaternion frames
        """
        if self.quat_frames is not None:
            self.quat_frames = fast_quat_frames_alignment(self.quat_frames,
                                                          new_frames,
                                                          self.apply_spatial_smoothing,
                                                          self.smoothing_window)
        elif self.start_pose is not None:
            self.quat_frames = transform_quaternion_frames(new_frames,
                                                      self.start_pose["orientation"],
                                                      self.start_pose["position"])
        else:
            self.quat_frames = new_frames
        self.n_frames = len(self.quat_frames)

    def export(self, skeleton, output_dir, output_filename, add_time_stamp=True):
        export_quat_frames_to_bvh_file(output_dir, skeleton, self.quat_frames, prefix=output_filename, start_pose=None, time_stamp=add_time_stamp)

    def has_frames(self):
         return self.quat_frames is not None

    def clear(self, end_frame=0):
        if end_frame == 0:
            self.quat_frames = None
        else:
            self.quat_frames = self.quat_frames[:end_frame]
