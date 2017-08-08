import numpy as np
from .keyframe_detection import KeyframeDetector
from .utils import convert_poses_to_point_clouds, _convert_pose_to_point_cloud


class Segmentation(object):
    def __init__(self, skeleton, min_segment_size=10):
        self._keyframe_detector = KeyframeDetector(skeleton)
        self.min_segment_size = min_segment_size

    def extract_single_segments(self, motions, start_keyframe, end_keyframe):
        skeleton = self._keyframe_detector._skeleton
        point_clouds = np.array(convert_poses_to_point_clouds(skeleton, motions, normalize=False))
        start_keyframe = _convert_pose_to_point_cloud(skeleton, start_keyframe, normalize=False)
        end_keyframe = _convert_pose_to_point_cloud(skeleton, end_keyframe, normalize=False)
        print(np.array(start_keyframe).shape, point_clouds.shape)
        print(np.array(end_keyframe).shape)

        segments = []
        for idx, m in enumerate(motions):
            start_frame_idx = self._keyframe_detector.find_instance(point_clouds[idx], start_keyframe)
            end_frame_idx = self._keyframe_detector.find_instance(point_clouds[idx], end_keyframe)
            print("found", idx,start_frame_idx, end_frame_idx)
            segments.append(m[start_frame_idx: end_frame_idx])
        return segments

    def extract_segments(self, motions, start_keyframe, end_keyframe, threshold=1.0):
        skeleton = self._keyframe_detector._skeleton
        print("convert motions to point clouds...")
        point_clouds = np.array(convert_poses_to_point_clouds(skeleton, motions, normalize=False))
        start_keyframe = _convert_pose_to_point_cloud(skeleton, start_keyframe, normalize=False)
        end_keyframe = _convert_pose_to_point_cloud(skeleton, end_keyframe, normalize=False)

        print(np.array(start_keyframe).shape, point_clouds.shape)
        print(np.array(end_keyframe).shape)
        print("start keyframe search...")
        segments = []
        for idx, m in enumerate(motions):
            start_frame_indices = self._keyframe_detector.find_instances(point_clouds[idx], start_keyframe, threshold)
            n_instances = len(start_frame_indices)
            print("found ",n_instances,"start keyframe instances in motion", idx)
            if n_instances == 0:
                continue
            for instance_idx, start_frame_idx in enumerate(start_frame_indices):
                if instance_idx+1 == n_instances:
                    search_window_end = len(m)-1
                else:
                    search_window_end = start_frame_indices[instance_idx+1]
                print("found search window", start_frame_idx, search_window_end, len(m))
                if search_window_end-start_frame_idx < self.min_segment_size:
                    print("ignore search window")
                    continue
                search_window = point_clouds[idx][start_frame_idx:search_window_end]
                end_frame_idx = start_frame_idx+self._keyframe_detector.find_instance(search_window, end_keyframe)
                if end_frame_idx-start_frame_idx > self.min_segment_size:
                    print("add segment", start_frame_idx, end_frame_idx)
                    segments.append(m[start_frame_idx: end_frame_idx])
        return segments
