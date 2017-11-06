import numpy as np
from ..animation_data.motion_distance import _point_cloud_distance, _transform_invariant_point_cloud_distance
from .local_minima_search import extracted_filtered_minima


def argmin(values):
    min_idx = 0
    min_v = np.inf
    for idx, v in enumerate(values):
        if v < min_v:
            min_idx = idx
            min_v = v
    return min_idx


def argmin_multi(values, threshold=1.0):
    min_v = np.inf
    for idx, v in enumerate(values):
        if v < min_v:
            min_v = v

    indices = []
    for idx, v in enumerate(values):
        if v <= min_v + threshold:
            indices.append(idx)
    return indices


class KeyframeDetector(object):
    def __init__(self, skeleton):
        self._skeleton = skeleton

    def find_instance(self, point_cloud, keyframe, distance_measure=_transform_invariant_point_cloud_distance):
        distances = []
        for f in point_cloud:
            d = distance_measure(f, keyframe)
            distances.append(d)
        return argmin(distances)

    def calculate_distances(self, point_clouds, keyframe, distance_measure=_transform_invariant_point_cloud_distance):
        D = []
        for m_idx, m in enumerate(point_clouds):
            D.append([])
            for f in m:
                d = distance_measure(f, keyframe)
                D[m_idx].append(d)
        return D

    def find_instances2(self, point_clouds, keyframe):
        """
        Returns:
            result (list<Tuple>): List containing motion index and frame index
        """
        D = self.calculate_distances(point_clouds, keyframe)
        return extracted_filtered_minima(D, 5)

    def find_instances(self, point_cloud, keyframe, threshold=1.0, distance_measure=_transform_invariant_point_cloud_distance):
        distances = []
        for f in point_cloud:
            d = distance_measure(f, keyframe)
            distances.append(d)
        return argmin_multi(distances, threshold)
