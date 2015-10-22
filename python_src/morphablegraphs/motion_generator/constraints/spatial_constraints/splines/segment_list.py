__author__ = 'erhe01'

import heapq
import numpy as np
from math import sqrt
from spline_segment import SplineSegment


class SegmentList(object):
    def __init__(self, closest_point_search_accuracy=0.001, closest_point_search_max_iterations=5000, segments=None):
        self.segments = segments
        self.closest_point_search_accuracy = closest_point_search_accuracy
        self.closest_point_search_max_iterations = closest_point_search_max_iterations

    def construct_from_spline(self, spline, min_arc_length=0, max_arc_length=-1, granularity=1000):
        """ Constructs line segments out of the evualated points
         with the given granularity
        Returns
        -------
        * segments : list of tuples
            Each entry defines a line segment and contains
            start,center and end points
        """
        points = []
        step_size = 1.0 / granularity
        u = 0
        if max_arc_length <= 0:
            max_arc_length = spline.full_arc_length
        while u <= 1.0:
            arc_length = spline.get_absolute_arc_length(u)
            # todo make more efficient by looking up min_u
            if arc_length >= min_arc_length and arc_length <= max_arc_length:
                point = spline.query_point_by_parameter(u)
                points.append(point)
            u += step_size

        self.segments = []
        index = 0
        while index < len(points) - 1:
            start = np.array(points[index])
            end = np.array(points[index + 1])
            center = 0.5 * (end - start) + start
            segment = SplineSegment(start, center, end)
            self.segments.append(segment)
            index += 1

    def find_closest_point(self, point):
        if self.segments is None or len(self.segments) == 0:
            return None, -1
        candidates = self.find_two_closest_segments(point)
        if len(candidates) >= 2:
            closest_point_1, distance_1 = self._find_closest_point_on_segment(candidates[0][1], point)
            closest_point_2, distance_2 = self._find_closest_point_on_segment(candidates[1][1], point)
            if distance_1 < distance_2:
                return closest_point_1, distance_1
            else:
                return closest_point_2, distance_2
        elif len(candidates) == 1:
            closest_point, distance = self._find_closest_point_on_segment(candidates[0][1], point)
            return closest_point, distance

    def find_closest_segment(self, point):
        """
        Returns
        -------
        * closest_segment : Tuple
           Defines line segment. Contains start,center and end
        * min_distance : float
          distance to this segments center
        """
        closest_segment = None
        min_distance = np.inf
        for s in self.segments:
            delta = s.center - point
            distance = 0
            for v in delta:
                distance += v**2
            distance = sqrt(distance)
            if distance < min_distance:
                closest_segment = s
                min_distance = distance
        return closest_segment, min_distance

    def find_two_closest_segments(self, point):
        """ Ueses a heap queue to find the two closest segments
        Returns
        -------
        * closest_segments : List of Tuples
           distance to the segment center
           Defineiation of a line segment. Contains start,center and end points

        """
        heap = []  # heap queue
        index = 0
        while index < len(self.segments):
            delta = self.segments[index].center - point
            distance = 0
            for v in delta:
                distance += v**2
            distance = sqrt(distance)
#            print point,distance,segments[index]
#            #Push the value item onto the heap, maintaining the heap invariant.
            heapq.heappush(heap, (distance, index))
            index += 1

        closest_segments = []
        count = 0
        while len(heap) > 0 and count < 2:
            distance, index = heapq.heappop(heap)
            segment = (distance, self.segments[index])
            closest_segments.append(segment)
            count += 1
        return closest_segments

    def _find_closest_point_on_segment(self, segment, point):
            """ Find closest point by dividing the segment until the
                difference in the distance gets smaller than the accuracy
            Returns
            -------
            * closest_point :  np.ndarray
                point on the spline
            * distance : float
                distance to input point
            """
            segment_length = np.inf
            distance = np.inf
            segment_list = SegmentList(self.closest_point_search_accuracy, self.closest_point_search_max_iterations, segment.divide())
            iteration = 0
            while segment_length > self.closest_point_search_accuracy and distance > self.closest_point_search_accuracy and iteration < self.closest_point_search_max_iterations:
                closest_segment, distance = segment_list.find_closest_segment(point)
                delta = closest_segment.end - closest_segment.start
                s_length = 0
                for v in delta:
                    s_length += v**2
                segment_length = sqrt(segment_length)
                segment_list = SegmentList(self.closest_point_search_accuracy, self.closest_point_search_max_iterations, closest_segment.divide())
                iteration += 1
            closest_point = closest_segment.center  # extract center of closest segment
            return closest_point, distance
