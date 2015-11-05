# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:28:22 2015

@author: erhe01
"""
import numpy as np
from math import sqrt, acos
from catmull_rom_spline import CatmullRomSpline
from segment_list import SegmentList
from b_spline import BSpline
from fitted_b_spline import FittedBSpline
LOWER_VALUE_SEARCH_FOUND_EXTACT_VALUE = 0
LOWER_VALUE_SEARCH_FOUND_LOWER_VALUE = 1
LOWER_VALUE_SEARCH_VALUE_TOO_SMALL = 2
LOWER_VALUE_SEARCH_VALUE_TOO_LARGE = 3
SPLINE_TYPE_CATMULL_ROM = 0
SPLINE_TYPE_BSPLINE = 1
SPLINE_TYPE_FITTED_BSPLINE = 2

def get_closest_lower_value(arr, left, right, value, getter=lambda A, i: A[i]):
    """
    Uses a modification of binary search to find the closest lower value
    Note this algorithm was copied from http://stackoverflow.com/questions/4257838/how-to-find-closest-value-in-sorted-array
    - left smallest index of the searched range
    - right largest index of the searched range
    - arr array to be searched
    - parameter is an optional lambda function for accessing the array
    - returns a tuple (index of lower bound in the array, flag: 0 = exact value was found, 1 = lower bound was returned, 2 = value is lower than the minimum in the array and the minimum index was returned, 3= value exceeds the array and the maximum index was returned)
    """

    delta = int(right - left)
    if delta > 1:  #test if there are more than two elements to explore
        i_mid = int(left + ((right - left) / 2))
        test_value = getter(arr, i_mid)
        if test_value > value:
            return get_closest_lower_value(arr, left, i_mid, value, getter)
        elif test_value < value:
            return get_closest_lower_value(arr, i_mid, right, value, getter)
        else:
            return i_mid, LOWER_VALUE_SEARCH_FOUND_EXTACT_VALUE
    else:  # always return the lowest closest value if no value was found, see flags for the cases
        left_value = getter(arr, left)
        right_value = getter(arr, right)
        if value >= left_value:
            if value <= right_value:
                return left, LOWER_VALUE_SEARCH_FOUND_LOWER_VALUE
            else:
                return right, LOWER_VALUE_SEARCH_VALUE_TOO_SMALL
        else:
            return left, LOWER_VALUE_SEARCH_VALUE_TOO_LARGE


class ParameterizedSpline(object):
    """ Parameterize a spline by arc length using a mapping table from parameter
    to relative arch length

    Implemented based on the following resources and examples:
    #http://www.cs.cmu.edu/~462/projects/assn2/assn2/catmullRom.pdf
    #http://algorithmist.net/docs/catmullrom.pdf
    #http://www.mvps.org/directx/articles/catmull/
    #http://hawkesy.blogspot.de/2010/05/catmull-rom-spline-curve-implementation.html
    #http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
    """

    def __init__(self, control_points,  spline_type=SPLINE_TYPE_CATMULL_ROM,
                 granularity=1000, closest_point_search_accuracy=0.001,
                 closest_point_search_max_iterations=5000, verbose=False):
        if spline_type == SPLINE_TYPE_CATMULL_ROM:
            self.spline = CatmullRomSpline(control_points, verbose=verbose)
        elif spline_type == SPLINE_TYPE_BSPLINE:
            self.spline = BSpline(control_points)
        elif spline_type == SPLINE_TYPE_FITTED_BSPLINE:
            self.spline = FittedBSpline(control_points, degree=1)
        else:
            raise NotImplementedError()
        self.granularity = granularity
        self.full_arc_length = 0
        self.number_of_segments = 0
        self._relative_arc_length_map = []
        self._update_relative_arc_length_mapping_table()
        self.closest_point_search_accuracy = closest_point_search_accuracy
        self.closest_point_search_max_iterations = closest_point_search_max_iterations

    def _initiate_control_points(self, control_points):
        """
        @param ordered control_points array of class accessible by control_points[index][dimension]
        """
        self.spline._initiate_control_points(control_points)
        self._update_relative_arc_length_mapping_table()
        return

    def add_control_point(self, point):
        """
        Adds the control point at the end of the control point sequence
        """
        if self.spline.initiated:
            self.spline.add_control_point(point)
            self._update_relative_arc_length_mapping_table()
        else:
            self._initiate_control_points([point, ])

    def clear(self):
        self.spline.clear()
        self.full_arc_length = 0
        self.number_of_segments = 0
        self._relative_arc_length_map = []

    def transform_by_matrix(self, matrix):
        """
        matrix nxn transformation matrix where n is the number of dimensions of the catmull rom spline
        """
        self.spline.transform_by_matrix(matrix)

    def _update_relative_arc_length_mapping_table(self):
        """
        creates a table that maps from parameter space of query point to relative arc length based on the given granularity in the constructor of the catmull rom spline
        http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
        """
        self.full_arc_length = 0
        granularity = self.granularity
        accumulated_steps = np.arange(granularity + 1) / float(granularity)
        last_point = None
        number_of_evalulations = 0
        self._relative_arc_length_map = []
        for accumulated_step in accumulated_steps:
            point = self.query_point_by_parameter(accumulated_step)
            if last_point is not None:
                #delta = []
                #d = 0
                #while d < self.spline.dimensions:
                #    delta.append((point[d] - last_point[d])**2)
                #    d += 1
                self.full_arc_length += np.linalg.norm(point-last_point)#sqrt(np.sum(delta))
            self._relative_arc_length_map.append(
                [accumulated_step, self.full_arc_length])
            number_of_evalulations += 1
            last_point = point

        # normalize values
        if self.full_arc_length > 0:
            for i in xrange(number_of_evalulations):
                self._relative_arc_length_map[i][1] /= self.full_arc_length
        return

    def get_full_arc_length(self, granularity=1000):
        """
        Apprioximate the arc length based on the sum of the finite difference using
        a step size found using the given granularity
        """
        accumulated_steps = np.arange(granularity + 1) / float(granularity)
        arc_length = 0.0
        last_point = np.zeros((self.spline.dimensions, 1))
        for accumulated_step in accumulated_steps:
            # print "sample",accumulated_step
            point = self.query_point_by_parameter(accumulated_step)
            if point is not None:
                #delta = []
                #d = 0
                #while d < self.spline.dimensions:
                #    sq_k = (point[d] - last_point[d])**2
                #    delta.append(sq_k)
                #    d += 1
                arc_length += np.lnalg.norm(point-last_point)#sqrt(np.sum(delta))
                last_point = point
                # print point
            else:
                raise ValueError(
                    'queried point is None at %f' %
                    (accumulated_step))

        return arc_length

    def get_absolute_arc_length(self, t):
        """Returns the absolute arc length given a parameter value
        #SLIDE 29
        http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
        gets absolute arc length based on t in relative arc length [0..1]
        """
        step_size = 1.0 / self.granularity
        table_index = int(t / step_size)
        if table_index < len(self._relative_arc_length_map) - 1:
            t_i = self._relative_arc_length_map[table_index][0]
            t_i_1 = self._relative_arc_length_map[table_index + 1][0]
            a_i = self._relative_arc_length_map[table_index][1]
            a_i_1 = self._relative_arc_length_map[table_index + 1][1]
            arc_length = a_i + ((t - t_i) / (t_i_1 - t_i)) * (a_i_1 - a_i)
            arc_length *= self.full_arc_length
        else:
            arc_length = self._relative_arc_length_map[table_index][1] * self.full_arc_length
        return arc_length

    def query_point_by_parameter(self, u):
        return self.spline.query_point_by_parameter(u)

    def query_point_by_absolute_arc_length(self, absolute_arc_length):
        """
        normalize absolute_arc_length and call query_point_by_relative_arc_length
        SLIDE 30 a
        http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
        """

        #point = np.zeros((1, self.spline.dimensions))  # source of bug
        if absolute_arc_length <= self.full_arc_length:
            # parameterize curve by arc length
            relative_arc_length = absolute_arc_length / self.full_arc_length
            return self.query_point_by_relative_arc_length(relative_arc_length)
            #point = self.query_point_by_parameter(relative_arc_length)
        else:
            # return last control point
            return self.spline.get_last_control_point()
            #raise ValueError('%f exceeded arc length %f' % (absolute_arc_length,self.full_arc_length))
        #return point

    def map_relative_arc_length_to_parameter(self, relative_arc_length):
        """
        #see slide 30 b
         http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
        #note it does a binary search so it is rather expensive to be called at every frame
        """
        floorP, ceilP, floorL, ceilL, found_exact_value = self._find_closest_values_in_relative_arc_length_map(
            relative_arc_length)
        if not found_exact_value:
            alpha = (relative_arc_length - floorL) / (ceilL - floorL)
            #t = floorL+alpha*(ceilL-floorL)
            return floorP + alpha * (ceilP - floorP)
        else:
            return floorP

    def query_point_by_relative_arc_length(self, relative_arc_length):
        """Converts relative arc length into a spline parameter between 0 and 1
            and queries the spline for the point.
        """
        u = self.map_relative_arc_length_to_parameter(relative_arc_length)
        return self.query_point_by_parameter(u)

    def get_last_control_point(self):
        """
        Returns the last control point ignoring the last auxilliary point
        """
        return self.spline.get_last_control_point()

    def get_distance_to_path(self, absolute_arc_length, point):
        """ Evaluates a point with absoluteArcLength on self to get a point on the path
        then the distance between the given position and the point on the path is returned
        """
        point_on_path = self.query_point_by_absolute_arc_length(
            absolute_arc_length)
        return np.linalg.norm(point - point_on_path)

    def _find_closest_values_in_relative_arc_length_map(
            self, relative_arc_length):
        """ Given a relative arc length between 0 and 1 it uses get_closest_lower_value
        to search the self._relative_arc_length_map for the values bounding the searched value
        Returns
        -------
        floor parameter,
        ceiling parameter,
        floor arc length,
        ceiling arc length
        and a bool if the exact value was found
        """
        found_exact_value = True
        # search for the index and a flag value, requires a getter for the
        # array
        result = get_closest_lower_value(self._relative_arc_length_map, 0,
                                         len(self._relative_arc_length_map) - 1,
                                         relative_arc_length,
                                         getter=lambda A, i: A[i][1])
        # print result
        index = result[0]
        if result[1] == LOWER_VALUE_SEARCH_VALUE_TOO_SMALL:  # value smaller than smallest element in the array, take smallest value
            ceilP = self._relative_arc_length_map[index][0]
            floorL = self._relative_arc_length_map[index][1]
            floorP = ceilP
            ceilL = floorL
            #found_exact_value = True
        elif result[1] == LOWER_VALUE_SEARCH_VALUE_TOO_LARGE:  # value larger than largest element in the array, take largest value
            ceilP = self._relative_arc_length_map[index][0]
            ceilL = self._relative_arc_length_map[index][1]
            floorP = ceilP
            floorL = ceilL
            #found_exact_value = True
        elif result[1] == LOWER_VALUE_SEARCH_FOUND_EXTACT_VALUE:  # found exact value
            floorP, ceilP = self._relative_arc_length_map[index][0], self._relative_arc_length_map[index][0]
            floorL, ceilL = self._relative_arc_length_map[index][1], self._relative_arc_length_map[index][1]
            #found_exact_value = True
        elif result[1] == LOWER_VALUE_SEARCH_FOUND_LOWER_VALUE:  # found lower value
            floorP = self._relative_arc_length_map[index][0]
            floorL = self._relative_arc_length_map[index][1]
            if index < len(self._relative_arc_length_map):  # check array bounds
                ceilP = self._relative_arc_length_map[index + 1][0]
                ceilL = self._relative_arc_length_map[index + 1][1]
                found_exact_value = False
            else:
                #found_exact_value = True
                ceilP = floorP
                ceilL = floorL

        # print relative_arc_length,floorL,ceilL,found_exact_value
        return floorP, ceilP, floorL, ceilL, found_exact_value

    def get_min_control_point(self, arc_length):
        """yields the first control point with a greater abs arclength than the
        given one"""
        min_index = 0
        num_points = len(self.control_points) - 3
        index = 1
        while index < num_points:
            eval_arc_length, eval_point = self.get_absolute_arc_length_of_point(self.spline.control_points[index])
            print 'check arc length', index, eval_arc_length
            if arc_length < eval_arc_length:
                min_index = index
                break
            index += 1

        return min_index

    def get_tangent_at_arc_length(self, arc_length, eval_range=0.5):
        """
        Returns
        ------
        * dir_vector : np.ndarray
          The normalized direction vector
        * start : np.ndarry
          start of the tangent line / the point evaluated at arc length
        """
        start = self.query_point_by_absolute_arc_length(arc_length)
        magnitude = 0
        while magnitude == 0:  # handle cases where the granularity of the spline is too low
            l1 = arc_length - eval_range
            l2 = arc_length + eval_range
            p1 = self.query_point_by_absolute_arc_length(l1)
            p2 = self.query_point_by_absolute_arc_length(l2)
            dir_vector = p2 - p1
            magnitude = np.linalg.norm(dir_vector)
            eval_range += 0.1
            if magnitude != 0:
                dir_vector /= magnitude
        return start, dir_vector

    def get_angle_at_arc_length_2d(self, arc_length, reference_vector):
        """
        Parameters
        ---------
        * arc_length : float
          absolute arc length for the evaluation of the spline
        * reference_vector : np.ndarray
          2D vector

        Returns
        ------
        * angle : float
          angles in degrees

        """
        assert self.spline.dimensions == 3

        start, tangent_line = self.get_tangent_at_arc_length(arc_length)
        # todo get angle with reference_frame[1]
        a = reference_vector
        b = np.array([tangent_line[0], tangent_line[2]])
        a /= sqrt(a[0]**2 + a[1]**2)
        b /= sqrt(b[0]**2 + b[1]**2)
        angle = acos((a[0] * b[0] + a[1] * b[1]))
        return start, tangent_line, np.degrees(angle)

    def get_absolute_arc_length_of_point(self, point, min_arc_length=0):
        """ Finds the approximate arc length of a point on a spline
        Returns
        -------
        * arc_length : float
          arc length that leads to the closest point on the path with the given
          accuracy. If input point does not lie on the path, i.e. the accuracy
          condition can not be fullfilled -1 is returned)
        """
        u = 0.0
        step_size = 1.0 / self.granularity
        min_distance = np.inf
        min_u = None
        while u <= 1.0:
            if self.get_absolute_arc_length(u) > min_arc_length:
                eval_point = self.query_point_by_parameter(u)
                #delta = eval_point - point
                #distance = 0
                #for v in delta:
                #    distance += v**2
                #distance = sqrt(distance)
                distance = np.linalg.norm(eval_point-point)
                if distance < min_distance:
                    min_distance = distance
                    min_u = u
                    min_point = eval_point
            u += step_size

        if min_u is not None:
            return self.get_absolute_arc_length(min_u), min_point
        else:
            return -1, None

    def find_closest_point(self, point, min_arc_length=0, max_arc_length=-1):
        """ Find closest segment by dividing the closest segments until the
            difference in the distance gets smaller than the accuracy
        Returns
        -------
        * closest_point :  np.ndarray
            point on the spline
        * distance : float
            distance to input point
        """

        if min_arc_length >= self.full_arc_length:  # min arc length was too close to full arc length
            return self.get_last_control_point(), self.full_arc_length
        else:
            segment_list = SegmentList(self.closest_point_search_accuracy, self.closest_point_search_max_iterations)
            segment_list.construct_from_spline(self, min_arc_length, max_arc_length)
            result = segment_list.find_closest_point(point)
            if result[0] is None:
                print "failed to generate trajectory segments for the closest point search"
                print point, min_arc_length, max_arc_length
                return None, -1
            else:
                return result
