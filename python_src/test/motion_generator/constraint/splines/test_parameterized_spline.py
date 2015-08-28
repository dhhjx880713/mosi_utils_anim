# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:36:00 2015

@author: erhe01
"""
import sys
import os
ROOTDIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-5]) + os.sep
sys.path.append(ROOTDIR)
TESTDIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-4]) + os.sep
sys.path.append(TESTDIR)
from libtest import params, pytest_generate_tests
from morphablegraphs.motion_generator.constraints.spatial_constraints.splines.parameterized_spline import ParameterizedSpline


class TestParameterizedSpline(object):
    def setup_method(self, method):
        dimensions = 2
        control_points = [[0, 0],
                          [1, 3],
                          [0, 6],
                          [0, 12]

                          ]
        granularity = 1000
        self.spline = ParameterizedSpline(control_points, dimensions, granularity)

    param_arc_length_parameterization = [{"input_arc_length": 3.0},
                        {"input_arc_length": 0.5},
                        {"input_arc_length": 0.8},
                        {"input_arc_length": 1.8},
                        {"input_arc_length": 0.6},
                        {"input_arc_length": 8.0},
                        {"input_arc_length": 5.0},
                        {"input_arc_length": 11.0},
                        {"input_arc_length": 12.0},

                        ]
    @params(param_arc_length_parameterization)
    def test_arc_length_parameterization(self, input_arc_length):
        """Makes sure that the parameterization works approximately by testing a 2D
        spline at different arc lenghts l as follows:
            x = query_point_by_absolute_arc_length(l) # Query a point by the given arc length,
            l', tmp = find_closest_point(x) # find the closest point on the spline to this queried point
            x' = get_absolute_arc_length_of_point(l') #calculate the arc length of this closest point
            round(x) = round(x')
            round(l) = round(l')
        """

        input_point = self.spline.query_point_by_absolute_arc_length(input_arc_length)
        # print "in",input_arc_length,input_point
        closest_point, distance = self.spline.find_closest_point(input_point)
        # print "closest point",closest_point,distance
        output_arc_length, output_point = self.spline.get_absolute_arc_length_of_point(
            closest_point)

        # print "out",arc_length,eval_point
        #assert np.allclose(input_point, output_point)

        for i in xrange(len(input_point)):
            # print input_point[i],output_point[i],len(input_point)
            # print  round(input_point[i], 1), round(output_point[i], 1)
            assert round(input_point[i], 1) == round(output_point[i], 1)

        # print input_arc_length, output_arc_length
        assert round(input_arc_length, 0) == round(output_arc_length, 0)
