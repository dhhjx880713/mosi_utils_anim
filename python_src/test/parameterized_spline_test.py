# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:36:00 2015

@author: erhe01
"""
import sys
import numpy as np
sys.path.append("..")
from motion_generator.constraint.splines.parameterized_spline import ParameterizedSpline

def compare_result(spline,input_arc_length):
    """
    Query a point by the given arc length,
    find the closest point on the spline to this queried point
    then calculate and compare the arc length of this closest point
    """
    #print "test #############"
    input_point = spline.query_point_by_absolute_arc_length(input_arc_length)
    #print "in",input_arc_length,input_point
    closest_point,distance = spline.find_closest_point(input_point)
    #print "closest point",closest_point,distance
    # plot_spline(spline)
    output_arc_length,output_point = spline.get_absolute_arc_length_of_point(closest_point)

    #print "out",arc_length,eval_point
    #assert np.allclose(input_point, output_point)
    
    for i in xrange(len(input_point)):
        #print input_point[i],output_point[i],len(input_point)
        #print  round(input_point[i], 1), round(output_point[i], 1)
        assert round(input_point[i], 1) == round(output_point[i], 1)

    #print input_arc_length, output_arc_length
    assert round(input_arc_length,0) == round(output_arc_length,0)

def test_parameterization():
    """Makes sure that the parameterization works approximately by testing a 2D
    spline at different arc lenghts l as follows:
        x = query_point_by_absolute_arc_length(l)
        l', tmp = find_closest_point(x)
        x' = get_absolute_arc_length_of_point(l')
        round(x) = round(x')
        round(l) = round(l')
    """
    control_points = [[0,0],
                      [1,3],
                      [0,6],
                      [0,12]

                     ]
    dimensions = 2
    granularity = 1000
    spline =  ParameterizedSpline(control_points,dimensions,granularity)
    input_arc_length = 3.0
    compare_result(spline,input_arc_length)
    input_arc_length = 0.5
    compare_result(spline,input_arc_length)
    input_arc_length = 0.8
    compare_result(spline,input_arc_length)
    input_arc_length = 1.8
    compare_result(spline,input_arc_length)
    input_arc_length = 0.6#9.8
    compare_result(spline,input_arc_length)
    input_arc_length = 8.0
    compare_result(spline,input_arc_length)
    input_arc_length = 5.0
    compare_result(spline,input_arc_length)
    input_arc_length = 10.0
    compare_result(spline,input_arc_length)
    input_arc_length = 11.0
    compare_result(spline,input_arc_length)
    input_arc_length = 12.0
    compare_result(spline,input_arc_length)
#    point = np.array([1.0,1.2])

#    print "full arc length",arc_length,spline.get_full_arc_length()
#    print "parameter",spline.query_point_by_parameter(arc_length)
#    arc_length = arc_length* spline.get_full_arc_length()
#    print "arc length",spline.query_point_by_absolute_arc_length(arc_length)

if __name__ == "__main__":
    test_parameterization()
