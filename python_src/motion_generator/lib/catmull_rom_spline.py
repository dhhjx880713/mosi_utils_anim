# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 14:54:37 2015

@author: erhe01
"""

import numpy as np
from math import floor,sqrt


class CatmullRomSpline(object):
    '''
    Implements a Catmull-Rom Spline 

    implemented based on the following resources and examples:
    #http://www.cs.cmu.edu/~462/projects/assn2/assn2/catmullRom.pdf
    #http://algorithmist.net/docs/catmullrom.pdf
    #http://www.mvps.org/directx/articles/catmull/
    #http://hawkesy.blogspot.de/2010/05/catmull-rom-spline-curve-implementation.html
    #http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
    '''
    def __init__(self, control_points, dimensions, verbose=False):
        self.verbose = verbose

        #http://algorithmist.net/docs/catmullrom.pdf
        # base matrix to calculate one component of a point on the spline based
        # on the influence of control points
        self._catmullrom_basemat = np.array( [[-1.0,3.0,-3.0,1.0], \
                                             [2.0,-5.0,4.0,-1.0],\
                                             [-1.0,0.0,1.0,0.0],\
                                             [0.0,2.0,0.0,0.0]])
        self.dimensions = dimensions
        self.initiated = False
        self.control_points = []
        if len(control_points) > 0:
            self._initiate_control_points(control_points)
            self.initiated = True


    def _initiate_control_points(self,control_points):
        '''
        @param ordered control_points array of class accessible by control_points[index][dimension]
        '''
        self.number_of_segments = len(control_points)-1
        #as a workaround add multiple points at the end instead of one
        self.control_points = [control_points[0]]+control_points+[control_points[-1],control_points[-1]]
        if self.verbose:
            print "length of control point list ",len(self.control_points)
            print "number of segments ",self.number_of_segments
            print "number of dimensions",self.dimensions
        return

    def add_control_point(self,point):
        """
        Adds the control point at the end of the control point sequence
        """
        #add point replace auxiliary control points
        if  self.initiated:
            del self.control_points[-2:]
            self.number_of_segments = len(self.control_points)-1
            self.control_points += [point,point,point]
        else:
            self._initiate_control_points([point,])
            self.initiated = True


    def clear(self):
        self.control_points = []#[startPoint,startPoint,startPoint]#as a workaround add multiple points at the end instead of one]
        self.initiated = False
      

    def transform_by_matrix(self,matrix):
        '''
        matrix nxn transformation matrix where n is the number of dimensions of the catmull rom spline
        '''
        if self.dimensions < matrix.shape[0]:
            for i in xrange(len(self.control_points)):
                self.control_points[i] = np.dot(matrix, self.control_points[i]+[1])[:3]
    #             print "t",i
        else:
            print "failed",matrix.shape
        return

    def get_full_arc_length(self, granularity = 1000):
        """
        Apprioximate the arc length based on the sum of the finite difference using
        a step size found using the given granularity
        """
        #granularity = self.granularity
        accumulated_steps = np.arange(granularity+1) / float(granularity)
        arc_length = 0.0
        last_point = np.zeros((self.dimensions,1))
        for accumulated_step in accumulated_steps:
            #print "sample",accumulated_step
            point = self.query_point_by_parameter(accumulated_step)
            if point is not None:
                delta = []
                d = 0
                while d < self.dimensions:
                    sq_k = (point[d]-last_point[d])**2
                    delta.append(sqrt (sq_k)  )
                    d+=1
                arc_length += np.sum(delta)
                #arc_length += np.linalg.norm(point-last_point)#(point-last_point).length()
                last_point= point
                #print point
            else:
                raise ValueError('queried point is None at %f'%(accumulated_step))

        return arc_length



    def get_last_control_point(self):
        """
        Returns the last control point ignoring the last auxilliary point
        """
        if len(self.control_points)> 0:
            return np.array(self.control_points[-1])
        else:
            print "no control points defined"
            return np.zeros((1,self.dimensions))



    def map_parameter_to_segment(self,u):
        """
        Returns the segment index associated with parameter u in range
        [0..1]
        Returns the index of the segment and the corresponding relative parameter value in this segment
        """
        #N =self.number_of_segments #(len(self.control_points)-3)/2#number of segments = 1/2 * (control points without the added points and -1 so the last auxiliary control point gets ignored)
#         i = math.floor( self.number_of_segments *t)#the part of t before i, e.g. N = 10 and t = 0.62 => i = 6 and the rest is 0.02
#         localT =(self.number_of_segments*t) -i#the rest
#         i = min(i,self.number_of_segments)
        index =  min(floor( self.number_of_segments *u),self.number_of_segments)#the part of t before i
        local_u =(self.number_of_segments*u) - floor( self.number_of_segments *u)#the rest, e.g. N = 10 and t = 0.62 => i = 6 and the rest is 0.02
        #i = min(i,self.number_of_segments)
        return index+1,local_u#increment i by 1 to ignore the first auxiliary control point

#
    def query_point_by_parameter(self,u):
        """
        Slide 32
        http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
        Returns a point on the curve by parametric value u in the range between
        0 and 1.
        P
        """
        segment_index,local_u = self.map_parameter_to_segment(u)
        weight_vector = [local_u**3,local_u**2,local_u,1]#defines the influence of the 4 closest control points
        control_point_vectors = self._get_control_point_vectors(segment_index)
        #print "segment ",t,  localT,  (self.number_of_segments-1)*t,i
        point =[]
        d =0
        while d < self.dimensions:
            point.append(self._query_component_by_parameter(weight_vector,control_point_vectors[d]))
            d+=1
        return np.array(point)

    def _query_component_by_parameter(self,weight_vector,control_point_vector):
        """
        Slide 32
        http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
        Queries a component of point on the curve based on control points
        and their weights and the _catmullrom_basemat
        """
        transformed_control_point_vector = np.dot(self._catmullrom_basemat,control_point_vector)
        value = np.dot(weight_vector,transformed_control_point_vector)
        return 0.5* value

    def _get_control_point_vectors(self,index):
        """
        Returns the 4 control points that influence values within the i-th segment
        Note the auxiliary segments should not be queried
        """
        #assert i <= self.number_of_segments
        index = int(index)
        d = 0
        vectors = []
        while d < self.dimensions:
            vectors.append([float(self.control_points[index-1][d]),\
                            float(self.control_points[index][d]),\
                            float(self.control_points[index+1][d]),\
                            float(self.control_points[index+2][d])])
            d+=1
        return vectors


