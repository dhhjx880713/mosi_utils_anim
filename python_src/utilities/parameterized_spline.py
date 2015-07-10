# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:28:22 2015

@author: erhe01
"""
import numpy as np
from matplotlib import pyplot as plt
import datetime
from math import floor,sqrt,acos
import heapq
from catmull_rom_spline import CatmullRomSpline



def get_angle_between_vectors2d(a,b):
    """
    Returns
    -------
    angle in degrees
    """
    a = a/(sqrt(a[0]**2 + a[1]**2))
    b = b/(sqrt(b[0]**2 + b[1]**2))
    return acos((a[0] * b[0] + a[1]* b[1]))

def sign(value):
    if value >= 0:
        return 1
    else:
        return  -1

def get_magnitude(vector):
    magnitude = 0
    for v in vector:
        magnitude += v**2
    magnitude = sqrt(magnitude)
    return magnitude
    


def get_closest_lower_value(arr,left,right,value,getter= lambda A,i : A[i]):
    '''
    Uses a modification of binary search to find the closest lower value
    Note this algorithm was copied from http://stackoverflow.com/questions/4257838/how-to-find-closest-value-in-sorted-array
    - left smallest index of the searched range
    - right largest index of the searched range
    - arr array to be searched
    - parameter is an optional lambda function for accessing the array
    - returns a tuple (index of lower bound in the array, flag: 0 = exact value was found, 1 = lower bound was returned, 2 = value is lower than the minimum in the array and the minimum index was returned, 3= value exceeds the array and the maximum index was returned)
    '''


    delta = int(right -left)
    #print delta
    if (delta> 1) :#or (left ==0 and (delta> 0) ):# or (right == len(A)-1 and ()):#test if there are more than two elements to explore
        i_mid = int(left+((right-left)/2))
        test_value = getter(arr,i_mid)
        #print "getter",testValue
        if test_value>value:
            #print "right"
            return get_closest_lower_value(arr, left, i_mid, value,getter)
        elif test_value<value:
            #print "left"
            return get_closest_lower_value(arr, i_mid, right, value,getter)
        else:
            #print "done"
            return (i_mid,0)
    else:#always return the lowest closest value if no value was found, see flags for the cases
        left_value = getter(arr,left)
        right_value = getter(arr,right)
        if value >= left_value:
            if value <= right_value:
                return (left,1)
            else:
                return (right,2)
        else:
            return(left,3)


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
    def __init__(self, control_points, dimensions, granularity=1000, verbose=False):
        self.spline = CatmullRomSpline(control_points, dimensions, verbose=verbose)
        self.granularity = granularity
        self.full_arc_length = 0
        self.number_of_segments = 0
        self._relative_arc_length_map = []
        self._update_relative_arc_length_mapping_table()
        return

    def _initiate_control_points(self,control_points):
        '''
        @param ordered control_points array of class accessible by control_points[index][dimension]
        '''
        self.spline._initiate_control_points(control_points)
        self._update_relative_arc_length_mapping_table()
        return

    def add_control_point(self,point):
        """
        Adds the control point at the end of the control point sequence
        """
        if self.spline.initiated:
            self.spline.add_control_point(point)
            self._update_relative_arc_length_mapping_table()
        else:
            self._initiate_control_points([point,])
            

    def clear(self):
        self.spline.clear()
        self.full_arc_length = 0
        self.number_of_segments = 0
        self._relative_arc_length_map = []

    def transform_by_matrix(self,matrix):
        '''
        matrix nxn transformation matrix where n is the number of dimensions of the catmull rom spline
        '''
        self.spline.transform_by_matrix(matrix)


    def _update_relative_arc_length_mapping_table(self):
        '''
        creates a table that maps from parameter space of query point to relative arc length based on the given granularity in the constructor of the catmull rom spline
        http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
        '''
        self.full_arc_length = 0
        granularity = self.granularity
        accumulated_steps = np.arange(granularity+1) / float(granularity)
        last_point = None
        number_of_evalulations = 0
        self._relative_arc_length_map = []
        for accumulated_step in accumulated_steps:
            point = self.query_point_by_parameter(accumulated_step)
            if last_point is not None:
                delta = []
                d = 0
                while d < self.spline.dimensions:
                    delta.append(sqrt((point[d]-last_point[d])**2))
                    d+=1
                self.full_arc_length += np.sum(delta)#(point-last_point).length()
                #print self.full_arc_length
            self._relative_arc_length_map.append([accumulated_step,self.full_arc_length])
            number_of_evalulations += 1
            last_point = point

        #normalize values
        if self.full_arc_length > 0 :
            for i in range(number_of_evalulations):
                self._relative_arc_length_map[i][1] /= self.full_arc_length

#        for entry in self._relative_arc_length_map:
#            print entry
        return

    def get_full_arc_length(self, granularity = 1000):
        """
        Apprioximate the arc length based on the sum of the finite difference using
        a step size found using the given granularity
        """
        #granularity = self.granularity
        accumulated_steps = np.arange(granularity+1) / float(granularity)
        arc_length = 0.0
        last_point = np.zeros((self.spline.dimensions,1))
        for accumulated_step in accumulated_steps:
            #print "sample",accumulated_step
            point = self.query_point_by_parameter(accumulated_step)
            if point is not None:
                delta = []
                d = 0
                while d < self.spline.dimensions:
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


    def get_absolute_arc_length(self,t):
        """Returns the absolute arc length given a paramter value
        #SLIDE 29
        http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
        gets absolute arc length based on t in relative arc length [0..1]
        """
        step_size = 1.0/self.granularity
        table_index = int(t/step_size)
        if table_index < len(self._relative_arc_length_map)-1:
            t_i =self._relative_arc_length_map[table_index][0]
            t_i_1 =self._relative_arc_length_map[table_index+1][0]
            a_i =self._relative_arc_length_map[table_index][1]
            a_i_1 =self._relative_arc_length_map[table_index+1][1]

            arc_length =a_i+ ((t -t_i )/(t_i_1 - t_i))* (a_i_1-a_i)
            #unscale
            arc_length = arc_length*self.full_arc_length
        else:
            arc_length= self._relative_arc_length_map[table_index][1]*self.full_arc_length
        return arc_length
#        return t*self.full_arc_length

    def query_point_by_parameter(self, u):
        return self.spline.query_point_by_parameter(u)
        
    def query_point_by_absolute_arc_length(self,absolute_arc_length):
        """
        normalize absolute_arc_length and call query_point_by_relative_arc_length
        SLIDE 30 a
        http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
        """

        point = np.zeros((1,self.spline.dimensions))#source of bug
        if absolute_arc_length <= self.full_arc_length:
            # parameterize curve by arc length
            relative_arc_length = absolute_arc_length/self.full_arc_length
            point = self.query_point_by_relative_arc_length(relative_arc_length)
            #point = self.query_point_by_parameter(relative_arc_length)
        else:
            #return last control point
            point = self.spline.get_last_control_point()
            #raise ValueError('%f exceeded arc length %f' % (absolute_arc_length,self.full_arc_length))
        return point

    def map_relative_arc_length_to_parameter(self, relative_arc_length):
        """
        #see slide 30 b
         http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
        #note it does a binary search so it is rather expensive to be called at every frame
        """
        floorP,ceilP,floorL,ceilL,found_exact_value = self._find_closest_values_in_relative_arc_length_map(relative_arc_length)
        if not found_exact_value:
            alpha = (relative_arc_length-floorL)/(ceilL-floorL)
            #t = floorL+alpha*(ceilL-floorL)
            u = floorP+alpha*(ceilP-floorP)
        else:
            u = floorP
        return u

    def query_point_by_relative_arc_length(self,relative_arc_length):
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

    def get_distance_to_path(self,absolute_arc_length, point):
        '''
        evaluates a point with absoluteArcLength on self to get a point on the path
        then the distance between the given position and the point on the path is returned
        '''
        point_on_path = self.get_point_at_absolute_arc_length(absolute_arc_length)
        return np.linalg.norm(point-point_on_path)

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


    def _find_closest_values_in_relative_arc_length_map(self,relative_arc_length):
        '''Given a relative arc length between 0 and 1 it uses get_closest_lower_value
        to search the self._relative_arc_length_map for the values bounding the searched value
        Returns
        -------
        floor parameter,
        ceiling parameter,
        floor arc length,
        ceiling arc length
        and a bool if the exact value was found
        '''
        found_exact_value = True
        #search for the index and a flag value, requires a getter for the array
        result = get_closest_lower_value(self._relative_arc_length_map,0,\
                                        len(self._relative_arc_length_map)-1,\
                                        relative_arc_length,\
                                        getter = lambda A,i: A[i][1])
        #print result
        index = result[0]

        if result[1] == 0:#found exact value
            floorP, ceilP = self._relative_arc_length_map[index][0],self._relative_arc_length_map[index][0]
            floorL, ceilL = self._relative_arc_length_map[index][1],self._relative_arc_length_map[index][1]
            found_exact_value = True
        elif result[1] ==1:#found lower value
            floorP = self._relative_arc_length_map[index][0]
            floorL = self._relative_arc_length_map[index][1]
            if index <len(self._relative_arc_length_map):#check array bounds
                ceilP = self._relative_arc_length_map[index+1][0]
                ceilL = self._relative_arc_length_map[index+1][1]
                found_exact_value = False
            else:
                found_exact_value = True
                ceilP= floorP
                ceilL = floorL
        elif result[1] ==2:#value smaller than smallest element in the array
            ceilP = self._relative_arc_length_map[index][0]
            floorL = self._relative_arc_length_map[index][1]
            floorP  = ceilP
            ceilL = floorL
            found_exact_value = True
        elif result[1] ==3:#value larger than largest element in the array
            ceilP = self._relative_arc_length_map[index][0]
            ceilL = self._relative_arc_length_map[index][1]
            floorP = ceilP
            floorL = ceilL
            found_exact_value = True
        #print relative_arc_length,floorL,ceilL,found_exact_value
        return floorP,ceilP,floorL,ceilL,found_exact_value



        
    def _construct_segment_list(self, min_arc_length=0, max_arc_length=-1,granularity=1000):
        """ Constructs line segments out of the evualated points
         with the given granularity
        Returns
        -------
        * segments : list of tuples
            Each entry defines a line segment and contains
            start,center and end points
        """
        points = []
        step_size = 1.0/granularity
        u = 0 #min_u
        if max_arc_length <= 0:
            max_arc_length = self.full_arc_length
            
        while u <= 1.0 :
            arc_length = self.get_absolute_arc_length(u)
            if  arc_length >= min_arc_length  and arc_length <=  max_arc_length:#todo make more efficient by looking up min_u
                point = self.query_point_by_parameter(u)
                points.append(point)
            u += step_size

        segments = []
        index = 0
        while index < len(points)-1:
            start = np.array(points[index])
            end = np.array(points[index+1])
            center = 0.5 * (end - start) + start
            segment = (start,center,end)
            segments.append(segment)
            index += 1
        return segments
        
        
    def _divide_segment(self,segment):
        """Divides a segment into two segments
        Returns
        -------
        * segments : list of tuples
            Contains segment_a and segment_b. Each defines a line segment and 
            contains start,center and end points
        """
        start_a = segment[0]
        end_a = segment[1]
        center_a = 0.5 * (end_a-start_a) + start_a
        start_b = segment[1]
        end_b =segment[2]
        center_b = 0.5 * (end_b-start_b) + start_b

#        print
#        print "old",segment
#        print  "new segment a",start_a,end_a
#        print "new segment b",start_b,end_b
        return [(start_a,center_a,end_a),(start_b,center_b,end_b)]

    def get_min_control_point(self,arc_length):
        """yields the first control point with a greater abs arclength than the
        given one"""
        min_index = 0
        num_points = len(self.control_points)-3
        index = 1
        while index < num_points:
            eval_arc_length,eval_point = self.get_absolute_arc_length_of_point(self.spline.control_points[index])
            print 'check arc length',index,eval_arc_length
            if arc_length < eval_arc_length:
                min_index = index
                break
            index+=1

        return min_index
        
    def get_tangent_at_arc_length(self,arc_length,eval_range=0.5):
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
        while magnitude == 0:#handle cases where the granularity of the spline is too low
            l1 = arc_length-eval_range
            l2 = arc_length+eval_range
            p1 = self.query_point_by_absolute_arc_length(l1)
            p2 = self.query_point_by_absolute_arc_length(l2)
            dir_vector = p2 - p1
            magnitude = get_magnitude(dir_vector)
            eval_range +=0.1
            if magnitude != 0:
                dir_vector = dir_vector/magnitude
                
        return  start,dir_vector


    def  get_angle_at_arc_length_2d(self,arc_length,reference_vector):
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

        start,tangent_line = self.get_tangent_at_arc_length(arc_length)
        #todo get angle with reference_frame[1],

        angle = get_angle_between_vectors2d(reference_vector,\
                                    np.array([tangent_line[0],tangent_line[2]]) )
        return start,tangent_line,np.degrees(angle)

    def get_absolute_arc_length_of_point(self,point,accuracy =10.0,min_arc_length = 0):
        """ Finds the approximate arc length of a point on a spline
        Returns
        -------
        * arc_length : float
          arc length that leads to the closest point on the path with the given
          accuracy. If input point does not lie on the path, i.e. the accuracy
          condition can not be fullfilled -1 is returned)
        """

        u = 0.0
        step_size = 1.0/self.granularity
        min_distance = np.inf
        min_u = None
        while u <= 1.0 :
            if self.get_absolute_arc_length(u )  > min_arc_length:
                eval_point = self.query_point_by_parameter(u)
                delta = eval_point - point
                distance = 0
                for v in delta:
                    distance += v**2
                distance = sqrt(distance)
                if distance < min_distance:
                    min_distance = distance
                    min_u = u
                    min_point=eval_point
            u += step_size

        if min_u is not None:
            return self.get_absolute_arc_length(min_u ) ,min_point
        else:
            return -1,None

      
        

    def _find_closest_point_on_segment(self,point,segment,accuracy= 0.001, max_iterations = 5000):
        """ Find closest point by dividing the segment until the 
            difference in the distance gets smaller than the accuracy
        Returns
        -------
        * closest_point :  np.ndarray
            point on the spline
        * distance : float
            distance to input point
        """
#        print "start closest point search",point
        segment_length = np.inf
        distance = np.inf
        segments = self._divide_segment(segment)

        iteration = 0
        while segment_length > accuracy and distance > accuracy and iteration <max_iterations:
            closest_segment, distance = self._find_closest_segment(point,segments)
            
            delta = closest_segment[2] - closest_segment[0]
            s_length = 0
            for v in delta:
                s_length += v**2
            segment_length = sqrt(segment_length)
            segments = self._divide_segment(closest_segment)
            iteration += 1
        closest_point = closest_segment[1] #extract center of closest segment
        return closest_point,distance
#
        
    
#    def find_closest_point(self, point, accuracy=0.001, max_iterations=5000, min_arc_length=0, max_arc_length=-1):
#        return self.spline.find_closest_point(point, accuracy, max_iterations, min_arc_length, max_arc_length)

    def find_closest_point(self, point, accuracy=0.001, max_iterations=5000, min_arc_length=0, max_arc_length=-1):
        """ Find closest segment by dividing the closest segments until the 
            difference in the distance gets smaller than the accuracy
        Returns
        -------
        * closest_point :  np.ndarray
            point on the spline
        * distance : float
            distance to input point
        """
        if min_arc_length >= self.full_arc_length:
            return self.get_last_control_point(),0.0
        else:
            #first_control_point = self.get_min_control_point(min_arc_length)
            segments = self._construct_segment_list(min_arc_length=min_arc_length,max_arc_length=max_arc_length)#first_control_point=first_control_point
    #        closest_segment, distance = self._find_closest_segment(point,segments)
    #        closest_point,distance = self.find_closest_point_on_segment(point,closest_segment,accuracy,max_iterations,min_arc_length)        
    #        return closest_point,distance
            if len(segments) == 0:#min arc length was too close to full arc length
                print point,min_arc_length,max_arc_length,len(segments)
                
                return self.get_last_control_point(),self.full_arc_length
            candidates = self. _find_two_closest_segments(point,segments)
            if len(candidates) >= 2:
                closest_point_1,distance_1 = self._find_closest_point_on_segment(point,candidates[0][1],accuracy,max_iterations)        
                closest_point_2,distance_2 = self._find_closest_point_on_segment(point,candidates[1][1],accuracy,max_iterations)          
                    
                if distance_1 < distance_2:
                    return closest_point_1,distance_1
                else:
                    return closest_point_2,distance_2
            elif len(candidates) == 1:
                closest_point,distance = self._find_closest_point_on_segment(point,candidates[0][1],accuracy,max_iterations)        
                return closest_point,distance
            else:
                print "failed to generate trajectory segments for the closest point search"
                print point,min_arc_length,max_arc_length,len(segments)
                return None,-1
            
    def _find_closest_segment(self,point,segments):
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
        for s in segments:
            delta = s[1]-point
            distance = 0
            for v in delta:
                distance += v**2
            distance = sqrt(distance)
            if distance < min_distance:
                closest_segment = s
                min_distance = distance
        return closest_segment,min_distance

    def _find_two_closest_segments(self,point,segments):
        """ Ueses a heap queue to find the two closest segments
        Returns
        -------
        * closest_segments : List of Tuples
           distance to the segment center
           Defineiation of a line segment. Contains start,center and end points
           
        """
        heap = [] # heap queue 
        index = 0
        while index  < len(segments):
            delta = segments[index][1]-point
            distance = 0
            for v in delta:
                distance += v**2
            distance = sqrt(distance)  
#            print point,distance,segments[index]
#            #Push the value item onto the heap, maintaining the heap invariant.
            heapq.heappush(heap, (distance,index) ) 
            index += 1 
            
        closest_segments = []
        count = 0
        while len(heap) > 0 and count < 2:
            distance,index = heapq.heappop(heap)
            segment = (distance,segments[index])
            closest_segments.append(segment)
            count += 1
        return closest_segments   



def verify_spline():
    control_points = [[0,0,0],
                      [0,1,0],
                      [0,8,0]
                     ]
    dimensions = 3
    granularity = 1000
    spline =  ParameterizedSpline(control_points,dimensions,granularity)
    print spline.query_point_by_parameter(0.5)
    print spline.query_point_by_relative_arc_length(0.5)
    print spline.full_arc_length


def plot_line(start,dir_vector,ax= None,length=0.1):

    end = dir_vector* length  + start
    #print start,end
    line = np.array([start,end]).T
    if ax is not None:
        ax.plot(line[0],line[1])
    else:
        plt.plot(line[0],line[1])

def plot_line_with_text(start,dir_vector,text,ax=None,length=0.1):
    plot_line(start,dir_vector,ax,length)
    if ax is not None:
        ax.text(start[0],start[1],text, size=8, ha='center', va='center')
    else:
        plt.text(start[0],start[1],text, size=8, ha='center', va='center')




def plot_spline(spline,granularity = 1000.0, ax = None):

    us = np.arange(1,granularity) / granularity
    print "max",us[-1]
    full_len = spline.get_full_arc_length(1000) # should be equal to full_arc_length
    reference_vector = np.array([1,0])
    v,tangents= zip (* [(  spline.query_point_by_parameter(u),  spline.get_angle_at_arc_length_2d(u*full_len,reference_vector) ) for u in us])

#    us = np.arange(1,granularity) / granularity
#    us = us * full_len
#    v = [(  spline.query_point_by_absolute_arc_length(u) ) for u in us]
#

#   print "tangent",spline.get_tangent_at_arc_length(3)
#    reference_frame = [[0,0],[1,1],[1,0] ]
#    print spline.get_orientation_at_arc_length(3,reference_frame)
    v = np.array(v).T
    px = [ p[0] for p in spline.control_points]
    py = [ p[1] for p in spline.control_points]
#    x = [ p[0] for p in v]
#    y = [ p[1] for p in v]
#    vs = np.array(vs)
#    xs,ys = [x,y]
#    y = vs[1,:]
#    print x.shape
#    print y.shape
#    print len(x)
#    print len(y)

    if not ax:
       fig = plt.figure()
       plt.plot(v[0],v[1])#x,y
       plt.plot(px,py,'ro')
       count = 0
       for start, dir_vector,angle in tangents:
           if count %20 == 0:
               plot_line_with_text(start,dir_vector,str(angle),ax=None,length=1.0)
#           else:
#               plot_line(start,dir_vector,ax=None,length=1.0)
           count+=1
       fig.show()

    else:
        ax.plot(v[0],v[1])#x,y
        ax.plot(px,py,'ro')
        for start, dir_vector,angle in tangents:
           plot_line_with_text(start,dir_vector,str(angle),ax=None,length=1.0)



def plot_splines(title,splines,granularity = 100.0, ax = None,output_dir=None):
    us = np.arange(1,granularity) / granularity
    show = False
    if not ax:

       fig = plt.figure()

       ax = fig.add_subplot(111)
       ax.set_xlim([-50,50])
       ax.set_ylim([-50,50])
       #fig.subplots_adjust(top=0.85)
       ax.set_title(title)
       show = True
    for spline in splines:
        v = [spline.query_point_by_parameter(u) for u in us]
        v = np.array(v).T
        px = [ p[0] for p in spline.control_points]
        py = [ p[1] for p in spline.control_points]
        ax.plot(v[0],v[1])#x,y
        ax.plot(px,py,'ro')

    if show:
      fig.show()
      if output_dir is not None:
          time_code = unicode(datetime.datetime.now().strftime("%d%m%y_%H%M%S"))
          filename = output_dir + title + time_code+".png"
          plt.savefig(filename, format='png')


def test_plot_spline():


    control_points = [[0,0],
                      [1,1],
                      [2,10],
                      [4,5],
                      [3,2],
                      [2,1],
                      [1,3],
                      [1,5]
                     ]
    dimensions = 2
    granularity = 1000
    spline =  CatmullRomSpline(control_points,dimensions,granularity)
    plot_spline(spline)


def compare_result(spline,input_arc_length):
    """
    query a point by the given arc length
    then find the closest point on the spline to this queried point
    then calculate and print the arc length of this closest point
    """
    print "test #############"
    point = spline.query_point_by_absolute_arc_length(input_arc_length)
    print "in",input_arc_length,point
    closest_point,distance = spline.find_closest_point(point)
    print "closest point",closest_point,distance
    # plot_spline(spline)
    arc_length,eval_point = spline.get_absolute_arc_length_of_point(closest_point)

    print "out",arc_length,eval_point

def test_find_closest_point():
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

def main():
    verify_spline()
    #test_plot_spline()
    test_find_closest_point()
    return



if __name__ ==  "__main__":


    main()