# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:42:13 2015

@author: erhe01
"""
from animation_data.motion_editing import align_quaternion_frames
from constraint_check import check_dir_constraint, check_frame_constraint, check_pos_and_rot_constraint
       
class MotionPrimitiveConstraints(object):
    """ Represents the input to the generate_motion_primitive_from_constraints
        method of the MotionPrimitiveGenerator class.
    Attributes
     -------
     * constraints : list of dicts
      Each dict contains joint, position,orientation and semanticAnnotation describing a constraint
    """
    def __init__(self):
        self.pose_constraint_set = False
        self.motion_primitive_name = None
        self.settings = None
        self.constraints = []
        self.goal_arc_length = 0           
        self.use_optimization = False
        self.step_goal = None
        self.step_start = None
        self.start_pose = None
        self.skeleton = None
        self.precision = {"pos": 1, "rot": 1, "smooth": 1}
        self.verbose = False
        
    def print_status(self):
#        print  "starting from",last_pos,last_arc_length,"the new goal for", \
#                current_motion_primitive,"is",goal,"at arc length",arc_length
        print "starting from: "
        print self.step_start
        print "the new goal for " + self.motion_primitive_name
        print self.step_goal
        print "arc length is: " + str(self.goal_arc_length)

    def evaluate(self, motion_primitive, sample, prev_frames, use_time_parameters=False):
        """
        Calculates the error of a list of constraints given a sample parameter value s.
        
        Returns
        -------
        * sum_error : float
        \tThe sum of the errors for all constraints
        * sucesses : list of bool
        \tSets for each entry in the constraints list wether or not a given precision was reached
    
        """
        error_sum = 0
        #find aligned frames once for all constraints
        quat_frames = motion_primitive.back_project(sample, use_time_parameters=use_time_parameters).get_motion_vector()
        aligned_frames  = align_quaternion_frames(quat_frames, prev_frames, self.start_pose)
    
        for c in self.constraints:
             error_sum += c.evaluate_motion_sample(aligned_frames)#self._check_constraint(aligned_frames, c)

        return error_sum
        

    
    def _check_constraint(self, quat_frames, constraint):
        """ Main function of the modul. Check whether a sample fullfiles the
        constraint with the given precision or not. 
        Note only one type of constraint is allowed at once.
    
        Parameters
        ----------
        * quat_frames: np.ndarry
            contains the animation that is supposed to be checked
        * constraint : tuple
            The constraint as (joint, [pos_x, pos_y, pos_z], [rot_x, rot_y, rot_z])
            where unconstrained variables are set to None
    
        Returns
        -------
        * error: float
            the distance to the constraint calculated using l2 norm ignoring None
        """
              
        constrain_first_frame = constraint["semanticAnnotation"]["firstFrame"]
        constrain_last_frame = constraint["semanticAnnotation"]["lastFrame"]
        #handle the different types of constraints
        if "dir_vector" in constraint.keys() and constrain_last_frame: # orientation constraint on last frame
           error, in_precision = check_dir_constraint(quat_frames,
                                       constraint["dir_vector"],
                                        self.precision["rot"])
           return error
        elif "frame_constraint" in constraint.keys():
           error, in_precision = check_frame_constraint(quat_frames, constraint["frame_constraint"],
                                      self.precision["smooth"], self.skeleton)
           return error
    
        elif "position" or "orientation" in constraint.keys():
            good_frames, bad_frames = check_pos_and_rot_constraint(quat_frames, 
                                                                   constraint,
                                                                   self.precision,
                                                                   self.skeleton,
                                                                   (constrain_first_frame,
                                                                    constrain_last_frame),
                                                                    verbose=self.verbose)
            if len(good_frames)>0:               
                c_min_distance = min((zip(*good_frames))[1])
            else:
                c_min_distance = min((zip(*bad_frames))[1])
            return c_min_distance
        else:
            print "Error: Constraint type not recognized"
            return 10000
