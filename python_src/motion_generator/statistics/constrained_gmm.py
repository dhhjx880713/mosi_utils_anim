# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 10:09:45 2015

@author: mamauer,FARUPP,erhe01
"""

import numpy as np
import sklearn.mixture as mixture
from ..constraint.constraint_check import check_constraint, find_aligned_quaternion_frames
from operator import itemgetter
from utilities.exceptions import ConstraintError

class ConstrainedGMM(mixture.GMM):
    """ A GMM that has the ability to constraint itself. The GMM is build based
    on a GMM

    Parameters
    ----------
    * motion_primitve_node : MotionPrimitiveNode
        The original MotionPrimitive which will be constrained
    * constraint : tuple, optional
        The constraint as (joint, [pos_x, pos_y, pos_z], [rot_x, rot_y, rot_z])
        where unconstrained variables are set to None. The default is None,
        which means that no constraint is set.

    """
    def __init__(self,motion_primitve_node, gmm, algorithm_config, start_pose, skeleton):
        super(ConstrainedGMM, self).__init__(
            n_components=gmm.n_components,
            covariance_type=gmm.covariance_type,
            thresh=gmm.thresh,
            min_covar=gmm.min_covar,
            random_state=gmm.random_state,
            n_iter=gmm.n_iter,
            n_init=gmm.n_init,
            params=gmm.params,
            init_params=gmm.init_params
        )
        self.motion_primitve_node = motion_primitve_node
        self.verbose = algorithm_config["verbose"]
        self.start_pose = start_pose
        self.skeleton = skeleton

        self.mm_ = motion_primitve_node.motion_primitive
        self.weights_ = gmm.weights_
        self.means_ = gmm.means_
        self.converged_ = gmm.converged_
        self.covars_ = gmm.covars_
        self.samples_ = None
    
        self.n_random_samples = self._algorithm_config["n_random_samples"]
        self.max_bad_samples = algorithm_config["constrained_gmm_settings"]["max_bad_samples"]
        self.strict = algorithm_config["constrained_gmm_settings"]["strict"]
        self.precision = algorithm_config["constrained_gmm_settings"]["precision"]
        self.activate_parameter_check = algorithm_config["constrained_gmm_settings"]

    def sample_and_check_constraint(self,constraint, prev_frames, firstFrame=None, lastFrame=None):
        success = False
        s = self.sample()[0]
        aligned_frames  = find_aligned_quaternion_frames(self.mm_, s, prev_frames,
                                                     self.start_pose)
        ok,failed = check_constraint(aligned_frames, constraint,
                                     self.skeleton,
                                     start_pose=self.start_pose,
                                     precision=self.precision,
                                     constrain_first_frame=firstFrame,
                                     constrain_last_frame=lastFrame,
                                     verbose=self.verbose)
                 
        #assign the sample as either a good or a bad sample
        if len(ok)>0:               
            min_distance = min((zip(*ok))[1])
            success =True
        else:
            min_distance =  min((zip(*failed) )[1])

        return s,min_distance,success
        
    def set_constraint(self, constraint, prev_frames, firstFrame=None, lastFrame=None):
        """ Constrain the GMM with the given value

        Parameters
        ----------
        * constraint : tuple
        The constraint as (joint, [pos_x, pos_y, pos_z],
        [rot_x, rot_y, rot_z]) where unconstrained variables
        are set to None
        * prev_frames : numpy.ndarray
        \t euler frames of all previous steps
        * start_pose : dict
        \t contains start position and orientation. 
        \t Is needed if prev_frames is None
        * size : int
        \tThe number of samples we want to build the GMM with
        * precision : float
        \tThe precision of the sample to be rated as "constraint fullfiled"
        Raises
        ------
        exception : RuntimeError
           If a maximum  number of samples in a row are not successful
        """
        num = 0
        tmp_bad_samples = 1
        good_samples = []
        good_distances = []
        bad_samples = []
        bad_distances = []
        while len(good_samples) < self.n_random_samples:
            s,distance,success = self.sample_and_check_constraint(constraint, prev_frames, firstFrame, lastFrame)
            if success:               
                good_samples.append(s)
                good_distances.append(distance)
            else:
                bad_samples.append(s) 
                bad_distances.append(distance)
                tmp_bad_samples+=1
            if self.verbose:
                print "sample no",num,"min distance",distance
            num += 1
             
            if tmp_bad_samples>self.max_bad_samples:
                if not self.strict:
                    print "could not reach constraints use",self.n_random_samples,"best samples instead"
                    #merge good and bad samples
                    merged_samples = good_samples + bad_samples 
                    merged_distances = good_distances + bad_distances
                    #sample missing samples if necessary
                    while len(merged_samples) < self.n_random_samples:
                         s,distance,success = self.sample_and_check_constraint(constraint, prev_frames, firstFrame, lastFrame)
                         merged_samples.append(s)
                         merged_distances.append(distance)
                    #order them based on distance
                    sorted_samples = zip(merged_samples,merged_distances)
                    sorted_samples.sort(key=itemgetter(1))
                    #print type(sorted_samples)
                    good_samples = zip(*sorted_samples)[0][:self.n_random_samples]
                else:
                    #stop the conversion and output the motion up to the previous step
                    raise ConstraintError(bad_samples)
                break
            
        if self.verbose:
            print len(good_samples), " out of ", num
            print "Using %d samples out of %d" % (len(good_samples), num)


        good_samples = np.array(good_samples)
        self.fit(good_samples)



    
    
    
