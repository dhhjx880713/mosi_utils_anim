# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 10:09:45 2015

@author: mamauer,FARUPP,erhe01
"""

import numpy as np
import sklearn.mixture as mixture
from ...animation_data.motion_editing import align_quaternion_frames
from operator import itemgetter
from ...utilities.exceptions import ConstraintError

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

        self.mm_ = motion_primitve_node
        self.weights_ = gmm.weights_
        self.means_ = gmm.means_
        self.converged_ = gmm.converged_
        self.covars_ = gmm.covars_
    
        self.n_random_samples = algorithm_config["n_random_samples"]
        self.max_bad_samples = algorithm_config["constrained_gmm_settings"]["max_bad_samples"]
        self.strict = algorithm_config["constrained_gmm_settings"]["strict"]
        self.precision = algorithm_config["constrained_gmm_settings"]["precision"]
        self.activate_parameter_check = algorithm_config["constrained_gmm_settings"]

    def _check_constraint(self, sample, constraint, prev_frames):

        new_frames = self.mm_.back_project(sample, use_time_parameters=False).get_motion_vector()
        aligned_frames  = align_quaternion_frames(new_frames, prev_frames, self.start_pose)
        error, in_precision = constraint.evaluate_motion_sample_with_precision(aligned_frames)
        return error, in_precision
        
    def set_constraint(self, constraint, prev_frames):
        """ Constrain the GMM with the given value

        Parameters
        ----------
        * constraint : tuple
        The constraint as (joint, [pos_x, pos_y, pos_z],
        [rot_x, rot_y, rot_z]) where unconstrained variables
        are set to None
        * prev_frames : numpy.ndarray
        \t euler frames of all previous steps

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
            s = self.sample()[0]
            distance,success = self._check_constraint(s, constraint, prev_frames)
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
                         s = self.sample()[0]
                         distance,success = self._check_constraint(s, constraint, prev_frames)
                        
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

    
    
    
