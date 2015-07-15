# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 10:09:45 2015

@author: mamauer,FARUPP,erhe01
"""

import os
import numpy as np
import sklearn.mixture as mixture
from constraint.constraint_check import check_constraint, find_aligned_quaternion_frames
from motion_model.motion_primitive import MotionPrimitive
from operator import itemgetter


def _get_mm_input_file(action, primitive):
    """ Return the path to the inputfile """
    ROOT = os.sep.join(['..'] * 3)
    return os.sep.join([ROOT, 'data', '3 - Motion primitives',
                        'motion_primitives_quaternion_PCA95',
                        'elementary_action_%s' % action,
                        '%s_%s_quaternion_mm.json' % (action, primitive)])

class ConstraintError(Exception):
    def __init__(self,  bad_samples):
        message = "Could not reach constraint"
        super(ConstraintError, self).__init__(message)
        self.bad_samples = bad_samples


class ConstrainedGMM(mixture.GMM):
    """ A GMM that has the ability to constraint itself. The GMM is build based
    on a GMM

    Parameters
    ----------
    * mm : MotionPrimitive
        The original MotionPrimitive which will be constrained
    * constraint : tuple, optional
        The constraint as (joint, [pos_x, pos_y, pos_z], [rot_x, rot_y, rot_z])
        where unconstrained variables are set to None. The default is None,
        which means that no constraint is set.

    """
    def __init__(self,graph_node, gmm, constraint=None, bvh_reader=None, node_name_map=None, 
                 prev_frames=None, settings=None, verbose=False):
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
        self.graph_node = graph_node
        self.verbose = verbose
        self.bvh_reader = bvh_reader
        self.prev_frames = prev_frames
        self.node_name_map = node_name_map
        self.mm_ = graph_node.mp
        self.weights_ = gmm.weights_
        self.means_ = gmm.means_
        self.converged_ = gmm.converged_
        self.covars_ = gmm.covars_
        self.samples_ = None
        if settings is not None:
            
            self.max_bad_samples = settings["max_bad_samples"]
            self.strict = settings["strict"]
            self.precision = settings["precision"]
        else:
            self.max_bad_samples = 200
            self.strict = False
            self.precision = {"pos":1,"rot":1,"smooth":1}
            
        if constraint is not None:
            self.set_constraint(constraint)

    def sample_and_check_constraint(self,constraint, prev_frames, start_pose,size=100,
                       precision={"pos":1,"rot":1}, firstFrame=None, lastFrame=None,
                        activate_parameter_check =True):
        success = False
        s = self.sample()[0]
        aligned_frames  = find_aligned_quaternion_frames(self.mm_, s, prev_frames,
                                                     start_pose, self.bvh_reader,
                                                     self.node_name_map)
        ok,failed = check_constraint(aligned_frames, constraint,
                                     self.bvh_reader,
                                     node_name_map=self.node_name_map,
                                     start_pose=start_pose,
                                     precision=self.precision,
                                     firstFrame=firstFrame,
                                     lastFrame=lastFrame,
                                     verbose=self.verbose)
                 
        #assign the sample as either a good or a bad sample
        if len(ok)>0:               
            min_distance = min((zip(*ok))[1])
            success =True
        else:
            min_distance =  min((zip(*failed) )[1])

        return s,min_distance,success
        
    def set_constraint(self, constraint, prev_frames, start_pose,size=100,
                       precision={"pos":1,"rot":1,"smooth":1}, firstFrame=None, lastFrame=None, activate_parameter_check=True):
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
        
        while len(good_samples) < size:
            s,distance,success = self.sample_and_check_constraint(constraint, prev_frames, start_pose,size,
                                                       precision, firstFrame, lastFrame,
                                                        activate_parameter_check)
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
                    print "could not reach constraints use",size,"best samples instead"
                    #merge good and bad samples
                    merged_samples = good_samples + bad_samples 
                    merged_distances = good_distances + bad_distances
                    #sample missing samples if necessary
                    while len(merged_samples) < size:
                         s,distance,success = self.sample_and_check_constraint(constraint, prev_frames, start_pose,size,
                                                                           precision, firstFrame, lastFrame,
                                                                            activate_parameter_check)
                         merged_samples.append(s)
                         merged_distances.append(distance)
                    #order them based on distance
                    sorted_samples = zip(merged_samples,merged_distances)
                    sorted_samples.sort(key=itemgetter(1))
                    #print type(sorted_samples)
                    good_samples = zip(*sorted_samples)[0][:size]
                else:
                    #stop the conversion and output the motion up to the previous step
                    raise ConstraintError(bad_samples)
                break
            
        if self.verbose:
            print len(good_samples), " out of ", num
            print "Using %d samples out of %d" % (len(good_samples), num)


        good_samples = np.array(good_samples)
        self.fit(good_samples)


def main():
    action = 'walk'
    primitive = 'beginRightStance'
    mmfile = _get_mm_input_file(action, primitive)
    mm = MotionPrimitive(mmfile)

    cgmm = ConstrainedGMM(mm, mm.gmm)

    constraint = ('Hips', [12, None, -20], [None, None, None])
    #  transformation = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0],
    #                             [0, 0, 0, 1]])
    transformation = None
    cgmm.set_constraint(constraint, transformation, start_pose =np.array([0,0,0]), size=30,
                        precision=3, firstFrame=None, lastFrame=None)

    new_s = cgmm.sample(10)
    for i, s in enumerate(new_s):
        ok = check_constraint(mm, s, constraint, prev_frames=None, start_pose=None,
                              precision=3, firstFrame=None, lastFrame=None)
        print "sample %s ok?" % i, ok

def test():
    action = 'pick'
    primitive = 'first'
    mmfile = _get_mm_input_file(action, primitive)
    mm = MotionPrimitive(mmfile)
    cgmm = ConstrainedGMM(mm, mm.gmm)
    constraint = ('LeftHand', [-46.822415923765426, 23.191993354140852, -45.92779134992672])
    constraint = ('RightHand', [59.39961171964804, 21.092615703131738, -38.04536998280619])


if __name__ == '__main__':
    main()
