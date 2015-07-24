# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 10:09:45 2015

@author: mamauer,FARUPP,erhe01
"""
import time
import numpy as np
import sklearn.mixture as mixture
from constraint.constraint_check import check_constraint, find_aligned_quaternion_frames
from operator import itemgetter
from statistics.gmm_math import mul

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
    * motion_primitve_node : MotionPrimitiveNode
        The original MotionPrimitive which will be constrained
    * constraint : tuple, optional
        The constraint as (joint, [pos_x, pos_y, pos_z], [rot_x, rot_y, rot_z])
        where unconstrained variables are set to None. The default is None,
        which means that no constraint is set.

    """
    def __init__(self,motion_primitve_node, gmm, algorithm_config, start_pose, skeleton, constraint=None, 
                 prev_frames=None):
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
        self.prev_frames = prev_frames
        self.mm_ = motion_primitve_node.motion_primitive
        self.weights_ = gmm.weights_
        self.means_ = gmm.means_
        self.converged_ = gmm.converged_
        self.covars_ = gmm.covars_
        self.samples_ = None
   
        self.sample_size = algorithm_config["constrained_gmm_settings"]["sample_size"]
        self.max_bad_samples = algorithm_config["constrained_gmm_settings"]["max_bad_samples"]
        self.strict = algorithm_config["constrained_gmm_settings"]["strict"]
        self.precision = algorithm_config["constrained_gmm_settings"]["precision"]
        self.activate_parameter_check = algorithm_config["constrained_gmm_settings"]
            
        if constraint is not None:
            self.set_constraint(constraint)

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
        
        while len(good_samples) < self.sample_size:
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
                    print "could not reach constraints use",self.sample_size,"best samples instead"
                    #merge good and bad samples
                    merged_samples = good_samples + bad_samples 
                    merged_distances = good_distances + bad_distances
                    #sample missing samples if necessary
                    while len(merged_samples) < self.sample_size:
                         s,distance,success = self.sample_and_check_constraint(constraint, prev_frames, firstFrame, lastFrame)
                         merged_samples.append(s)
                         merged_distances.append(distance)
                    #order them based on distance
                    sorted_samples = zip(merged_samples,merged_distances)
                    sorted_samples.sort(key=itemgetter(1))
                    #print type(sorted_samples)
                    good_samples = zip(*sorted_samples)[0][:self.sample_size]
                else:
                    #stop the conversion and output the motion up to the previous step
                    raise ConstraintError(bad_samples)
                break
            
        if self.verbose:
            print len(good_samples), " out of ", num
            print "Using %d samples out of %d" % (len(good_samples), num)


        good_samples = np.array(good_samples)
        self.fit(good_samples)


    
class ConstrainedGMMBuilder(object):
    def __init__(self, morphable_graph, algorithm_config, start_pose, skeleton):
        self._morphable_graph = morphable_graph
        self.algorithm_config = algorithm_config
        self.use_transition_model = algorithm_config["use_transition_model"]
        self.verbose = algorithm_config["verbose"]
        self.skeleton = skeleton
        self.start_pose = start_pose
        return
        

    def build(self, action_name, mp_name, constraints, prev_action_name=None, prev_mp_name=None, prev_frames=None, prev_parameters=None):
        """ Restrict the gmm to samples that roughly fit the constraints and 
            multiply with a predicted GMM from the transition model.
        """
     
         
        # Perform manipulation based on settings and the current state.
        if self.use_transition_model and prev_parameters is not None:
    
            transition_key = action_name +"_"+mp_name
            
            #only proceed the GMM prediction if the transition model was loaded
            if self._morphable_graph.subgraphs[prev_action_name].nodes[prev_mp_name].has_transition_model(transition_key):
                gpm = self._morphable_graph.subgraphs[prev_action_name].nodes[prev_mp_name].outgoing_edges[transition_key].transition_model 
                prev_primitve = self._morphable_graph.subgraphs[prev_action_name].nodes[prev_mp_name].motion_primitive
    
                gmm = self._create_next_motion_distribution(prev_parameters, prev_primitve,\
                                                    self._morphable_graph.subgraphs[action_name].nodes[mp_name],\
                                                    gpm, prev_frames,\
                                                    constraints)
    
        else:
            gmm = self._create_constrained_gmm(self._morphable_graph.subgraphs[action_name].nodes[mp_name],\
                                            constraints,\
                                            prev_frames)   
        
        return gmm



    def _constrain_primitive(self, mp_node,constraint, prev_frames,
                            firstFrame=None, lastFrame=None):
        """constrains a primitive with a given constraint
    
        Parameters
        ----------
        * mp_node : MotionPrimitiveNode
        \t\b
    
        * constraint : tuple
        \tof the shape (joint, [pos_x, pos_y, pos_z], [rot_x, rot_y, rot_z])
        * prev_frames : dict
        \t Used to estimate transformation of new samples 
        Returns
        -------
        * cgmm : ConstrainedGMM
        \tThe gmm of the motion_primitive constrained by the constraint
        """
    
        cgmm = ConstrainedGMM(mp_node,mp_node.motion_primitive.gmm, self.algorithm_config, self.start_pose, self.skeleton, constraint=None)
        cgmm.set_constraint(constraint, prev_frames, firstFrame=firstFrame,
                            lastFrame=lastFrame)
        return cgmm



    def _create_constrained_gmm(self, mp_node, constraints, prev_frames):
    
        """constrains a primitive with all given constraints and yields one gmm
        Parameters
        ----------
        * mp_node : MotionPrimitiveNode
        \t\b
    
        * constraints : list of tuples
        \tof the shape (joint, [pos_x, pos_y, pos_z], [rot_x, rot_y, rot_z])
        * prev_frames : dict
        \t Used to estimate transformation of new samples 
    
        Returns
        -------
        * cgmm : ConstrainedGMM
        \tThe gmm of the motion_primitive constrained by the constraints
        """
        if self.verbose:
            print "generating gmm using",len(constraints),"constraints"
            start = time.clock()
        cgmms = []
        
    
        for i, constraint in enumerate(constraints):
            print "\t checking constraint %d" % i
            print constraint
            #constraint = (c['joint'], c['position'], c['orientation'])
            firstFrame = constraint['semanticAnnotation']['firstFrame']
            lastFrame = constraint['semanticAnnotation']['lastFrame']
            cgmms.append(self._constrain_primitive(mp_node, constraint, prev_frames,
                                                         firstFrame=firstFrame,
                                                         lastFrame=lastFrame))
        cgmm = cgmms[0]
        for k in xrange(1, len(cgmms)):
            cgmm = mul(cgmm, cgmms[k])
        if self.verbose:
            print "generated gmm in ",time.clock()-start,"seconds"
        return cgmm
        
    
    
    def create_next_motion_distribution(self, prev_parameters, prev_primitive, mp_node,
                                        gpm, prev_frames, constraints=None):
        """ creates the motion following the first_motion fulfilling the given
        constraints and multiplied by the output_gmm
    
        Parameters
        ----------
        * first_motion : numpy.ndarray
        \tThe s-vector of the first motion
        * first_primitive : MotionPrimitive object
        \tThe first primitive
        * second_primitive : MotionPrimitive object
        \tThe second primitive
        * second_gmm : sklearn.mixture.gmm
        * constraints : list of numpy.dicts
        \tThe constraints for the second motion
        * prev_frames : dict
        \t Used to estimate transformation of new samples 
        * gpm : GPMixture object
        \tThe GPM from the transition model for the transition\
        first_primitive_to_second_primitive
    
        Returns
        -------
        * predict_gmm : sklearn.mixture.gmm
        \tThe predicted and constrained new gmm multiplied with the output gmm
    
        """
    
        predict_gmm = gpm.predict(prev_parameters)
        if constraints:
            cgmm = self._create_constrained_gmm(mp_node,constraints, prev_frames)
            constrained_predict_gmm = mul(predict_gmm, cgmm)
            return mul(constrained_predict_gmm, mp_node.motion_primitive.gmm)
        else:
            return mul(predict_gmm, mp_node.motion_primitive.gmm)
    
    
    
