# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 19:56:30 2015

@author: mamauer,FARUPP,erhe01
"""
import time
from constrained_gmm import ConstrainedGMM
from gmm_math import mul

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
                                            constraints, prev_frames)   
        
        return gmm



    def _constrain_primitive(self, mp_node,constraint, prev_frames):
        """constrains a primitive with a given constraint
    
        Parameters
        ----------
        * mp_node : MotionPrimitiveNode
        \t\b
        * constraint : tuple
        \tof the shape (joint, [pos_x, pos_y, pos_z], [rot_x, rot_y, rot_z])
        * prev_frames : np.ndarray
        \t Used to estimate transformation of new samples 
        Returns
        -------
        * cgmm : ConstrainedGMM
        \tThe gmm of the motion_primitive constrained by the constraint
        """
        firstFrame = constraint['semanticAnnotation']['firstFrame']
        lastFrame = constraint['semanticAnnotation']['lastFrame']
        cgmm = ConstrainedGMM(mp_node, mp_node.motion_primitive.gmm, self.algorithm_config, 
                              self.start_pose, self.skeleton)
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
        * prev_frames : np.ndarray
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
            #print constraint
            #constraint = (c['joint'], c['position'], c['orientation'])
            cgmms.append(self._constrain_primitive(mp_node, constraint, prev_frames))
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
        * prev_frames : np.ndarray
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
            cgmm = self._create_constrained_gmm(mp_node, constraints, prev_frames)
            constrained_predict_gmm = mul(predict_gmm, cgmm)
            return mul(constrained_predict_gmm, mp_node.motion_primitive.gmm)
        else:
            return mul(predict_gmm, mp_node.motion_primitive.gmm)