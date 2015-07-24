# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 19:24:10 2015

@author: erhe01
"""
import random
import numpy as np
from . import NODE_TYPE_START, NODE_TYPE_STANDARD, NODE_TYPE_END
from motion_primitive import MotionPrimitive
from animation_data.motion_editing import extract_root_positions_from_frames, get_arc_length_from_points
from math import sqrt
from space_partitioning.cluster_tree import ClusterTree

class MotionPrimitiveNode(object):
    """ Contains a motion primitive and all its outgoing transitions. 

    Parameters
    ----------
    * motion_primitive_filename: string
    \tThe filename with the saved data in json format.
    
    Attributes
    ----------
    * motion_primitive: MotionPrimitive
    \tThe motion primitive instance that is wrapped by the node class.
    * outgoing_edges: OrderedDict containing tuples
    \tEach entry contains a tuple (transition model, transition type)
    """
    def __init__(self):

        self.outgoing_edges = {}
        self.node_type = NODE_TYPE_STANDARD
        self.n_standard_transitions = 0
        self.parameter_bb = None
        self.cartesian_bb = None
        self.velocity_data = None
        self.cluster_annotation = None
        self.average_step_length = 0 
        self.action_name = None
        self.primitive_name = None
        self.motion_primitive = None
        self.cluster_tree = None 

        
    def init_from_file(self, action_name, primitive_name, motion_primitive_filename, node_type=NODE_TYPE_STANDARD):
                 
        self.outgoing_edges = {}
        self.node_type = node_type
        self.n_standard_transitions = 0
        self.parameter_bb = None
        self.cartesian_bb = None
        self.velocity_data = None
        self.cluster_annotation = None
        self.average_step_length = 0 
        self.action_name = action_name
        self.primitive_name = primitive_name
        self.motion_primitive = MotionPrimitive(motion_primitive_filename)
        
        self.cluster_tree = None
        cluster_file_name =  motion_primitive_filename[:-7]
        self._construct_space_partition(cluster_file_name)
        
    def init_from_dict(self,action_name, desc):
        self.action_name = action_name
        self.primitive_name = desc["name"]
        self.motion_primitive = MotionPrimitive(None)
        self.motion_primitive._initialize_from_json(desc["mm"])
        self.cluster_tree = desc["space_partition"]
        #print desc.keys()
        if "stats" in desc.keys():
            self.parameter_bb = desc["stats"]["pose_bb"]
            self.cartesian_bb = desc["stats"]["cartesian_bb"]
            self.velocity_data = desc["stats"]["pose_velocity"]
   
        
        return
        
    def _construct_space_partition(self, cluster_file_name, reconstruct=False):
        if not reconstruct and os.path.isfile(cluster_file_name+"cluster_tree.pck"):#os.path.isfile(cluster_file_name+"tree.data") and os.path.isfile(cluster_file_name+"tree.json"):
            print "load space partitioning data structure"
            self.cluster_tree = ClusterTree()#)
            #self.cluster_tree.load_from_file(cluster_file_name+"tree")
            self.cluster_tree.load_from_file_pickle(cluster_file_name+"cluster_tree.pck")
        else:
            print "construct space partitioning data structure"
            n_samples = 10000
            X = np.array([self.sample_parameters() for i in xrange(n_samples)])
            self.cluster_tree = ClusterTree()
            self.cluster_tree.construct(X)
            #self.cluster_tree.save_to_file(cluster_file_name+"tree")
            self.cluster_tree.save_to_file_pickle(cluster_file_name+"cluster_tree.pck")
                
    def search_best_sample(self,obj,data,n_candidates=2):
        """ Searches the best sample from a space partition data structure.
        Parameters
        ----------
        * obj : function
            Objective function returning a scalar of the form obj(x,data).
        * data : anything usually tuple
            Additional parameters for the objective function.
        * n_candidates: Integer
            Maximum number of candidates for each level when traversing the
            space partitioning data structure.
        
        Returns
         -------
         * parameters: numpy.ndarray
         \tLow dimensional motion parameters.
        """
        return self.cluster_tree.find_best_example_exluding_search_candidates_knn(obj, data, n_candidates)#_boundary
        
    def sample_parameters(self):
         """ Samples a low dimensional vector from statistical model.
         Returns
         -------
         * parameters: numpy.ndarray
         \tLow dimensional motion parameters.
        
         """
         return self.motion_primitive.sample(return_lowdimvector=True)
        
    def generate_random_transition(self, transition_type=NODE_TYPE_STANDARD):
        """ Returns the key of a random transition.

        Parameters
        ----------
        * transition_type: string
        \t Idententifies edges as either standard or end transitions
        """
        if self.outgoing_edges:
            edges = [e for e in self.outgoing_edges.keys() if self.outgoing_edges[e].transition_type == transition_type]
            if len(edges) > 0:
                random_index = random.randrange(0, len(edges), 1)
                to_key = edges[random_index]
                #print "to",to_key
                return to_key
        return None
        
    def generate_random_action_transition(self, elementary_action):
        """ Returns the key of a random transition to the given elementary action.

        Parameters
        ----------
        * elementary_action: string
        \t Identifies an elementary action
        """
        if self.outgoing_edges:
            edges = [e for e in self.outgoing_edges.keys() if e.split("_")[0] == elementary_action]
            if len(edges) > 0:
                random_index = random.randrange(0, len(edges), 1)
                to_key = edges[random_index]
                #print "to",to_key
                return to_key
        return None
        
    
    def update_attributes(self):
        """ Updates attributes for faster look up
        """
        self.n_standard_transitions = len([e for e in self.outgoing_edges.keys() if self.outgoing_edges[e].transition_type == NODE_TYPE_STANDARD])
        n_samples = 50 
        sample_lengths = [self._get_sample_step_length()for i in xrange(n_samples)]
        method = "median"
        if method == "average":
            self.average_step_length = sum(sample_lengths)/n_samples
        else:
            self.average_step_length = np.median(sample_lengths)
        
    def _get_sample_step_length(self, method="arc_length"):
        """Backproject the motion and get the step length and the last keyframe on the canonical timeline
        Parameters
        ----------
        * morphable_subgraph : MotionPrimitiveGraph
          Represents an elementary action
        * motion_primitive_name : string
          Identifier of the morphable model
        * method : string
          Can have values arc_length or distance. If any other value distance is used.
        Returns
        -------
        *step_length: float
        \tThe arc length of the path of the motion primitive
        """
        current_parameters =  self.sample_parameters()
        quat_frames = self.motion_primitive.back_project(current_parameters,use_time_parameters=False).get_motion_vector()
        if method == "arc_length":
            root_pos = extract_root_positions_from_frames(quat_frames)
            #print root_pos
            step_length = get_arc_length_from_points(root_pos)
        else:# use distance
            vector = quat_frames[-1][:3] - quat_frames[0][:3] 
            magnitude = 0
            for v in vector:
                magnitude += v**2
            step_length = sqrt(magnitude)
        return step_length
            
            
    def has_transition_model(self, to_key):
        return to_key in self.outgoing_edges.keys() and self.outgoing_edges[to_key].transition_model is not None
        
    def predict_parameters(self, to_key, current_parameters):
        """ Predicts parameters for a transition using the transition model.
        
        Parameters
        ----------
        * to_key: string
        \t Name of the action and motion primitive we want to transition to. 
        \t Should have the format "action_motionprimitive" 
        
        * current_parameters: numpy.ndarray
        \tLow dimensional motion parameters.
        
        Returns
        -------
        * next_parameters: numpy.ndarray
        \tThe predicted parameters for the next state.
        """
        gmm = self.outgoing_edges[to_key].transition_model.predict(current_parameters)
        next_parameters = np.ravel(gmm.sample())  
        return next_parameters
        
    def predict_gmm(self, to_key, current_parameters):
        """ Predicts a Gaussian Mixture Model for a transition using the transition model.
        
        Parameters
        ----------
        * to_key: string
        \t Name of the action and motion primitive we want to transition to. 
        \t Should have the format "action_motionprimitive" 
        
        * current_parameters: numpy.ndarray
        \tLow dimensional motion parameters.
        
        Returns
        -------
        * gmm: sklearn.mixture.GMM
        \tThe predicted Gaussian Mixture Model.
        """
        if self.outgoing_edges[to_key].transition_model is not None:
            return self.outgoing_edges[to_key].transition_model.predict(current_parameters)
        else:
            return self.motion_primitive.gmm
            


 