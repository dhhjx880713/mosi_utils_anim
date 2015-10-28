# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 19:24:10 2015

@author: erhe01
"""
import os
import random
import numpy as np
from . import NODE_TYPE_START, NODE_TYPE_STANDARD, NODE_TYPE_END
from motion_primitive import MotionPrimitive
from ..animation_data.motion_editing import extract_root_positions_from_frames, get_arc_length_from_points
from ..space_partitioning.cluster_tree import ClusterTree


class MotionState(MotionPrimitive):
    """ Contains a motion primitive and all its outgoing transitions. 

    Parameters
    ----------
    * motion_primitive_filename: string
    \tThe filename with the saved data in json format.
    
    Attributes
    ----------
    * outgoing_edges: OrderedDict containing tuples
    \tEach entry contains a tuple (transition model, transition type)
    """
    def __init__(self):
        super(MotionState, self).__init__(None)
        self.outgoing_edges = {}
        self.node_type = NODE_TYPE_STANDARD
        self.n_standard_transitions = 0
        self.parameter_bb = None
        self.cartesian_bb = None
        self.velocity_data = None
        self.average_step_length = 0 
        self.action_name = None
        self.primitive_name = None
        self.cluster_tree = None

    def init_from_file(self, action_name, primitive_name, motion_primitive_filename, node_type=NODE_TYPE_STANDARD):
        self.outgoing_edges = {}
        self.node_type = node_type
        self.n_standard_transitions = 0
        self.average_step_length = 0
        self.action_name = action_name
        self.primitive_name = primitive_name
        self._load(motion_primitive_filename)
        self.cluster_tree = None
        cluster_file_name = motion_primitive_filename[:-7]
        self._construct_space_partition(cluster_file_name)

    def init_from_dict(self, action_name, desc):
        self.action_name = action_name
        self.primitive_name = desc["name"]
        self._initialize_from_json(desc["mm"])
        self.cluster_tree = desc["space_partition"]
        if "stats" in desc.keys():
            self.parameter_bb = desc["stats"]["pose_bb"]
            self.cartesian_bb = desc["stats"]["cartesian_bb"]
            self.velocity_data = desc["stats"]["pose_velocity"]

    def _construct_space_partition(self, cluster_file_name, reconstruct=False):
        self.cluster_tree = None
        if not reconstruct and os.path.isfile(cluster_file_name+"cluster_tree.pck"):
            print "load space partitioning data structure"
            self.cluster_tree = ClusterTree()
            self.cluster_tree.load_from_file_pickle(cluster_file_name+"cluster_tree.pck")

    def search_best_sample(self, obj, data, n_candidates=2):
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
         * parameters: numpy.ndarray or None
             Low dimensional motion parameters. 
             None is returned if data structure was not initialized
        """
        if self.cluster_tree is not None:
            return self.cluster_tree.find_best_example_excluding_search_candidates(obj, data, n_candidates)
        else:
            return None

    def generate_random_transition(self, transition_type=NODE_TYPE_STANDARD):
        """ Returns the key of a random transition.

        Parameters
        ----------
        * transition_type: string
        \t Idententifies edges as either standard or end transitions
        """
        if self.outgoing_edges:
            edges = [edge_key for edge_key in self.outgoing_edges.keys()
                     if self.outgoing_edges[edge_key].transition_type == transition_type]
            if len(edges) > 0:
                random_index = random.randrange(0, len(edges), 1)
                to_node_key = edges[random_index]
                print "to", to_node_key, self.outgoing_edges[edges[random_index]].transition_type
                return to_node_key
        return None
        
    def generate_random_action_transition(self, elementary_action_name):
        """ Returns the key of a random transition to the given elementary action.

        Parameters
        ----------
        * elementary_action: string
        \t Identifies an elementary action
        """
        if self.outgoing_edges:
            edges = [edge_key for edge_key in self.outgoing_edges.keys()
                     if edge_key[0] == elementary_action_name]
            if len(edges) > 0:
                random_index = random.randrange(0, len(edges), 1)
                to_node_key = edges[random_index]
                print "to", to_node_key
                return to_node_key
        return None

    def update_attributes(self, n_samples=50, method="median"):
        """ Updates attributes for faster look up
        """
        self.n_standard_transitions = len([e for e in self.outgoing_edges.keys()
                                           if self.outgoing_edges[e].transition_type == NODE_TYPE_STANDARD])
        sample_lengths = [self._get_random_sample_step_length() for i in xrange(n_samples)]
        if method == "average":
            self.average_step_length = sum(sample_lengths)/n_samples
        else:
            self.average_step_length = np.median(sample_lengths)

    def _get_random_sample_step_length(self, method="arc_length"):
        """Backproject the motion and get the step length and the last keyframe on the canonical timeline
        Parameters
        ----------
        * method : string
          Can have values arc_length or distance. If any other value distance is used.
        Returns
        -------
        *step_length: float
        \tThe arc length of the path of the motion primitive
        """
        current_parameters = self.sample_low_dimensional_vector()
        return self.get_step_length_for_sample(current_parameters, method)

    def get_step_length_for_sample(self, parameters, method="arc_length"):
        """Backproject the motion and get the step length and the last keyframe on the canonical timeline
        Parameters
        ----------
        * parameters: np.ndarray
          Low dimensional motion parameters.
        * method : string
          Can have values arc_length or distance. If any other value distance is used.
        Returns
        -------
        *step_length: float
        \tThe arc length of the path of the motion primitive
        """
        # get quaternion frames from s_vector
        quat_frames = self.back_project(parameters, use_time_parameters=False).get_motion_vector()
        if method == "arc_length":
            root_pos = extract_root_positions_from_frames(quat_frames)        
            step_length = get_arc_length_from_points(root_pos)
        elif method == "distance":
            step_length = np.linalg.norm(quat_frames[-1][:3] - quat_frames[0][:3])
        else:
            raise NotImplementedError
        return step_length
            
    def has_transition_model(self, to_node_key):
        return to_node_key in self.outgoing_edges.keys() and self.outgoing_edges[to_node_key].transition_model is not None
        
    def predict_parameters(self, to_node_key, current_parameters):
        """ Predicts parameters for a transition using the transition model.
        
        Parameters
        ----------
        * to_node_key: tuple
        \t Identitfier of the action and motion primitive we want to transition to.
        \t Should have the format (action name, motionprimitive name)
        * current_parameters: numpy.ndarray
        \tLow dimensional motion parameters.
        
        Returns
        -------
        * next_parameters: numpy.ndarray
        \tThe predicted parameters for the next state.
        """
        gmm = self.outgoing_edges[to_node_key].transition_model.predict(current_parameters)
        next_parameters = np.ravel(gmm.sample())  
        return next_parameters

    def predict_gmm(self, to_node_key, current_parameters):
        """ Predicts a Gaussian Mixture Model for a transition using the transition model.
        Parameters
        ----------
        * to_key: tuple
        \t Identitfier of the action and motion primitive we want to transition to.
        \t Should have the format (action name, motionprimitive name)
        * current_parameters: numpy.ndarray
        \tLow dimensional motion parameters.
        Returns
        -------
        * gmm: sklearn.mixture.GMM
        \tThe predicted Gaussian Mixture Model.
        """
        if self.outgoing_edges[to_node_key].transition_model is not None:
            return self.outgoing_edges[to_node_key].transition_model.predict(current_parameters)
        else:
            return self.gaussian_mixture_model
            


 