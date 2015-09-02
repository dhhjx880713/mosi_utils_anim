# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:51:03 2015

@author: erhe01
"""
import json
import os
import numpy as np
from ..space_partitioning.cluster_tree import ClusterTree
from ..motion_model.motion_primitive import MotionPrimitive
MOTION_PRIMITIVE_FILE_ENDING = "mm.json"
CLUSTER_TREE_FILE_ENDING = "cluster_tree.pck"


class ClusterTreeBuilder(object):
    """ Creates ClusterTrees for all motion primitives by sampling from the statistical model
        The motion primitives are assumed to be organized in a directory 
        hierarchy as follows
        - model_data_root_dir
            - elementary_action_dir
                - motion_primitive_mm.json
    """
    def __init__(self, ):
        self.morphable_model_directory = None
        self.n_samples = 10000
        self.n_subdivisions_per_level = 4
        self.n_levels = 4
        self.random_seed = None
        self.only_spatial_parameters = True
        return
        
    def set_config(self, config_file_path):
        config_file = open(config_file_path)
        config = json.load(config_file)
        self.morphable_model_directory = config["model_data_dir"]
        self.n_samples = config["n_random_samples"]
        self.n_subdivisions_per_level = config["n_subdivisions_per_level"]
        self.n_levels = config["n_levels"]
        self.random_seed = config["random_seed"]
        self.only_spatial_parameters = config["only_spatial_parameters"]
        
    def _create_space_partitioning(self, motion_primitive, cluster_file_name):
        print "construct space partitioning data structure for", motion_primitive.name

        data = np.array([motion_primitive.sample_low_dimensional_vector() for i in xrange(self.n_samples)])
        if self.only_spatial_parameters:
            n_dims = motion_primitive.s_pca["n_components"]
            print "maximum dimension set to", n_dims, "ignoring time parameters"
        else:
            n_dims = len(data[0])
        cluster_tree = ClusterTree(self.n_subdivisions_per_level, self.n_levels, n_dims)
        cluster_tree.construct(data)
        #self.cluster_tree.save_to_file(cluster_file_name+"tree")
        cluster_tree.save_to_file_pickle(cluster_file_name + CLUSTER_TREE_FILE_ENDING)
        n_leafs = cluster_tree.root.get_number_of_leafs()
        print "number of leafs", n_leafs
       
    def _process_elementary_action(self, elementary_action):
        elementary_action_dir = self.morphable_model_directory + os.sep + elementary_action
        for root, dirs, files in os.walk(elementary_action_dir):
            for file_name in files:
                if file_name.endswith(MOTION_PRIMITIVE_FILE_ENDING):
                    motion_primitive = MotionPrimitive(elementary_action_dir + os.sep + file_name)
                    cluster_file_name = file_name[:-7]
                    self._create_space_partitioning(motion_primitive, elementary_action_dir + os.sep + cluster_file_name)
        return

    def build(self):

        if self.random_seed is not None:
            print "apply random seed", self.random_seed
            np.random.seed(self.random_seed)
        if self.morphable_model_directory is not None:
            print "start construction in directory", self.morphable_model_directory
            for elementary_action in next(os.walk(self.morphable_model_directory))[1]:
                self._process_elementary_action(elementary_action)
            return True
        else:
            return False
