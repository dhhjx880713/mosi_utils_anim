# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:51:03 2015

@author: erhe01
"""
import time
import json
import numpy as np
import sys
import os
sys.path.append("..")
from space_partitioning.cluster_tree import ClusterTree
from motion_model.motion_primitive import MotionPrimitive

MOTION_PRIMITIVE_FILE_ENDING = "mm.json"
CONIFG_FILE_PATH = ".." + os.sep + "config" + os.sep + "space_partitioning.json"

class ClusterTreeBuilder(object):
    def __init__(self, ):
        self.morphable_model_directory = None
        self.n_samples = 10000
        self.n_subdivisions_per_level = 4
        self.n_levels = 8
        
        return
        
    def set_config(self, config_file_path):
        config_file = open(config_file_path)
        config = json.load(config_file)
        self.morphable_model_directory = config["model_data_dir"]
        self.n_samples = config["n_random_samples"]
        self.n_subdivisions_per_level = config["n_subdivisions_per_level"]
        self.n_levels = config["n_levels"]
        
    def _create_space_partitioning(self, motion_primitive, cluster_file_name):
        print "construct space partitioning data structure for", motion_primitive.name

        X = np.array([motion_primitive.sample(return_lowdimvector=True) for i in xrange(self.n_samples)])
        cluster_tree = ClusterTree(self.n_subdivisions_per_level, self.n_levels)
        cluster_tree.construct(X)
        #self.cluster_tree.save_to_file(cluster_file_name+"tree")
        cluster_tree.save_to_file_pickle(cluster_file_name+"cluster_tree.pck")
       
    def _process_elementary_action(self, elementary_action):
        subgraph_path = self.morphable_model_directory+os.sep+elementary_action
        for root, dirs, files in os.walk(subgraph_path):
            for file_name in files:
                if file_name.endswith(MOTION_PRIMITIVE_FILE_ENDING):
                    motion_primitive = MotionPrimitive(subgraph_path + os.sep + file_name)
                    cluster_file_name =  file_name[:-7]
                    self._create_space_partitioning(motion_primitive, subgraph_path + os.sep + cluster_file_name)
        return
    def run(self):
        if self.morphable_model_directory is not None:
            for elementary_action in next(os.walk(self.morphable_model_directory))[1]:
                self._process_elementary_action(elementary_action)


def main():

    cluster_tree_builder = ClusterTreeBuilder()
    cluster_tree_builder.set_config(CONIFG_FILE_PATH)
    start = time.clock()
    cluster_tree_builder.run()
    print "Finished construction in", time.clock()-start, "seconds"
    return
                        
if __name__ == "__main__":
    main()