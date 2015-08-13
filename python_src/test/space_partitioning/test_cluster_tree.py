# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:24:01 2015

@author: erhe01
"""

import sys
import os
import pytest
sys.path.append(os.sep.join([".."]*2))
import time
import numpy as np
from space_partitioning.cluster_tree import ClusterTree

TESTOUTPATH = os.sep.join([".."]*3 + ["test_output"])


def distance_objective(x, test_value):
    return np.linalg.norm(np.asarray(x)-np.asarray(test_value))

def run_cluster_hierarchy_construction(n_subdivisions, max_level, X, filename):
    cluster_tree = ClusterTree(n_subdivisions, max_level)
    cluster_tree.construct(X)
    #cluster_tree.save_to_file(path)
    cluster_tree.save_to_file_pickle(filename)
    return cluster_tree

class TestClusterTree(object):

    def setup_class(self):
        self.n_samples = 10000
        self.n_dim = 3
        self.n_subdivisions = 4
        self.max_level = 4
        self.error_margin = 0.1
        self.n_candidates = 5
        self.X_shape = (self.n_samples, self.n_dim)
        self.X = np.random.random(self.X_shape)
        self.test_value = self.X[0]
        #print self.X.shape
        self.filename = TESTOUTPATH + os.sep + "tree.pck"
        start = time.clock()
        self.cluster_tree = run_cluster_hierarchy_construction(self.n_subdivisions, self.max_level, self.X, self.filename)
        print "finished construction in ", time.clock()-start, "seconds"


    def test_search_from_file(self):
        print "from file"
        loaded_cluster_tree = ClusterTree(self.n_subdivisions, self.max_level)
        loaded_cluster_tree.load_from_file(TESTOUTPATH + os.sep + "tree")
        loaded_cluster_tree.load_from_file_pickle(self.filename)
        self.run_search(loaded_cluster_tree)

    def test_search_from_memory(self):
        print "from memory"
        self.run_search(self.cluster_tree)

    def run_search(self, cluster_tree):
        start = time.clock()
        approx_distance, approx_result = cluster_tree.find_best_example_excluding_search_candidates(distance_objective, self.test_value, self.n_candidates)
        distance, result = cluster_tree.find_best_example_exhaustive(distance_objective, self.test_value)
        print "best result for", self.test_value
        print "exhaustive:", result, distance
        print "approximate:", approx_result, approx_distance
        assert distance < self.error_margin
        print "finished search in ", time.clock()-start, "seconds"


if __name__ == "__main__":
    test_class = TestClusterTree()
    test_class.setup_class()
    test_class.test_search_from_memory()
    test_class.test_search_from_file()
