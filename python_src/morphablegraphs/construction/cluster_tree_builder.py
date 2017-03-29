# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:51:03 2015

@author: erhe01
"""
import json
import os
import time
import heapq
import numpy as np
from ..space_partitioning.cluster_tree import ClusterTree
from ..motion_model.motion_primitive_wrapper import MotionPrimitiveModelWrapper
from ..animation_data.bvh import BVHReader
from ..animation_data.motion_editing import euler_to_quaternion

try:
    from mgrd import Skeleton as MGRDSkeleton
    from mgrd import SkeletonNode as MGRDSkeletonNode
    has_mgrd = True
except ImportError:
    has_mgrd = False
    pass

MOTION_PRIMITIVE_FILE_ENDING = "mm.json"
MOTION_PRIMITIVE_FILE_ENDING2 = "mm_with_semantic.json"
CLUSTER_TREE_FILE_ENDING = "_cluster_tree.pck"




class MGRDSkeletonBVHLoader(object):
    """ Load a Skeleton from a BVH file.

    Attributes:
        file (string): path to the bvh file
    """

    def __init__(self, file):
        self.file = file
        self.bvh = None

    def load(self):
        self.bvh = BVHReader(self.file)
        root = self.create_root()
        self.populate(root)
        return MGRDSkeleton(root)

    def create_root(self):
        return self.create_node(self.bvh.root, None)

    def create_node(self, name, parent):
        node_data = self.bvh.node_names[name]
        offset = node_data["offset"]
        if "channels" in node_data:
            angle_channels = ["Xrotation", "Yrotation", "Zrotation"]
            angles_for_all_frames = self.bvh.get_angles(*[(name, ch) for ch in angle_channels])
            orientation = euler_to_quaternion(angles_for_all_frames[0])
        else:
            orientation = euler_to_quaternion([0, 0, 0])
        return MGRDSkeletonNode(name, parent, offset, orientation)

    def populate(self, node):
        node_data = self.bvh.node_names[node.name]
        if "children" not in node_data:
            return
        for child in node_data["children"]:
            child_node = self.create_node(child, node)
            node.add_child(child_node)
            self.populate(child_node)



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
        self.mgrd_skeleton = None
        self.use_kd_tree = True
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
        self.store_indices = config["store_data_indices_in_nodes"]
        self.use_kd_tree = config["use_kd_tree"]

    def load_skeleton(self, skeleton_path):
        self.mgrd_skeleton = MGRDSkeletonBVHLoader(skeleton_path).load()

    def _get_samples_using_threshold(self, motion_primitive, threshold=0, max_iter_count=5):
        # data = np.array([motion_primitive.sample_low_dimensional_vector() for i in xrange(self.n_samples)])
        data = []
        count = 0
        iter_count = 0

        while count < self.n_samples and iter_count < max_iter_count:
            samples = motion_primitive.sample_low_dimensional_vectors(self.n_samples)
            for new_sample in samples:
                likelihood = motion_primitive.get_gaussian_mixture_model().score([new_sample,])[0]
                if likelihood > threshold:
                    data.append(new_sample)
                    count += 1
            iter_count += 1
        if iter_count < max_iter_count:
            return np.asarray(data)
        else:
            return None

    def _get_best_samples(self, motion_primitive):
        # data = np.array([motion_primitive.sample_low_dimensional_vector() for i in xrange(self.n_samples)])
        likelihoods = []
        data = motion_primitive.sample_low_dimensional_vectors(self.n_samples*2)
        for idx, sample in enumerate(data):
            likelihood = motion_primitive.get_gaussian_mixture_model().score([sample,])[0]
            #print i, likelihood, new_sample
            heapq.heappush(likelihoods, (-likelihood, idx))

        l, indices = zip(*likelihoods[:self.n_samples])
        data = np.asarray(data)
        data = data[list(indices)]
        np.random.shuffle(data)
        return data

    def _create_space_partitioning(self, motion_primitive_file_name, cluster_file_name):
        if os.path.isfile(cluster_file_name + CLUSTER_TREE_FILE_ENDING):
            print "Space partitioning data structure", cluster_file_name, "already exists"
        elif os.path.isfile(motion_primitive_file_name):
            print "construct space partitioning data structure", cluster_file_name

            motion_primitive = MotionPrimitiveModelWrapper()
            motion_primitive._load_from_file(self.mgrd_skeleton, motion_primitive_file_name)
            print "Create",self.n_samples,"good samples"
            data = self._get_samples_using_threshold(motion_primitive)
            if data is None:
                data = self._get_best_samples(motion_primitive)
            if self.only_spatial_parameters:
                n_dims = motion_primitive.get_n_spatial_components()
                print "maximum dimension set to", n_dims, "ignoring time parameters"
            else:
                n_dims = len(data[0])
            cluster_tree = ClusterTree(self.n_subdivisions_per_level, self.n_levels, n_dims, self.store_indices, self.use_kd_tree)
            cluster_tree.construct(data)
            # self.cluster_tree.save_to_file(cluster_file_name+"tree")
            cluster_tree.save_to_file_pickle(cluster_file_name + CLUSTER_TREE_FILE_ENDING)
            n_leafs = cluster_tree.root.get_number_of_leafs()
            print "number of leafs", n_leafs
        else:
            print "Could not read motion primitive", motion_primitive_file_name
       
    def _process_elementary_action(self, elementary_action):
        elementary_action_dir = self.morphable_model_directory + os.sep + elementary_action
        for root, dirs, files in os.walk(elementary_action_dir):
            for file_name in files:
                if file_name.endswith(MOTION_PRIMITIVE_FILE_ENDING) or file_name.endswith(MOTION_PRIMITIVE_FILE_ENDING2):
                    try:
                        print elementary_action_dir + os.sep + file_name
                        motion_primitive_filename = elementary_action_dir + os.sep + file_name
                        index = file_name.find("_mm")
                        cluster_file_name = file_name[:index]
                        self._create_space_partitioning(motion_primitive_filename, elementary_action_dir + os.sep + cluster_file_name)
                    except Exception as e:
                        print "Exception during loading of file",file_name
                        print e.message
                        continue

    def build(self):
        #self._process_elementary_action("elementary_action_carryRight")
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

    def build_for_one_motion_primitive(self, motion_primitive_file, space_partition_file):
        self._create_space_partitioning(motion_primitive_file, space_partition_file)