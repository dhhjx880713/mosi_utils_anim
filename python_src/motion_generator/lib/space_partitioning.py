# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:58:08 2015

@author: erhe01 
"""
import numpy as np
from operator import itemgetter
import time
import heapq
from sklearn import cluster
import uuid
import json
import cPickle as pickle


def discrete_sample(values,probabilities):
    """ Returns a sample from a discrete probability distribution
        Sources: 
        http://dept.stat.lsa.umich.edu/~jasoneg/Stat406/lab5.pdf
        http://stackoverflow.com/questions/11373192/generating-discrete-random-variables-with-specified-weights-using-scipy-or-numpy
    """
    u = np.random.uniform(0, 1)
    bins = np.add.accumulate(np.asarray(probabilities))
    index = np.digitize([u, ], bins)[0]
    return values[index]


def uniquifyList(seq, key=lambda x: x):
    """ Removes duplicate elements preserving the order
        http://www.peterbe.com/plog/uniqifiers-benchmark
    """

    seen = {}
    result = []
    for item in seq:
        marker = tuple(key(item))
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)
    return result


class Node(object):
    """
    Node for KDTree modified from
    https://en.wikipedia.org/wiki/K-d_tree
    """
    def __init__(self, data, dim, depth=0):
        self.index = depth
        if len(data)>1:
            self.type = "inner" 
     
           # Select axis based on depth so that axis cycles through all valid values   
            axis = depth % dim
  
            # Sort point list and choose median as pivot element
            data.sort(key=itemgetter(axis))
            median = len(data) // 2 # choose median
            left_data = data[:median]
            right_data = data[median+1:]
            
            # Create node and construct subtrees
            self.point = data[median]

            if len(left_data) > 0:
                self.left = Node(left_data,dim,depth+1)
            else:
                self.left = None
                
            if len(right_data) > 0:
                self.right = Node(right_data,dim,depth+1)
            else:
                self.right = None
        else:
            self.type = "leaf"
            self.point =  data[0]
            self.left = None
            self.right = None
       

class KDTree(object):
    """
    K-Dimensional space partitioning data structure modified from
    https://en.wikipedia.org/wiki/K-d_tree with additional search functions 
    based on an objective function.
    """
    def __init__(self):
        self.data = None
        self.root = None
        self.global_bb = None
        return
              
    def construct(self, data,dim):
        self.data = data
        self.root = Node(data,dim)
        return

    def _decide_direction_distance(self, target, left, right, distance):
        """ Chooses between left and right node allowing either to be None.
        Parameters
        ---------
        * left : Node or None
            First option.
        * right : Node or None
            Second option.
        * obj : function
            objective function returning a scalar of the form distance(x).

        Returns
        -------
        tuple containing
        * best_option: Node
            Either left or right.
        * least_cost: float
            distance for the best option.
        """
        best_option = None
        least_distance = 100000
        if left is not None and right is not None:
            r_d = distance(target, right.point)
            l_d = distance(target, left.point)
            if l_d < r_d:
                best_option = left
                least_distance = l_d
            else:
                best_option = right
                least_distance = r_d
        elif right is not None:
            best_option = right
            least_distance = distance(target, right.point)
        elif left is not None:
            best_option = left
            least_distance = distance(target, left.point)
    #         if least_distance <= bound:
        return best_option, least_distance
    #         else:
    #             return None, 100000


    def _decide_direction_objective(self, left, right, obj, data):
        """ Chooses between left and right node allowing either to be None.
        Parameters
        ---------
        * left : Node or None
            First option.
        * right : Node or None
            Second option.
        * obj : function
            objective function returning a scalar of the form obj(x,data).
        * data : anything usually tuple
            additional parameters for the objective function.

        Returns
        -------
        * best_option: Node
            Either left or right.
        * least_cost: float
            result of obj for the best option.
        """
        best_option = None
        least_cost = 100000
        if left is not None and right is not None:
            r_d = obj(right.point, data)
            l_d = obj(left.point, data)
            if l_d < r_d:
                best_option = left
                least_cost = l_d
            else:
                best_option = right
                least_cost = r_d
        elif right is not None:
            best_option = right
            least_cost = obj(right.point, data)
        elif left is not None:
            best_option = left
            least_cost = obj(left.point, data)
        return best_option, least_cost

    def query(self, target, K=1):
        """Returns the K nearest neighbors using a Depth First search.
            distances,
            node
        """
        def point_distance(a, b):
            return np.linalg.norm(a-b)
        assert K >= 1
        #bound = 10000
        result_queue = [] 
        node_stack = []
        heapq.heappush(result_queue,(point_distance(self.root.point,target),self.root.point))
        node_stack.append(self.root)
        while len(node_stack) > 0:
            node = node_stack.pop(-1)
            best_option, least_distance = self._decide_direction_distance(target,node.left,node.right)
            if best_option is not None:
                heapq.heappush(result_queue,(least_distance,best_option)) 
#                if len(result_queue)>=K:
#                    bound = result_queue[K][0]
        return zip([(n[0], n[1]) for n in result_queue[:K]])

        
    def print_tree_df(self):
        """prints tree using a depth first traversal 
        """
        #depth = 0
        node_stack = []
        #heapq.heappush(node_stack,[d,i])
        node = self.root
        node_stack.append(self.root)
        element_count = 1
        while len(node_stack) > 0:
            node = node_stack.pop(-1)
            print element_count, node.index, node.point
            if node.type == "inner":
                if node.left != None:
                    node_stack.append(node.left)
                if node.right != None:
                    node_stack.append(node.right)
            element_count += 1
        
        
    def df_search(self,obj,data):
        """Depth first search to find the best of all samples
        """
        #depth = 0
        node_stack = []
        result_queue = [] 
      
        node = self.root
    
        heapq.heappush(result_queue,(obj(node.point,data),node.point))
        node_stack.append(self.root)
        element_count = 1
        while len(node_stack) > 0:
            node = node_stack.pop(-1)
            heapq.heappush(result_queue,(obj(node.point,data),node.point) )
            print element_count# node.point
            if node.type == "inner":
                if node.left != None:
                    node_stack.append(node.left)
                if node.right != None:
                    node_stack.append(node.right)
            element_count += 1
        if len(result_queue) > 0:
            return heapq.heappop(result_queue)
        else:
            return None
    
    def find_best_example(self,obj,data, k=1):
        """
        Traverses the KDTree using the direction of least cost until a leaf is reached
        """
        eval_points = []
        result_queue = []
        node = self.root
        depth = 0
        eval_points.append(node.point)
        heapq.heappush(result_queue,(obj(node.point,data),depth) )
   
        while node is not None and node.type  == "inner":
            #print "depth",depth,node.point#node.right,node.left
            
            depth += 1
            
            node, least_cost = self._decide_direction_objective(node.left,node.right,obj,data)
            if node is not None:
                heapq.heappush(result_queue,(least_cost,depth)) 
                eval_points.append(node.point)

#        best_value,best_index = heapq.heappop(result_queue)
#        return best_value,eval_points[best_index]# best_point#node.point
        return [ (value,eval_points[index]) for  value, index in result_queue[:k]]
        
        
    
class KDTreeWrapper(object):
    """ Wrapper for a KDTree used by the ClusterTree class
    
    Parameters
    ---------
    * dim: Integer
        Number of dimensions of the data.
    """
    def __init__(self,dim):
        self.id = str(uuid.uuid1())
        self.kdtree = KDTree()
        self.dim = dim
        self.indices = []
        self.type = "kdtree"
       
        
    def construct(self, X, indices):
        self.indices = indices
        self.kdtree.construct(X[indices].tolist(),self.dim)
        return
        
    def find_best_example(self,obj,data):
        return self.kdtree.find_best_example(obj,data,1)[0]

    def knn_interpolation(self, obj, data, k=50):
        
        results = self.kdtree.find_best_example(obj, data, k)#50
        if len(results)>1:#K
            distances, points = zip(*results)
            #distances, points = self.kdtree.query(target, )
        
            influences = []
            furthest_distance = distances[-1]
            #print furthest_distance,"#################"
            for d in distances[:-1]:
                 influences.append(1/d - 1/furthest_distance)
            ## calculate weight based on normalized influence
            weights = []
            n_influences = len(influences)
            sum_influence = np.sum(influences)
            for i in xrange(n_influences):
               weights.append(influences[i]/sum_influence)
    
            new_point = np.zeros(len(points[0]))
            for i in xrange(n_influences):
    
                #print index
                new_point += weights[i] * np.array(points[i])
            #print new_point,"#####"
            return obj(new_point,data), new_point # return also the evaluation of the new point
        else:
            return results[0]
        
    def get_desc(self):
        """
        used by save_to_file
        """
        node_desc = {}
        node_desc["id"] = str(self.id)
        #node_desc["depth"] = self.depth
        node_desc["type"] = self.type
     
        node_desc["children"] = []
        #node_desc["mean"] = self.mean.tolist() 
        
        node_desc["indices"] = self.indices
        return node_desc
        

class ClusterTreeNode(object):
    """ Node for the ClusterTree class. Subdivides samples using KMeans and
        creates a child node for each subdivision. Child nodes can be ClusterTreeNodes
        or KDTreeNodes depending on the current depth and the maximum depth.
        Stores the indices refering to the samples stored in ClusterTree.

    Parameters
    ---------
    * N: Integer
        Number of subdivisions.
    * K: Integer
        Maximum number of levels.
    * dim: Integer
        Number of dimensions of the data.
    """
    def __init__(self,N,K,dim):
        self.id = str(uuid.uuid1())
        self.clusters = []
        self.N = N
        self.K = K
        self.dim = dim
        #self.n_samples = []
        #self.samples = []
        self.indices = []
#        self.probabilities = []
        #self.labels = []
        #self.means = []
        self.mean = None
        self.kmeans = None
        self.leaf = False
        self.type = "None"
        self.depth = -1
        
   
    def create_subdivision(self, X, indices=None, depth=0):
        """ Creates a divides sample space into partitions using KMeans and creates
             a child for each space partition.
             
        Parameters
        ----------
        * X: np.ndarray
            2D array of samples
        * indices: list
            indices of X that should be considered for the subdivision
        * depth: int
            current depth used with self.K to decide the type of node and the type of subdivisions
        """
        ##decide on type
        self.depth = depth
        if depth < self.K-1:
            if depth == 0:
                self.type = "root"
            else:
                self.type = "inner"
        else:
            self.type = "leaf"

        #self.samples = X#X has always at least 1 sample
        self.indices = indices
        if  indices is None:
            n_samples = len(X)
            self.mean = np.mean(X,axis=0)
        else:
            n_samples = len(indices)
            self.mean = np.mean(X[indices],axis=0)
        if n_samples > self.N:#number of samples at least equal to the number of clusters required for kmeans
            ## create subdivision
            self.kmeans = cluster.KMeans(n_clusters=self.N)
            if indices is None:
                labels = self.kmeans.fit_predict(X)
            else:
                labels = self.kmeans.fit_predict(X[indices] )
            cluster_indices = [[] for i in xrange(self.N)] 
            if indices is None:
                for i in xrange(n_samples):
                    l = labels[i]
                    cluster_indices[l].append(i)#
            else:
                for i in xrange(n_samples):
                    l = labels[i]
                    original_index = indices[i]
                    cluster_indices[l].append(original_index)#self.samples[i]
            
            
            if depth < self.K:
                ## create inner node for each cluster
                 self.leaf = False
                 for j in xrange(len(cluster_indices) ):
#                    if len(cluster_data[j]) > 0: #ignore clusters that are empty
                
                    if len(cluster_indices[j])>0:
                        #node_id = #get_global_id()#depth *10 #+ self.id+=1
                        #node_id +=1
                        #print node_id
                        node = ClusterTreeNode(self.N,self.K,self.dim)
                        node.create_subdivision(X,cluster_indices[j],depth+1)
                        self.clusters.append(node)
                        
#                        n_samples = len(cluster_indices[j])
#                        self.n_samples.append(n_samples)
#                        self.probabilities.append([1/float(n_samples)]*n_samples)
#                        self.indices.append(range(n_samples))
#                        self.means.append(np.mean(X[cluster_indices[j]],axis=0))
            else:
                ## create kdtree for each cluster
                self.cluster_indices = cluster_indices
                self.leaf = True
                for j in xrange(len(cluster_indices) ):
#                    if len(cluster_data[j]) > 0: #ignore clusters that are empty
                    if len(cluster_indices[j])>0:
                        node = KDTreeWrapper(self.dim)
                        node.construct(X,cluster_indices[j])#[].tolist()
                        self.clusters.append(node)
#                      
#                        n_samples = len(cluster_indices[j])
#                        self.n_samples.append(n_samples)
#                        self.probabilities.append([1/float(n_samples)]*n_samples)
#                        self.indices.append(range(n_samples))
#                        self.means.append(np.mean(X[cluster_indices[j]],axis=0))
        else:
            #not enough samples to further divide it
            #so stop before reaching level K
            self.cluster_indices = [indices]
            self.leaf = True
            kdtree = KDTreeWrapper(self.dim)
            kdtree.construct(X,indices)
            self.clusters.append(kdtree)
#            n_samples = len(indices)
#            self.n_samples.append(n_samples)
#            self.probabilities.append([1/float(n_samples)]*n_samples)
#            self.indices.append(range(n_samples))
#            self.means.append(np.mean(X[indices],axis=0))
            
#            self. = cluster_data
            
                
#        return labels, clusters
                
    def find_best_example_knn(self, obj, data, k=50):
        """Return the best example based on the evaluation using an objective function.
            Interpolates the best k results.
        """
        if self.leaf:
            result_queue = []
            for i in xrange(len(self.clusters)):#kdtree

#                try:
                result = self.clusters[i].knn_interpolation(obj, data, k)
                #print result
                heapq.heappush(result_queue,result)
                
            return heapq.heappop(result_queue)
        return
        
    def find_best_example(self, obj, data):   
        """Return the best example based on the evaluation using an objective function.
        """
        if self.leaf:
            result_queue = []
            for i in xrange(len(self.clusters)):

#                try:
                result = self.clusters[i].find_best_example(obj,data)
                #print result
                heapq.heappush(result_queue,result)
                    
#                except ValueError as e:
#                 
#                    print e.message,len(self.clusters[i].data)
#                    return (10000,kdtree.root.point)#np.inf
#                    return [ obj(node.point,data),node.point#node.point
                 
            return heapq.heappop(result_queue)
        else:
            best_value = np.inf
            best_index = 0
            for cluster_index in xrange(len(self.clusters)):
#                len_samples = len(self.clusters[cluster_index].samples)
#                indices = range(len_samples)
#                random_sample_index = discrete_sample(self.indices,self.probabilities)
#                sample = self.clusters[cluster_index].samples[random_sample_index]
                sample = self.means[cluster_index]
                cluster_value = obj(sample,data)
                if cluster_value < best_value:
                    best_index = cluster_index
                    best_value = cluster_value
            return self.clusters[best_index].find_best_example(obj,data)
            
    def find_best_cluster(self, obj, data, use_mean=False):   
        """ Return the best cluster based on the evaluation using an objective function.

        Parameters
        ----------
        * obj: function
            Objective function of the form: scalar = obj(x,data).
        * data: Tuple
            Additional parameters for the objective function.
        * n_candidates: Integer
            Maximum number of candidates
        """
 
        best_value = np.inf
        best_index = 0
        for cluster_index in xrange(len(self.clusters) ):
#                len_samples = len(self.clusters[cluster_index].samples)
#                indices = range(len_samples)
            #if use_mean:
            sample = self.clusters[cluster_index].mean#self.means[cluster_index]#.cluster
#            else:
#                print cluster_index
#                random_sample_index = discrete_sample(self.indices[cluster_index],self.probabilities[cluster_index])
#                sample = self.clusters[cluster_index].samples[random_sample_index]
            cluster_value = obj(sample,data)
            
            if cluster_value < best_value:
                best_index = cluster_index
                best_value = cluster_value
        return best_index,best_value
        


        
    def find_best_cluster_canditates(self,obj,data,n_candidates):
        """Return the clusters with the least cost based on 
        an evaluation using an objective
        function.
        
        Parameters
        ----------
        * obj: function
            Objective function of the form: scalar = obj(x,data).
        * data: Tuple
            Additional parameters for the objective function.
        * n_candidates: Integer
            Maximum number of candidates
            
        Returns
        -------
        * best_candidates: list of (value, ClusterTreeNode) tuples.
            List of candidates ordered using the objective function value.
        """
        result_queue = []
        for cluster_index in xrange(len(self.clusters)):

            sample = self.clusters[cluster_index].mean# self.means[cluster_index]#.cluster

            cluster_value = obj(sample,data)
            heapq.heappush(result_queue,(cluster_value,self.clusters[cluster_index] ) )
      
        return result_queue[:n_candidates]
    
    def get_desc(self):
        """Used to save the node to file.
        Returns
        ------
        * node_desc: dict
            Dictionary containing the properties of the node.
        """
        node_desc = {}
        node_desc["id"] = str(self.id)
        node_desc["depth"] = self.depth
        node_desc["type"] = self.type
#        if self.leaf:
#            node_desc["type"] = "leaf"#means it has a list of kdtrees
#        else:
#            node_desc["type"] = "inner"# means it has a list of ClusterTreeNodes
        children = []
#        if not self.leaf:
        for node in self.clusters:
            children.append(str(node.id))
        node_desc["children"] = children
        node_desc["mean"] = self.mean.tolist() 
        
        if self.depth == 0:
            node_desc["indices"] = "all"
        else:
            if self.indices is not None:
                node_desc["indices"] = self.indices
            else:
                node_desc["indices"] = []
        
        return node_desc
        
               
    def construct_from_node_desc_list(self,node_id,node_desc_dict,X):
        """Recursively rebuilds the cluster tree given a dictionary containing 
           a description of all nodes and the data samples.
        
        Parameters
        ---------
        * node_id: String
            Unique identifier of the node.
        * node_desc: dict
            Dictionary containing the properties of each node. The node id is used as key.
        * X: np.ndarray
            Data samples.
        """
        desc = node_desc_dict["nodes"][node_id]
        self.id = node_id
        self.type = desc["type"]
        self.clusters = []
        self.mean = np.array(desc["mean"])
        if self.type != "root":
            self.indices = desc["indices"]
        else:
            self.indices = []
            
        if self.type != "leaf":
            self.leaf = False
            for c_id in desc["children"]:
                node = ClusterTreeNode(self.N,self.K,self.dim)
                node.construct_from_node_desc_list(c_id,node_desc_dict,X)
                self.clusters.append(node)
        else:
            self.leaf = True
            for c_id in desc["children"]:
                kdtree_node = KDTreeWrapper(self.dim)
  
                kdtree_node.id = c_id
                indices = node_desc_dict["nodes"][c_id]["indices"]
                kdtree_node.construct(X,indices)
                self.clusters.append(kdtree_node)
            
        return
        
        
    def get_node_desc_list(self):
        """Iteratively builds a dictionary containing a description of all 
           nodes.
        Returns:
        -------
        * node_desc_dict: dict
            Contains the result of ClusterTreeNode.get_desc() for all nodes.
        """
        node_desc_dict = {}
        stack = [self]
        node_desc_dict["root"] = str(self.id)
        node_desc_dict["nodes"] = {}
        while len(stack) > 0:
              node = stack.pop(-1)
              node_desc = node.get_desc()
              node_desc_dict["nodes"][node_desc["id"]] = node_desc
              #node_desc_list.append()
              if node.type != "kdtree":
                  for cluster in node.clusters:
                      stack.append(cluster)
        return node_desc_dict
 
        
class ClusterTree(object):
    """
    Create a hiearchy of clusters using KMeans and then use a kdtree for the leafs
    #TODO make faster
    Parameters
    -----------
    * N : Integer
        Number of subclusters/children per node in the tree. At least 2.
    * K : Integer
        Maximum levels in the tree. At least 1.
    
    """
    def __init__(self, N=4,K=4):
        self.N = max(N,2)
        self.K = max(K,1)
        self.root = None
        self.X = None
  
        return
      
    def save_to_file(self,file_name):
        #save tree structure to file
        fp = open(file_name+".json","wb")
        node_desc_list = self.root.get_node_desc_list()
        node_desc_list["data_shape"] = self.X.shape
        json.dump(node_desc_list,fp,indent=4)
        fp.close()
        ## save data to file
        self.X.tofile(file_name+".data")
        return
        
    def load_from_file(self,file_name):
        fp = open(file_name+".json","r")
        node_desc_dict = json.load(fp)
        fp.close()
        
        data_shape = node_desc_dict["data_shape"]#o
        self.X = np.fromfile(file_name+".data").reshape(data_shape) #load
        self.dim = data_shape[1]
        root_id = node_desc_dict["root"]
        self.root = ClusterTreeNode(self.N,self.K,self.dim)
        self.root.construct_from_node_desc_list(root_id,node_desc_dict,self.X)
        
    def save_to_file_pickle(self,file_name):
        pickleFileName = file_name
        pickleFile = open(pickleFileName, 'wb')
        pickle.dump(self, pickleFile, pickle.HIGHEST_PROTOCOL)
        pickleFile.close()        
       
    def load_from_file_pickle(self,file_name):
        pickleFileName = file_name
        pickleFile = open(pickleFileName, 'rb')
        data = pickle.load(pickleFile)
        self.X = data.X
        self.root = data.root
        self.K = data.K
        self.N = data.N
        pickleFile.close()     
      
    def construct(self,X):
        self.X = X
        self.dim = self.X.shape[1]
        self.root = ClusterTreeNode(self.N,self.K,self.dim)
        self.root.create_subdivision(self.X)

          
    def find_best_example(self,obj,data):
        return self.root.find_best_example(obj,data)


    def find_best_example_exluding_search(self,obj,data):
        node = self.root
        level = 0
        while level < self.K and node.leaf == False:
            print "level",level
            index, value = node.find_best_cluster(obj,data,use_mean=True)
            node = node.clusters[index]
            level += 1
        print level,node.leaf
        return node.find_best_example(obj,data)
          
    def find_best_example_exluding_search_candidates(self, obj, data, n_candidates=1):
        """ Traverses the cluster hierarchy iteratively by evaluating the means
            of the clusters at each level based on the objective function. 
            At the last level the method find_best_example for a KDTree is used.
            Multiple candidates are kept at each level in order to find the global
            optimum.
        """
        results = []
        candidates = []
        candidates.append( (np.inf,self.root) )
        level = 0
        while len(candidates) > 0:#level < self.K and 
            #print "level",level,len(candidates)#,len(results)
            new_candidates = []
            for value,node in candidates:
                if  node.leaf == False:
                    good_candidates = node.find_best_cluster_canditates(obj,data,n_candidates=n_candidates)
                    for c in good_candidates:# value , node tuples
                        heapq.heappush(new_candidates,c)
                else:
                    #heapq.heappush(results,(value,node.clusters[0].root.point) )
                    kdtree_result = node.find_best_example(obj,data) 
                    heapq.heappush(results,kdtree_result)
            
            candidates = new_candidates[:n_candidates]
            #node = node.clusters[index]
            level += 1
        #print level,node.leaf
        #print len(results),results
        if len(results)>0:
            return heapq.heappop(results)    
        else:
            print "#################failed to find a result"
            return np.inf, self.X[self.root.indices[0]]
        
    def find_best_example_exluding_search_candidates_boundary(self, obj, data, n_candidates=5):
        """ Traverses the cluster hierarchy iteratively by evaluating the means
            of the clusters at each level based on the objective function. 
            At the last level the method find_best_example for a KDTree is used.
            Multiple candidates are kept at each level in order to find the global
            optimum.
            Uses boundary based on maximum cost of last iteration to ignore bad candidates.
            Note requires more candidates to prevent
        """
        boundary = np.inf
        results = []
        candidates = []
        candidates.append( (np.inf,self.root) )
        level = 0
        while len(candidates) > 0:
            boundary = max([c[0] for c in candidates])
            print boundary
            new_candidates = []
            for value,node in candidates:
                
                if  node.leaf == False:
                    good_candidates = node.find_best_cluster_canditates(obj,data,n_candidates=n_candidates)
                    for c in good_candidates:# value , node tuples
                         heapq.heappush(new_candidates,c)
                else:
                    kdtree_result = node.find_best_example(obj,data) 
                    heapq.heappush(results,kdtree_result)

                                   
            candidates = [c for c in new_candidates[:n_candidates] if c[0] < boundary]
            if len(results)==0 and len(candidates) == 0:
                candidates = new_candidates[:n_candidates]
            #node = node.clusters[index]
            level += 1
        #print level,node.leaf
        #print len(results),results
        if len(results)>0:
            return heapq.heappop(results)    
        else:
            print "#################failed to find a good result"
            return np.inf, self.X[0]
        
    def find_best_example_exluding_search_candidates_knn(self, obj, data, n_candidates=1, K=20):
        """ Traverses the cluster hierarchy iteratively by evaluating the means
            of the clusters at each level based on the objective function. 
            At the last level the method find_best_example for a KNN Interpolation is used.
            Multiple candidates are kept at each level in order to find the global
            optimum.
        """
        results = []
        candidates = []
        candidates.append( (np.inf,self.root) )
        level = 0
        while len(candidates) > 0:#level < self.K and 
            #print "level",level,len(candidates)#,len(results)
            new_candidates = []
            for value,node in candidates:
                if  node.leaf == False:
                    good_candidates = node.find_best_cluster_canditates(obj,data,n_candidates=n_candidates)
                    for c in good_candidates:# value , node tuples
                        heapq.heappush(new_candidates,c)
                else:
                    #heapq.heappush(results,(value,node.clusters[0].root.point) )
                    #kdtree_result = node.find_best_example(obj,data)
  
                    kdtree_result = node.find_best_example_knn(obj, data, k=50)
             
                    heapq.heappush(results,kdtree_result)
            
            candidates = new_candidates[:n_candidates]
            #node = node.clusters[index]
            level += 1
        #print level,node.leaf
        #print len(results),results
        if len(results)>0:
            return heapq.heappop(results)    
        else:
            print "#################failed to find a result"
            return np.inf, self.X[self.root.indices[0]]

def test_cluster_hierarchy_construction(X):
    N = 4
    K = 4#0
    cluster_tree = ClusterTree(N, K)
    cluster_tree.construct(X)
    cluster_tree.save_to_file("tree")#.json
    cluster_tree2 = ClusterTree(N, K)
    cluster_tree2.load_from_file("tree")
    return
    
def main():
    n_samples = 10000#0
    n_dim = 30

    X = np.random.random( (n_samples,n_dim) )
    print X.shape
    start = time.clock()
    #kdtree = KDTree()
    #test_kd_tree(X)
    test_cluster_hierarchy_construction(X)
    print "finished construction in ",time.clock()-start, "seconds"
    return
    
if __name__ == "__main__":
    main()

    
    
