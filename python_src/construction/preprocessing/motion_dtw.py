# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:06:37 2015

@author: du, MAUERMA
"""

import os
import sys
ROOT_DIR = os.sep.join(['..'] * 2)
sys.path.append(ROOT_DIR)
from animation_data.bvh import BVHReader, BVHWriter
from motion_normalization import MotionNormalization
from animation_data.motion_editing import calculate_frame_distance
from animation_data.skeleton import Skeleton
import numpy as np
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
import matplotlib.pyplot as plt
import glob
import json

class MotionDynamicTimeWarping(MotionNormalization):
     
    def __init__(self, verbose=False):
        super(MotionDynamicTimeWarping, self).__init__()
        self.ref_motion = {}
        self.verbose = verbose
    
    def load_motion_from_files_for_DTW(self, folder_path):
        if not folder_path.endswith(os.sep):
            folder_path += os.sep
        print "search bvh files in " + folder_path
        motion_files = glob.glob(folder_path + '*.bvh')
        print str(len(motion_files)) + " are found!"
        self.aligned_motions = {}
        for bvh_file_path in motion_files:
            bvhreader = BVHReader(bvh_file_path)
            filename = os.path.split(bvh_file_path)[-1]
            self.aligned_motions[filename] = bvhreader.frames
        self.ref_bvhreader = bvhreader
    
    def dtw(self):
        if self.ref_motion == {}:
            print "automatically search best reference motion"
            self.find_ref_motion()
        self.warp_all_motions_to_ref_motion()
    
    def find_ref_motion(self):
        """The reference motion can be found by using the motion which has 
           minimal average distance to others
        """
        self.len_aligned_motions = len(self.aligned_motions)
        self.get_all_distgrid()
        # calculate motion distance from distgrid
        average_dists = {}
        counter = 0
        for ref_filename in self.dic_distgrid.keys():
            average_dist = 0
            for test_filename in self.dic_distgrid[ref_filename].keys():
                x, y ,dist = self.calculate_path(self.dic_distgrid[ref_filename][test_filename])
                average_dist += dist
            average_dist = average_dist/self.len_aligned_motions
            counter += 1
            average_dists[ref_filename] = average_dist
            print counter, '/', self.len_aligned_motions
        ref_filename = min(average_dists, key=lambda k:average_dists[k])
        self.ref_motion['filename'] = ref_filename
        self.ref_motion['frames'] = self.aligned_motions[ref_filename]
    
    def set_ref_motion(self, filepath):
        ref_filename = os.path.split(filepath)[-1]
        ref_bvh = BVHReader(filepath)
        self.ref_motion['filename'] = ref_filename
        self.ref_motion['frames'] = ref_bvh.frames
    
    def warp_test_motion_to_ref_motion(self, ref_motion, test_motion):
        distgrid = self.get_distgrid(ref_motion, test_motion)
        ref_indeces, test_indeces, dist = self.calculate_path(distgrid)  
        shape = distgrid.shape
        shape = (shape[1], shape[0])
        warping_index = self.get_warping_index(test_indeces, ref_indeces, shape)
        warped_frames = self.get_warped_frames(warping_index,
                                               test_motion['frames'])  
        return warped_frames, warping_index                                       
    
    def warp_all_motions_to_ref_motion(self):
        if self.ref_motion == {}:
            raise ValueError('There is no reference motion for DTW')
        if self.aligned_motions == {}:
            raise ValueError('No motion for DTW')
        self.warped_motions = {}
#        counter = 0
        for filename, frames in self.aligned_motions.iteritems():
            test_motion = {}
            test_motion['filename'] = filename
            test_motion['frames'] = frames
            warped_frames, warping_index = self.warp_test_motion_to_ref_motion(self.ref_motion,
                                                                               test_motion)                                                                  
            self.warped_motions[filename] = {}
            self.warped_motions[filename]['warping_index'] = warping_index
            self.warped_motions[filename]['frames'] = warped_frames                                                                   
                                                                               
    def save_warped_motion(self, save_path):
        if not save_path.endswith(os.sep):
            save_path += os.sep
        warping_index_dic = {}
        for filename, motion_data in self.warped_motions.iteritems():
            BVHWriter(save_path+filename, self.ref_bvhreader, 
                      motion_data['frames'], 
                      frame_time=self.ref_bvhreader.frame_time,
                      is_quaternion=False)
            warping_index_dic[filename] = np.asarray(motion_data['warping_index']).tolist()
        warping_index_file_path = save_path + 'timewarping.json'   
        with open(warping_index_file_path, 'wb') as outfile:
            json.dump(warping_index_dic, outfile)        
    
    def get_warped_frames(self, warping_index, frames):
        if warping_index[-1] > len(frames):
            raise ValueError('index is larger than length of frames!')
        warped_frames = []
        for idx in warping_index:
            warped_frames.append(frames[idx])
        return warped_frames
    
    def get_distgrid(self, ref_motion, test_motion):
        try:
            skeleton = Skeleton(self.ref_bvhreader)
        except:
            ValueError('Cannot initialize skeleton information')
        n_ref_frames = len(ref_motion['frames'])
        n_test_frames = len(test_motion['frames'])
        distgrid = np.zeros([n_ref_frames, n_test_frames])
        for i in xrange(n_ref_frames):
            for j in xrange(n_test_frames):
                distgrid[i,j] = calculate_frame_distance(skeleton,
                                                         ref_motion['frames'][i],
                                                         test_motion['frames'][j])
        if self.verbose:
            ref_indeces, test_indeces, dist = self.calculate_path(distgrid)
            shape = (n_test_frames, n_ref_frames)
            path = self.get_warping_index(test_indeces, ref_indeces, shape)
            plt.figure()
            plt.imshow(distgrid)
            plt.plot(range(len(path)), path, color='red')
            plt.xlabel(ref_motion['filename'])
            plt.ylabel(test_motion['filename'])
            plt.title('similarity grid with path')
            plt.show()                                                             
        return distgrid
    
    def get_warping_index(self, x, y, shape):
        """ @brief Calculate the warping index from a given set of x and y values
    
        Calculate the warping path from the return values of the dtw R function
        This R functions returns a set of (x, y) pairs saved as x vecotr and
        y vector. These pairs are used to initialize a Bitmatrix representing
        the Path through the Distance grid.
        The indexes for the testmotion is than calculated based on this matrix.
    
        @param x list of ints - The x-values
        @param y list of ints - The y-values
        @param shape array or tuple with two elements - The shape of the distgrid,\
            normaly (testmotion_framenumber, refmotion_framenumber)
        @param verbose (Optional) Displays the warping indexes
    
        @return A list with exactly refmotion_framenumber Elements.
        """
        # create Pairs:
        path_pairs = [(int(x[i])-1, int(y[i])-1) for i in xrange(len(x))]
    
        # create Pathmatirx:
        pathmatrix = np.zeros(shape)
        for pair in path_pairs:
            pathmatrix[pair] = 1
    
        warpingIndex = []
        for i in xrange(shape[1]):
            index = np.nonzero(pathmatrix[:, i])[0][-1]
            warpingIndex.append(index)
    
        if self.verbose:
            print "warping index from R is: "
            print warpingIndex
        return warpingIndex        
        
    def calculate_path(self, distgrid, steppattern="typeIb", window="itakura"):
        """ @brief Calculates the optimal path through the given Distance grid
    
        Calculates an optimal path through the given Distance grid with
        the R Package "dtw". The path restrictions can be varried with the
        steppattern and window parameter
        !!! NOTE: This package is distributed under the GPL(v2) Version !!!
    
        @param distgrid arraylike object with shape
            (testmotion_framenumber, refmotion_framenumber) -
            The calculated distance grid
        @param steppattern string - The steppattern to be used.
            The steppattern is normaly used to define local constraints
            See "http://cran.r-project.org/web/packages/dtw/dtw.pdf" for a detailed
            list of available options
        @param window string - The window to be used.
            The window is normaly used to define global constraints
            Available options are: "none", "itakura", "sakoechiba", "slantedband"
            See "http://cran.r-project.org/web/packages/dtw/dtw.pdf" for a detailed
            description
    
        @return numpy array - matched elements: indices in x
        @return numpy array - corresponding mapped indices in y
        @return float - the minimum global distance computed. normalized for path
            length, if normalization is
            known for chosen step pattern
        """
    
        robjects.conversion.py2ri = numpy2ri.numpy2ri
        distgrid = np.asarray(distgrid)
        m, n = distgrid.shape
        maxLen = float(max(m, n))
        minLen = float(min(m, n))
        ratio = maxLen/minLen
        if ratio > 1.5:
            steppattern = 'symmetric2'
            window='none'
        rdistgrid = robjects.Matrix(np.asarray(distgrid)) 
        rcode = '''
                library("dtw")
    
                path = dtw(x = as.matrix(%s), y = NULL,
                           dist.method="Euclidean",
                           step.pattern = %s,
                           window.type = "%s",
                           keep.internals = FALSE,
                           distance.only = FALSE,
                           open.end = FALSE,
                           open.begin = FALSE)
                xindex = path$index1
                yindex = path$index2
                dist = path$distance
    
                ''' % (rdistgrid.r_repr(), steppattern, window)
    
        robjects.r(rcode)
    
        return np.array(robjects.globalenv['xindex']), \
            np.array(robjects.globalenv['yindex']), \
            np.array(robjects.globalenv['dist'])[0]
        
    def get_all_distgrid(self):
        """calculate the distance matrix for each pair of motions in 
           aligned_motions
        """
        print "start to compute distance grid for all pairs pf motions"
        total_calculation = self.len_aligned_motions * (self.len_aligned_motions - 1)/2
        print "There are %d pairs in total" % total_calculation
        self.dic_distgrid = {}
        counter = 0
        keys = self.aligned_motions.keys()
        for i in xrange(self.len_aligned_motions):
            for j in xrange(i+1, self.len_aligned_motions):
                counter += 1
                print counter, '/', total_calculation
                ref_motion = {'filename': keys[i],
                              'frames': self.aligned_motions[keys[i]]}
                test_motion = {'filename': keys[j],
                               'frames': self.aligned_motions[keys[j]]}              
                distgrid = self.get_distgrid(ref_motion,
                                             test_motion)
                try:
                    self.dic_distgrid[keys[i]][keys[j]] = distgrid
                except:
                    self.dic_distgrid[keys[i]] = {}
                    self.dic_distgrid[keys[i]][keys[j]] = distgrid
                try:
                    self.dic_distgrid[keys[j]][keys[i]] = distgrid
                except:
                    self.dic_distgrid[keys[j]] = {}
                    self.dic_distgrid[keys[j]][keys[i]] = distgrid

def main():
    data_folder = r'C:\repo\data\1 - MoCap\3 - Cutting\test'
    save_path = r'C:\repo\data\1 - MoCap\4 - Alignment\test'
    ref_motion = data_folder + os.sep + 'pick_003_4_first_485_607.bvh'
    dynamicTimeWarper = MotionDynamicTimeWarping()
    dynamicTimeWarper.load_motion_from_files_for_DTW(data_folder)
    dynamicTimeWarper.set_ref_motion(ref_motion)
    dynamicTimeWarper.dtw()
    dynamicTimeWarper.save_warped_motion(save_path)
#    with open('time.json', 'wb') as outfile:
#        json.dump(warping_index_dic, outfile)

def test():
#    testFile1 = r'C:\repo\data\1 - MoCap\3 - Cutting\test\pick_001_1_first_502_604.bvh'
#    testFile2 = r'C:\repo\data\1 - MoCap\3 - Cutting\test\pick_001_3_first_630_771.bvh'
#    bvh1 = BVHReader(testFile1)
#    bvh2 = BVHReader(testFile2)
#    ref_motion = {'filename': 'pick_001_1_first_502_604.bvh', 
#                  'frames': bvh1.frames}
#    test_motion = {'filename': 'pick_001_3_first_630_771.bvh',
#                   'frames': bvh2.frames}
#    dtw =  MotionDynamicTimeWarping()
#    distgrid = dtw.get_distgrid(ref_motion, test_motion)   
    t = {'pick_003_4_first_485_607.bvh': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121], 'pick_001_1_first_502_604.bvh': [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101]}    
    with open('time.json', 'wb') as outfile:
        json.dumps(t, outfile)

if __name__ == "__main__":
    main()
#    test()