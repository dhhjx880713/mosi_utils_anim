# -*- coding: utf-8 -*-
"""
Created on Wed Aug 05 12:20:57 2015

@author: hadu01
"""

import os
import numpy as np
import glob
from ....morphablegraphs.animation_data.bvh import BVHReader
from ....morphablegraphs.construction.preprocessing.motion_dtw import MotionDynamicTimeWarping
from ...libtest import params, pytest_generate_tests
ROOTDIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-4]) + os.sep
TESTDATAPATH = ROOTDIR + os.sep + r'../test_data/constrction/preprocessing/motion_dtw'
TESTRESULTPATH = ROOTDIR + os.sep + r'../test_output/constrction/preprocessing/motion_dtw'


class TestMotionDynamicTimeWarping(object):

    def setup_method(self, method):
        self.motion_dtw = MotionDynamicTimeWarping()
        
    param_load_motion_from_files_for_DTW = [{'folder_path': TESTDATAPATH}]
    
    @params(param_load_motion_from_files_for_DTW)
    def test_load_motion_from_files_for_DTW(self, folder_path):
        files = glob.glob(folder_path + os.sep + '*.bvh')
        self.motion_dtw.load_motion_from_files_for_DTW(folder_path)
        assert len(files) == len(self.motion_dtw.aligned_motions)
    

    param_get_distgrid = [{'ref_bvh': TESTDATAPATH + os.sep + 'walk_001_4_sidestepLeft_139_263.bvh',
                           'test_bvh': TESTDATAPATH + os.sep + 'walk_001_4_sidestepLeft_263_425.bvh',
                           'res': 0.80117365}]

    @params(param_get_distgrid)
    def test_get_distgrid(self, ref_bvh, test_bvh, res):
        ref_bvhreader = BVHReader(ref_bvh)
        test_bvhreader = BVHReader(test_bvh)
        self.motion_dtw.ref_bvhreader = ref_bvhreader
        ref_motion = {'filename': os.path.split(ref_bvh)[-1], 
                      'frames': ref_bvhreader.frames}
        test_motion = {'filename': os.path.split(test_bvh)[-1],
                       'frames': test_bvhreader.frames}
        n_ref_frames = len(ref_bvhreader.frames)
        n_test_frames = len(test_bvhreader.frames)
        distgrid = self.motion_dtw.get_distgrid(ref_motion, test_motion)               
        assert distgrid.shape == (n_ref_frames, n_test_frames)
        assert round(distgrid[0, 0], 3) == round(res, 3)

    param_calculate_path = [{'ref_bvh': TESTDATAPATH + os.sep + 'walk_001_4_sidestepLeft_139_263.bvh',
                             'test_bvh': TESTDATAPATH + os.sep + 'walk_001_4_sidestepLeft_263_425.bvh',
                             'res': {'pathx': np.array([1.,    2.,    2.,    3.,    3.,    4.,    4.,    5.,    5.,
                                                        6.,    6.,    7.,    7.,    8.,    8.,    9.,    9.,   10.,
                                                        10.,   11.,   11.,   12.,   12.,   13.,   13.,   14.,   14.,
                                                        15.,   15.,   16.,   16.,   17.,   17.,   18.,   18.,   19.,
                                                        19.,   20.,   20.,   21.,   21.,   22.,   22.,   23.,   23.,
                                                        24.,   24.,   25.,   25.,   26.,   26.,   27.,   28.,   29.,
                                                        30.,   31.,   32.,   33.,   34.,   35.,   36.,   37.,   38.,
                                                        39.,   40.,   41.,   42.,   43.,   44.,   45.,   46.,   47.,
                                                        48.,   49.,   50.,   51.,   52.,   53.,   54.,   55.,   56.,
                                                        57.,   58.,   59.,   60.,   61.,   61.,   62.,   63.,   64.,
                                                        65.,   66.,   66.,   67.,   67.,   68.,   68.,   69.,   69.,
                                                        70.,   70.,   71.,   71.,   72.,   72.,   73.,   74.,   75.,
                                                        76.,   77.,   78.,   79.,   80.,   81.,   82.,   83.,   84.,
                                                        85.,   86.,   87.,   88.,   89.,   90.,   91.,   92.,   93.,
                                                        94.,   95.,   96.,   97.,   98.,   99.,  100.,  101.,  102.,
                                                        103.,  104.,  105.,  106.,  107.,  108.,  109.,  110.,  111.,
                                                        112.,  113.,  114.,  115.,  116.,  117.,  118.,  119.,  120.,
                                                        120.,  121.,  121.,  122.,  122.,  123.,  123.,  124.,  124.]),
                                     'pathy': np.array([1.,    2.,    3.,    4.,    5.,    6.,    7.,    8.,    9.,
                                                        10.,   11.,   12.,   13.,   14.,   15.,   16.,   17.,   18.,
                                                        19.,   20.,   21.,   22.,   23.,   24.,   25.,   26.,   27.,
                                                        28.,   29.,   30.,   31.,   32.,   33.,   34.,   35.,   36.,
                                                        37.,   38.,   39.,   40.,   41.,   42.,   43.,   44.,   45.,
                                                        46.,   47.,   48.,   49.,   50.,   51.,   52.,   53.,   54.,
                                                        55.,   56.,   57.,   58.,   59.,   60.,   61.,   62.,   63.,
                                                        64.,   65.,   66.,   67.,   68.,   69.,   70.,   71.,   72.,
                                                        73.,   74.,   75.,   76.,   77.,   78.,   79.,   80.,   81.,
                                                        82.,   83.,   84.,   85.,   86.,   87.,   88.,   89.,   90.,
                                                        91.,   92.,   93.,   94.,   95.,   96.,   97.,   98.,   99.,
                                                        100.,  101.,  102.,  103.,  104.,  105.,  106.,  107.,  108.,
                                                        109.,  110.,  111.,  112.,  113.,  114.,  115.,  116.,  117.,
                                                        118.,  119.,  120.,  121.,  122.,  123.,  124.,  125.,  126.,
                                                        127.,  128.,  129.,  130.,  131.,  132.,  133.,  134.,  135.,
                                                        136.,  137.,  138.,  139.,  140.,  141.,  142.,  143.,  144.,
                                                        145.,  146.,  147.,  148.,  149.,  150.,  151.,  152.,  153.,
                                                        154.,  155.,  156.,  157.,  158.,  159.,  160.,  161.,  162.]),
                                     'dist': 124.39094053296148}}]

    @params(param_calculate_path)
    def test_calculate_path(self, ref_bvh, test_bvh, res):
        ref_bvhreader = BVHReader(ref_bvh)
        test_bvhreader = BVHReader(test_bvh)
        self.motion_dtw.ref_bvhreader = ref_bvhreader
        ref_motion = {'filename': os.path.split(ref_bvh)[-1], 
                      'frames': ref_bvhreader.frames}
        test_motion = {'filename': os.path.split(test_bvh)[-1],
                       'frames': test_bvhreader.frames}
        distgrid = self.motion_dtw.get_distgrid(ref_motion, test_motion)
        pathx, pathy, dist = self.motion_dtw.calculate_path(distgrid)
        for i in xrange(len(pathx)):
            assert pathx[i] == res['pathx'][i]
        for i in xrange(len(pathy)):
            assert pathy[i] == res['pathy'][i]
        assert round(dist, 3) == round(res['dist'], 3)
    
    
    param_get_warping_index = [{'ref_bvh': TESTDATAPATH + os.sep + 'walk_001_4_sidestepLeft_139_263.bvh',
                                'test_bvh': TESTDATAPATH + os.sep + 'walk_001_4_sidestepLeft_263_425.bvh',
                                'res': [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 60, 61, 62, 63, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 71, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 120, 121, 121, 122, 122, 123, 123]}]

    @params(param_get_warping_index)
    def test_get_warping_index(self, ref_bvh, test_bvh, res):
        ref_bvhreader = BVHReader(ref_bvh)
        test_bvhreader = BVHReader(test_bvh)
        self.motion_dtw.ref_bvhreader = ref_bvhreader
        ref_motion = {'filename': os.path.split(ref_bvh)[-1], 
                      'frames': ref_bvhreader.frames}
        test_motion = {'filename': os.path.split(test_bvh)[-1],
                       'frames': test_bvhreader.frames}
        distgrid = self.motion_dtw.get_distgrid(ref_motion, test_motion)
        pathx, pathy, dist = self.motion_dtw.calculate_path(distgrid)
        warping_index = self.motion_dtw.get_warping_index(pathx, pathy, distgrid.shape)
        assert warping_index == res
    
