# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:06:37 2015

@author: du
"""

import numpy as np
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
import matplotlib.pyplot as plt
import glob
import json
import os
import sys


def get_warping_index(x, y, shape, verbose=False):
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

    if verbose:
        print "warping index from R is: "
        print warpingIndex
    return warpingIndex

def calculate_path(distgrid, steppattern="typeIb", window="itakura"):
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
    rdistgrid = robjects.Matrix(np.array(distgrid))
    

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
        
        
def calc_distance_matrix(refMotion, testMotion, distonly=False, verbose=0):
    """ @brief Calculate the Distancematrix

    Calculate the Distancematrix between the reference motion and the
    Testmotion based on the distancemetrik from
    [Kovar et al 2003, Registration curves]

    @param refMotion SkeletonAnimationData1DBlendingController - The reference
        Motion
    @param testMotion SkeletonAnimationData1DBlendingController - The test
        Motion
    @param verbose (optional) logical - If true, the distance matrix with
        the path will be plotted

    @returns The distance matrix as numpy array
    """
    refframenumber = refMotion.getNumberOfFrames()
    testframenumber = testMotion.getNumberOfFrames()

    distGrid = np.zeros([testframenumber, refframenumber])
    thetaMatrix = np.zeros([testframenumber, refframenumber])
    XoffsetMatrix = np.zeros([testframenumber, refframenumber])
    ZoffsetMatrix = np.zeros([testframenumber, refframenumber])

    if distonly:
        for i in xrange(testframenumber):
            for j in xrange(refframenumber):
                distance,  theta, offset_x, offset_z = \
                    frameDistance(refMotion, j, testMotion, i, windowSize=1)
                distGrid[i, j] = distance

    else:
        for i in xrange(testframenumber):
            for j in xrange(refframenumber):
                distance,  theta, offset_x, offset_z = \
                    frameDistance(refMotion, j, testMotion, i, windowSize=1)

                distGrid[i, j] = distance
                thetaMatrix[i, j] = distance
                XoffsetMatrix[i, j] = distance
                ZoffsetMatrix[i, j] = distance

    if verbose:
        pathx, pathy, dist = calculate_path(distGrid)
        shape = (len(distGrid), len(distGrid[0]))
        path = get_warping_index(pathx, pathy, shape)

        plt.figure()
        plt.imshow(distGrid)
        plt.plot(range(len(path)), path, color='black')
        plt.xlabel(refMotion.name)
        plt.ylabel(testMotion.name)
        plt.title('similarity grid with path')
        plt.show()

    if distonly:
        return distGrid

    return distGrid, thetaMatrix, XoffsetMatrix, ZoffsetMatrix
    
    
def warp_all_motions_to_ref(reffile, path, targetfolder=None, jsonfile=None):
    """ @brief Warp all motions to the reference motion

    Warp all motions to the reference motion and save their
    Timewarping functions in the specified JSON-File

    @param reffile string - The name of the referencefile
    @param path string - The path of the motions to be warped
    @param targetfolder string - the path of the folder where the
    warped motions should be saved. If None, the targetfolder will be equal
    to the path
    @param jsonfile string - the name of the json file where the timewarping
    function will be saved. If None, the targetfolder will be used as path and
    the name is set to 'timefunctions.json'

    @return None
    """
    if targetfolder is None:
        targetfolder = path

#     if jsonfile is None:
#         jsonfile = targetfolder + 'timefunctions.json'

    folder = glob.glob(path + '*.bvh')
    print len(folder)

    refMo_animationData = AnimationData.SkeletonAnimationData()
    refMo_animationData.buildFromBVHFile(reffile)

    refMo = translate_toroot(refMo_animationData)

    functions = {}

    n = len(folder)
    counter = 0

    for f in folder:
        testMo_animationData = AnimationData.SkeletonAnimationData()
        testMo_animationData.buildFromBVHFile(f)
        testMo = translate_toroot(testMo_animationData)

        distgrid, thetaMatrix, XoffsetMatrix, ZoffsetMatrix = \
            calc_distance_matrix(refMo, testMo)
        
        m1 = max(refMo.getNumberOfFrames(), testMo.getNumberOfFrames())
        n1 = min(refMo.getNumberOfFrames(), testMo.getNumberOfFrames())
        ratio = m1/n1
# 
#         print refMo.name
#         print testMo.name
        if ratio > 1.5:
            pathx, pathy, dist = calculate_path(distgrid, steppattern = 'symmetric2', window = 'none')
        else:
            pathx, pathy, dist = calculate_path(distgrid)

        """ Han-Conversion-Style """
        shape = (len(distgrid), len(distgrid[0]))
        warping_index = get_warping_index(pathx, pathy, shape)
#         print type(warping_index)
#         print warping_index
        functions[testMo.name] = np.array(warping_index).tolist()

        testMo.createNewMotionBasedOnFrameIndexList(warping_index)
#         testMo.saveToFile(targetfolder + 'registered_' + testMo.name)
        testMo.saveToFile(targetfolder +  testMo.name)

        counter += 1
        print counter, '/', n

    handle = open(jsonfile, 'wb')
    json.dump(functions, handle)
    handle.close()

class MotionDTW(object):
    def __init__(self):
        pass
    
    def dtw(self):
        pass
    
    def find_ref_motion(self):
        pass
    
    def save_warped_motion(self, save_path):
        pass
    
    def save_time_function(self, save_path):
        if not save_path.endswith(os.sep):
            save_path += os.sep
        filename = save_path + os.sep + 'timewarping.json'    
        pass