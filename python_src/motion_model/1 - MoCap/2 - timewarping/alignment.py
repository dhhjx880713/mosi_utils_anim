# -*- coding: utf-8 -*-
"""
@package Singularity removing

This Package contains functionality to align a set of motion segments.
It includes calculating an appropriate reference motion as well as
calculating the timewarping functions to align the rest of the segments to
this reference segment.

@author: MAUERMA

"""
from MotionSynthesis.frameDistance import frameDistance
from AnimationEngine.AnimationController import \
    SkeletonAnimationData1DBlendingController as SkeletonAnimationController
from AnimationEngine import AnimationData
from cgkit.cgtypes import vec3
import numpy as np
import matplotlib.pyplot as plt
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
import json
import os
import glob


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
#     print distgrid
#     print type(distgrid)
    robjects.conversion.py2ri = numpy2ri.numpy2ri
    rdistgrid = robjects.Matrix(distgrid)
#     rdistgrid = numpy2ri.numpy2ri(np.array(distgrid))
    

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


def align_all_frames(animationdata):
    """ @brief Align all frames from the given motion

    decoupling each fram from its translation in the ground plane and the
    rotation of its hips about the up axis

    @param animationdata SkeletonAnimationData - The animation data.
    Note: The animationdata is changed in place

    @return None
    """
    frames = animationdata.getFramesData(weighted=0)

    frames[:, 0] = 0
    frames[:, 2] = 0
    frames[:, 4] = 0

    animationdata.fromVectorToMotionData(np.ravel(frames), weighted=0)


def calculate_all_distgrids(folder, jsonfile, numsamples=15):
    """ @brief Calculate all distgrids

    Calculate the distance grids for each motion pair in the folder.
    If the number of motions in the given folder is greater than
    numsamples, then random motions from the folder are choosed for this
    process.

    @param folder string - The folder where the bvh files can be found
    @param jsonfile string - The json filename (including folder) where the
        distance grids will be saved
    @param mumsamples int - The number of motions to be used for this process

    @return A dictionary having the reference filename as first key
        and the test filename as second key and the calculated grid as value
        (e.g.: data['mo1.bvh']['mo2.bvh'])
    """

    jsondata = {}

    if folder[-1] != os.sep:
        folder = folder + os.sep

    files = glob.glob(folder + '*.bvh')
    motions = []

    # load and align a few random motions:
    indices = np.arange(len(files))
    if numsamples <= len(files):
        np.random.shuffle(indices)
        indices = indices[:numsamples]

    print files
    for i in indices:
        animationdata = AnimationData.SkeletonAnimationData()
        animationdata.buildFromBVHFile(files[i])
        align_all_frames(animationdata)
        motion = SkeletonAnimationController(animationdata, visualize=False)

        motions.append(motion)

    # calculate disgrids:
    n = len(motions)

    numcalcs = (n * (n-1)) / 2
    counter = 0

    for i in xrange(n):
        for j in xrange(i+1, n):
            counter += 1
            print counter, "/", numcalcs
            distgrid = calc_distance_matrix(motions[i], motions[j],
                                            distonly=True, verbose=False)

            try:
                jsondata[motions[i].name][motions[j].name] = distgrid.tolist()
            except KeyError:
                jsondata[motions[i].name] = {}
                jsondata[motions[i].name][motions[j].name] = distgrid.tolist()

            try:
                jsondata[motions[j].name][motions[i].name] = distgrid.tolist()
            except KeyError:
                jsondata[motions[j].name] = {}
                jsondata[motions[j].name][motions[i].name] = distgrid.tolist()

    handle = open(jsonfile, 'wb')
    tmpdata = {}
    tmpdata['data'] = jsondata
    json.dump(tmpdata, handle)
    handle.close()

    return jsondata


def get_best_motion(jsondata):
    """ @brief Calculate the best motion based on the given distmatrices in the
        format specified by "calculate_all_distgrids"

        @param jsondata dict of dicts - The data returned from the
        calculate_all_distgrids function

        @returns The name of the best
    """
    n = len(jsondata)

    errors = {}
    counter = 0
    for key in jsondata:
        error = 0
        for key2 in jsondata[key]:
            x, y, dist = calculate_path(jsondata[key][key2])
            error += dist
        errors[key] = error
        counter += 1
        print counter, '/', n

    return min(errors, key=errors.get)


def translate_toroot(animation):
    """ @brief Translates the motion to the root.

    Translates the given motions root position in the first frame to (0,90,0)
    and the rest of the files according to the changes made in the first frame
    Also rotates the y rotation of the root joint to be approximatly zero

    @param animation SkeletonAnimationData - The motion

    @returns The translated animation as SkeletonAnimationController
    """
    # Align motionroot to 0,0
    framedata = animation.getFrameLineVector(0, weighted=0)
    x = framedata[0]
    y = framedata[1]
    z = framedata[2]
    theta = framedata[4]
    vector = vec3(-x, -y + 90, -z)
    motion = SkeletonAnimationController(animation, visualize=False)
    motion.translateAnimationFrames(0, vector)
    motion.rotateAnimationFrames(-theta)

    return motion


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


def alignment(source, target, distgridfile, feature, numsamples=15):
    if source[-1] != os.sep:
        source = source + os.sep

    if os.path.exists(distgridfile):
        handle = open(distgridfile, 'rb')
        tmpdata = json.load(handle)
        jsondata = tmpdata['data']
        handle.close()

    else:
        jsondata = calculate_all_distgrids(source, distgridfile,
                                                    numsamples=numsamples)
    reffile = source + get_best_motion(jsondata)
    files = []

    if type(source) == basestring:
        source = [source]
    for tmp in source:
        files.extend(glob.glob(tmp + '*.bvh'))

    if not os.path.exists(target):
        os.makedirs(target)
    if target[-1] != os.sep:
        target = target + os.sep

    warpingfile = target + 'timewarping.json'

    warp_all_motions_to_ref(reffile, source, target, warpingfile)

    return reffile


def main():
    """ @brief Main function to show a default work process with this modul """

    # Specifie Folder
    feature = 'Bip01_L_Toe0'
    folder = r'Optitrack_rocketbox - 14.06.12/segments/' + \
        feature + '_segments/'

    # Specifie json file
    distgridfile = r'distgrids_' + feature + '.json'

    # Check if the json file already exists
    if os.path.exists(distgridfile):
        handle = open(distgridfile, 'rb')
        tmpdata = json.load(handle)
        jsondata = tmpdata['data']
        handle.close()

    # If not, calculate the distance grids
    else:
        jsondata = calculate_all_distgrids(folder, distgridfile, numsamples=15)

    # get the best motion
    reffile = get_best_motion(jsondata)

    # Specifie targetfolder and timefunction json file
    targetfolder = r'Optitrack_rocketbox - 14.06.12/segments/' \
        'registered_' + feature + '_segments/'

    warpingfile = r'Optitrack_rocketbox - 14.06.12/segments/' \
        'registered_' + feature + '_segments/timewarping.json'

    # warp the motion to the calculated reference motion
    warp_all_motions_to_ref(reffile, folder, targetfolder, warpingfile)

    print "The reference file is:", reffile
    return

    
if __name__ == '__main__':
     main()

