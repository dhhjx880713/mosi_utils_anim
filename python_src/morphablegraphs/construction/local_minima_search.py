import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology

#COPIED from http://stackoverflow.com/questions/3986345/how-to-find-the-local-minima-of-a-smooth-multidimensional-array-in-numpy-efficie
#The location of the local minima can be found for an array of arbitrary dimension using Ivan's detect_peaks function, with minor modifications:
def detect_local_minima(arr):
    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local minimum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood minimum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    #
    # we create the mask of the background
    background = (arr==0)
    #
    # a little technicality: we must erode the background in order to
    # successfully subtract it from local_min, otherwise a line will
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    #
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_min mask
    detected_minima = local_min - eroded_background
    #print eroded_background
    #print detected_minima.shape
    return local_min,np.array(np.where(detected_minima)).T

def get_global_minima(distance_matrix, candidates):
    global_minimum = np.inf
    for c in candidates:
        x = c[0]
        y = c[1]

        min = distance_matrix[x][y]
        if global_minimum > min:
            global_minimum = min

    return global_minimum

def filter_minima(distance_matrix, candidates, threshold_factor):
    # find global minimum
    global_minimum = get_global_minima(distance_matrix, candidates)
    print "global minimum", global_minimum
    # filter local minima using the threshold
    filtered_coords = []
    for c in candidates:
        x = c[0]
        y = c[1]
        min = distance_matrix[x][y]
        if min < np.inf and min < global_minimum + (global_minimum * threshold_factor):
            filtered_coords.append([x,y])

    return filtered_coords

def filter_infinity(distance_matrix, candidates):
    filtered_coords = []
    for c in candidates:
        x = c[0]
        y = c[1]
        value = distance_matrix[x][y]
        print x,y,value
        if value < np.inf:
            filtered_coords.append([x,y])
    return filtered_coords




def extracted_filtered_minima(distance_matrix, threshold_factor):
    values, candidates = detect_local_minima(distance_matrix)
    coordinates = filter_minima(distance_matrix, candidates, threshold_factor)
    return coordinates

def trysomething():
    x = np.random.random((5,5))
    values, coordinates = detect_local_minima(x)
    filtered_coordinates = extracted_filtered_minima(x, 5)
    print coordinates
    print filtered_coordinates

if __name__ == "__main__":
    from scipy.signal import argrelextrema

    x = np.random.random((5))
    # for local maxima

    # for local minima
    #print argrelextrema(x, np.less)
    print x
    extrema = argrelextrema(x, np.greater)
    print extrema
