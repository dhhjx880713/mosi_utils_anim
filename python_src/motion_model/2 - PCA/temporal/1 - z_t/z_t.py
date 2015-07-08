#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 21.11.2014

@author: MAMAUER
'''
import json
import numpy as np
import os

ROOT_DIR = os.sep.join([".."] * 4)
OUTPUT_DIR = os.sep.join((ROOT_DIR, 'data', '2 - PCA', 'temporal',
                         '1 - z_t', 'experiments'))


def load_timefunctions(filename):
    """ Load the Frameindices from the OriginalData from a .json File

    Parameters
    ----------
    fileName : str
        The name of the .json File

    Returns
    -------
    A dictionary containing the Frameindices as
    Data and the .bvh Filenames as Keys
    """
    infile = open(filename, 'rb')
    timefunctions = json.load(infile)
    infile.close()

    return timefunctions


def is_strict_incresing(indices):
    """ Check wether the indices are strictly incresing ore not

    Parameters
    ----------
    indices : list
    The Frameindices

    Returns
    -------
    boolean
    """
    for i in xrange(1, len(indices)):
        if np.allclose(indices[i], indices[i-1]) or indices[i] < indices[i-1]:
            return False
    return True


def get_monotonic_indices(indices, epsilon=0.01, delta=0):
    """Return an ajusted set of Frameindices which is strictly monotonic

    Parameters
    ----------
    indices : list
    The Frameindices

    Returns
    -------
    A numpy-Float Array with indices similar to the provided list,
    but enforcing strict monotony
    """
    shifted_indices = np.array(indices, dtype=np.float)
    if shifted_indices[0] == shifted_indices[-1]:
        raise ValueError("First and Last element are equal")

    for i in xrange(1, len(shifted_indices)-1):
        if shifted_indices[i] > shifted_indices[i-1] + delta:
            continue

        while np.allclose(shifted_indices[i], shifted_indices[i-1]) or \
                shifted_indices[i] <= shifted_indices[i-1] + delta:
            shifted_indices[i] = shifted_indices[i] + epsilon

    for i in xrange(len(indices)-2, 0, -1):
        if shifted_indices[i] + delta < shifted_indices[i+1]:
            break

        while np.allclose(shifted_indices[i], shifted_indices[i+1]) or \
                shifted_indices[i] + delta >= shifted_indices[i+1]:
            shifted_indices[i] = shifted_indices[i] - epsilon

    return shifted_indices

def transform_timefunction(w):
    """    Transform the Timewarping Function w to the new space z
           [ z(t) = ln(w(t) - w(t-1)) ]

    Parameters
    ----------
    w : numpy array
        A Numpy-Array containing the timewarping function w

    Returnsw
    -------
    Numpy-Array containing the timewarping function in the z-space
    """
    if not is_strict_incresing(w):
        print "HOSSA"
        w = get_monotonic_indices(w)

    w_tmp = np.array(w)
    w_tmp = w_tmp + 1   # add one to each entry, because we start with 0
    w_tmp = np.insert(w_tmp, 0, 0)  # set w(0) to zero

    w_diff = np.diff(w_tmp)
    z = np.log(w_diff)

    if np.isinf(z).any():
        print w
        print w_tmp
        print w_diff
        print z
    if not is_strict_incresing(w):
        raise ValueError("The Timewarping Functions have to be monotonic")

    return z

def get_input_data_folder(elementary_action, motion_primitive):
    """(COPY FROM "spatial PCA"
    Returns folder path as string without trailing os.sep

    Parameters
    ----------

     * elementary_action: String
    \tElementary action of the motion primitive
     * motion_primitive: String
    \tMotion primitive for which the folder shall be returned

    """

    data_dir_name = "data"
    mocap_dir_name = "1 - MoCap"
    alignment_dir_name = "4 - Alignment"

    input_dir = os.sep.join([ROOT_DIR,
                             data_dir_name,
                             mocap_dir_name,
                             alignment_dir_name,
                             elementary_action,
                             motion_primitive])

    return input_dir



def __main__():
    """ Main function to demonstrate functionality of this module """
    elementary_action = 'walk'
    primitive = 'endLeftStance'

    folder = get_input_data_folder('elementary_action_%s' % elementary_action,
                                   primitive)
    inputfile = os.sep.join((folder, 'timewarping.json'))

    outputfilename = 'z_t_%s_%s.json' % (elementary_action, primitive)
    outputfile = os.sep.join((OUTPUT_DIR, outputfilename))

    z_t = {}
    w = load_timefunctions(inputfile)       # Returns a dict
    for f in w:                             # Iterate over keys (filenames)
        w_t = get_monotonic_indices(w[f], epsilon=0.01)
        z_t[f] = transform_timefunction(w_t).tolist()
    f = open(outputfile, 'w+')
    json.dump(z_t, f)
    f.close()


if __name__=='__main__':
    __main__()
