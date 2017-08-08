""" get_distgrid and find_path are copied from from http://musicinformationretrieval.com/dtw.html"""
import numpy as np
import scipy as sp
from ..animation_data.motion_distance import _transform_invariant_point_cloud_distance


def get_distgrid(x, y, distance_measure=_transform_invariant_point_cloud_distance):
    Nx = len(x)
    Ny = len(y)
    # Step 1: compute pairwise distances.
    S = np.zeros([Nx, Ny])
    for i in range(Nx):
        for j in range(Ny):
            S[i, j] = distance_measure(x[i], y[j])

    # Step 2: compute cumulative distances.
    D = sp.zeros_like(S)
    D[0, 0] = S[0, 0]
    for i in range(1, Nx):
        D[i, 0] = D[i - 1, 0] + S[i, 0]
    for j in range(1, len(y)):
        D[0, j] = D[0, j - 1] + S[0, j]
    for i in range(1, Nx):
        for j in range(1, Ny):
            D[i, j] = min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]) + S[i, j]
    return D


def find_path(D, Nx, Ny):
    # Step 3: find optimal path.
    backsteps = [[None for j in range(Ny)] for i in range(Nx)]
    for i in range(1, Nx):
        backsteps[i][0] = (i - 1, 0)
    for j in range(1, Ny):
        backsteps[0][j] = (0, j - 1)
    for i in range(1, Nx):
        for j in range(1, Ny):
            candidate_steps = ((i - 1, j - 1), (i - 1, j), (i, j - 1),)
            candidate_distances = [D[m, n] for (m, n) in candidate_steps]
            backsteps[i][j] = candidate_steps[np.argmin(candidate_distances)]

    xi, yi = Nx - 1, Ny - 1
    path = [(xi, yi)]
    while xi > 0 or yi > 0:
        xi, yi = backsteps[xi][yi]
        path.insert(0, (xi, yi))
    return path


def run_dtw(x, y):
    Nx = len(x)
    Ny = len(y)
    D = get_distgrid(x, y)
    return find_path(D, Nx, Ny), D


def get_warping_function(path):
    """ @brief Calculate the warping function from a given set of x and y values

    Calculate the warping path from the return values of the dtw R function
    This R functions returns a set of (x, y) pairs saved as x vecotr and
    y vector. These pairs are used to initialize a Bitmatrix representing
    the Path through the Distance grid.
    The indexes for the testmotion is than calculated based on this matrix.

    @param path list of tuples- The path returned by find_path

    @return A list with exactly x Elements.
    """

    x, y = np.array(path).T
    #convert to coordinates
    coordinates = [(int(x[i]), int(y[i])) for i in range(len(x))]


    #set coordinates to 1
    shape = (int(x[-1])+1, int(y[-1])+1)
    pathmatrix = np.zeros(shape)
    for coord in coordinates:
        pathmatrix[coord] = 1

    warping_function = []
    for i in range(shape[1]):
        # find first non zero index along column i
        index = np.nonzero(pathmatrix[:, i])[0][-1]
        warping_function.append(index)

    return warping_function


def warp_motion(frames, warp_function):
    new_frames = []
    for idx in warp_function:
        new_frames.append(frames[idx])
    return new_frames


def find_reference_motion(point_clouds):
    return point_clouds[0]
    n = len(point_clouds)
    avg_distances = []
    for i, pi in enumerate(point_clouds):
        avg_distances.append(0)
        for pj in point_clouds:
            path, D = run_dtw(pi, pj)
            path_cost = 0
            for c in path:
                path_cost += D[c[0], c[1]]
            avg_distances[i] += path_cost
        avg_distances[i] /= n

    best_idx = 0
    best_d = np.inf
    for idx, d in enumerate(avg_distances):
        if d < best_d:
            best_idx = idx
    return point_clouds[best_idx]
