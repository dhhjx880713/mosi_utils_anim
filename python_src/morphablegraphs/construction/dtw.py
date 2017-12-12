""" get_distgrid and find_path are copied from from http://musicinformationretrieval.com/dtw.html"""
import numpy as np
import scipy as sp
from ..animation_data.motion_distance import _transform_invariant_point_cloud_distance
from fastdtw import fastdtw

import asyncio
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

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


def get_warping_function(coordinates):
    """ @brief Calculate the warping function from a given set of x and y values

    Calculate the warping path from the return values of the dtw R function
    This R functions returns a set of (x, y) pairs saved as x vecotr and
    y vector. These pairs are used to initialize a Bitmatrix representing
    the Path through the Distance grid.
    The indexes for the testmotion is than calculated based on this matrix.

    @param path list of tuples- The path returned by find_path

    @return A list with exactly x Elements.
    """

    #set coordinates to 1
    shape = (int(coordinates[-1][0])+1, int(coordinates[-1][1])+1)
    pathmatrix = np.zeros(shape)
    print("path matrix",shape)
    for coord in coordinates:
        pathmatrix[coord] = 1

    warping_function = []
    for i in range(shape[0]):
        # find first non zero index along row i
        index = np.nonzero(pathmatrix[i, :])[0][-1]
        warping_function.append(index)

    return warping_function




def warp_motion(frames, warp_function):
    new_frames = []
    for idx in warp_function:
        new_frames.append(frames[idx])

    print("warped",len(frames),len(new_frames))
    return new_frames


def find_optimal_dtw(point_clouds):
    #return point_clouds[0]
    n = len(point_clouds)
    dtw_results = []
    avg_distances = []
    for i, pi in enumerate(point_clouds):
        avg_distances.append(0)
        dtw_results.append([])
        for j, pj in enumerate(point_clouds):
            print("start dtw", i, j)
            path_cost, path = fastdtw(pi, pj, dist=_transform_invariant_point_cloud_distance)
            dtw_results[i].append(path)
            avg_distances[i] += path_cost

        avg_distances[i] /= n

    best_idx = 0
    best_d = np.inf
    for idx, d in enumerate(avg_distances):
        if d < best_d:
            best_idx = idx
    return dtw_results[best_idx]



def run_dtw_process(params):
    ref_idx, point_clouds = params
    dtw_results = []
    cost = 0
    pi = point_clouds[ref_idx]
    for j, pj in enumerate(point_clouds):
        print("start dtw", ref_idx, j)
        #path, D = run_dtw(pi, pj)
        #path_cost = sum([D[c[0], c[1]] for c in path])
        path_cost, path = fastdtw(pi, pj, dist=_transform_invariant_point_cloud_distance)
        dtw_results.append(path)
        cost += path_cost
    return cost/len(point_clouds), dtw_results

@asyncio.coroutine
def run_dtw_coroutine(pool, params, results):
    print("start task")
    ref_idx, point_clouds = params
    fut = pool.submit(run_dtw_process, params)
    while not fut.done() and not fut.cancelled():
        #print("run batch")
        yield from asyncio.sleep(0.1)
    print("done")
    results[ref_idx] = fut.result()


def find_optimal_dtw_async(point_clouds, mean_idx=-1):
    n_workers = max(cpu_count()-1, 1)
    pool = ProcessPoolExecutor(max_workers=n_workers)
    if 0 <= mean_idx < len(point_clouds):
        x_point_clouds = [point_clouds[mean_idx]]
    else:
        x_point_clouds = point_clouds
    dtw_results = []
    avg_distances = []
    tasks = []
    results = dict()
    for i, pi in enumerate(x_point_clouds):
        avg_distances.append(0)
        dtw_results.append([])
        t = run_dtw_coroutine(pool, (i, point_clouds), results)
        tasks.append(t)
    asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks))
    best_idx = 0
    best_d = np.inf
    print("best index", best_idx, len(point_clouds[best_idx]))
    for idx, result in results.items():
        if result[0] < best_d:
            best_idx = idx
    return results[best_idx][1]
