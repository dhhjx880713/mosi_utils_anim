"""
 https://www.sciencedirect.com/science/article/pii/S1524070311000178?via%3Dihub

 from pseudocode by Renzo Poddighe
 https://project.dke.maastrichtuniversity.nl/robotlab/wp-content/uploads/Bachelor-thesis-Renzo-Poddighe.pdf
"""
import numpy as np
from mg_analysis.External.transformations import quaternion_inverse, quaternion_multiply, quaternion_from_matrix


def quaternion_from_vector_to_vector(a, b):
    """src: http://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
    http://wiki.ogre3d.org/Quaternion+and+Rotation+Primer"""

    v = np.cross(a, b)
    w = np.sqrt((np.linalg.norm(a) ** 2) * (np.linalg.norm(b) ** 2)) + np.dot(a, b)
    q = np.array([w, v[0], v[1], v[2]])
    if np.dot(q,q) != 0:
        return q/ np.linalg.norm(q)
    else:
        idx = np.nonzero(a)[0]
        q = np.array([0, 0, 0, 0])
        q[1 + ((idx + 1) % 2)] = 1 # [0, 0, 1, 0] for a rotation of 180 around y axis
        return q

def to_local_cos(skeleton, node_name, frame, q):
    # bring into parent coordinate system
    pm = skeleton.nodes[node_name].get_global_matrix(frame)[:3,:3]
    inv_p = quaternion_inverse(quaternion_from_matrix(pm))
    inv_p /= np.linalg.norm(inv_p)
    return quaternion_multiply(inv_p, q)


class FABRIK(object):
    def __init__(self, skeleton, tolerance=0.01, max_iter=500):
        self.skeleton = skeleton
        self.tolerance = tolerance
        self.max_iter = max_iter

    def run(self, frame, target):
        self.skeleton.clear_cached_global_matrices()
        points = []
        for idx, node in enumerate(self.skeleton.nodes):
            print(idx, node)
            p = self.skeleton.nodes[node].get_global_position(frame, use_cache=True)
            points.append(p)
        new_points = self.perform_fabrik(points, target)
        new_frame = self.get_joint_parameters(new_points)
        #print(frame)
        #print(new_frame)
        return new_frame

    def perform_fabrik(self, points, target):
        n_points = len(points)
        distances = []
        for idx in range(0, n_points-1):
            d = np.linalg.norm(points[idx+1] - points[idx])
            distances.append(d)

        root_pos = np.array(points[0])
        dist = np.linalg.norm(target- root_pos)
        chain_length = np.sum(distances, axis=0)
        if dist > chain_length:
            print("unreachable", dist, chain_length)
            points = self.perform_fabrik_unreachable(points, distances, target)
        else:
            print("reachable")
            # if reachable perform forward and backward reaching until tolerance is reached
            iter = 0
            distance = np.linalg.norm(points[-1] - target)
            while distance > self.tolerance and iter< self.max_iter:
                #print(points)
                # backward
                points[-1] = np.array(target)
                for p_idx in range(n_points - 2, 0, -1):
                    r = np.linalg.norm(points[p_idx + 1] - points[p_idx])
                    if r > 0:
                        l = distances[p_idx] / r
                        points[p_idx] = (1 - l) * points[p_idx + 1] + l * points[p_idx]
                #print(points)
                # foward reaching
                points[0] = root_pos
                for p_idx in range(0, n_points-1, 1):
                    r = np.linalg.norm(points[p_idx + 1] - points[p_idx])
                    if r > 0:
                        l = distances[p_idx] / r
                        points[p_idx + 1] = l * points[p_idx + 1] + (1 - l) * points[p_idx]
                iter+=1
                distance = np.linalg.norm(points[-1] - target)
                print("iter",iter, distance)

        return points

    def perform_fabrik_unreachable(self, points, distances, target):
        # if unreachable orient joints to target
        # forward reaching
        n_points = len(points)
        for p_idx in range(0, n_points-1):
            r = np.linalg.norm(target - points[p_idx])
            l = distances[p_idx] / r
            points[p_idx+1] = (1 - l) * points[p_idx] + l * target
        print(target,points)
        return points

    def perform_ccd(self, points, target):
        # print("get parameters", points[0])
        frame = np.zeros(len(self.skeleton.animated_joints) * 4 + 3)
        o = 3
        prev_point = np.array([0, 0, 0])
        for idx, node in enumerate(self.skeleton.animated_joints):
            # print(node)
            target_dir = target - prev_point
            offset = np.array(self.skeleton.nodes[node].children[0].offset)
            target_len = np.linalg.norm(target_dir)
            if target_len > 0:  # FIXME workaround for exception
                # print(idx, target, offset, points[idx+1],prev_point)
                target_dir /= target_len
                offset /= np.linalg.norm(offset)
                q = quaternion_from_vector_to_vector(offset, target_dir)
                # print(q)
                frame[o:o + 4] = to_local_cos(self.skeleton, node, frame, q)
            else:
                frame[o:o + 4]
            prev_point = points[idx + 1]
            o += 4
        return frame

    def get_joint_parameters(self, points):
        #print("get parameters", points[0])
        frame = np.zeros(len(self.skeleton.animated_joints)*4+3)
        o = 3
        prev_point = np.array([0,0,0])
        for idx, node in enumerate(self.skeleton.animated_joints):
            #print(node)
            target = points[idx+1]-prev_point
            offset = np.array(self.skeleton.nodes[node].children[0].offset)
            target_len = np.linalg.norm(target)
            if target_len > 0:# FIXME workaround for exception
                #print(idx, target, offset, points[idx+1],prev_point)
                target /= target_len
                offset /= np.linalg.norm(offset)
                q = quaternion_from_vector_to_vector(offset, target)
                # print(q)
                frame[o:o+4] = to_local_cos(self.skeleton, node, frame, q)
            else:
                frame[o:o+4] = [1,0,0,0]
            prev_point = points[idx+1]
            o += 4
        return frame
