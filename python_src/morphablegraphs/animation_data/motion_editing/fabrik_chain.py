"""
 https://www.sciencedirect.com/science/article/pii/S1524070311000178?via%3Dihub

 from pseudocode by Renzo Poddighe
 https://project.dke.maastrichtuniversity.nl/robotlab/wp-content/uploads/Bachelor-thesis-Renzo-Poddighe.pdf
"""
import numpy as np
from ...external.transformations import quaternion_inverse, quaternion_multiply, quaternion_from_matrix


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


class FABRIKChain(object):
    def __init__(self, skeleton, joints, effectors, tolerance=0.01, max_iter=500):
        self.skeleton = skeleton
        self.joints = joints
        self.effectors = effectors
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.positions = []
        self.distances = []
        self.target = None
        self.root_pos = None


    def set_positions_from_frame(self, frame, parent_length):
        self.skeleton.clear_cached_global_matrices()
        self.positions = []
        for idx, node in enumerate(self.effectors):
            print(idx, node)
            p = self.skeleton.nodes[node].get_global_position(frame, use_cache=True)
            self.positions.append(p)
            #print("pos ", node, p)

        self.n_points = len(self.positions)
        self.distances = []
        for idx in range(0, self.n_points - 1):
            d = np.linalg.norm(self.positions[idx + 1] - self.positions[idx])
            self.distances.append(d)
            #print("length ", self.effectors[idx], d)

        self.parent_length = parent_length
        self.chain_length = np.sum(self.distances, axis=0)
        self.root_pos = np.array(self.positions[0])

    def target_is_reachable(self):
        dist = np.linalg.norm(self.target - self.root_pos)
        #print("unreachable", dist, self.chain_length)
        return dist < self.chain_length+ self.parent_length


    def run(self, frame, target):
        self.target = target
        self.set_positions_from_frame(frame, 0)

        self.solve()
        new_frame = self.get_joint_parameters()
        return new_frame

    def run_with_constraints(self, frame, target):
        self.target = target
        self.set_positions_from_frame(frame, 0)
        self.solve_with_constraints()
        new_frame = self.get_joint_parameters()
        return new_frame

    def solve(self):
        if not self.target_is_reachable():
            print("unreachable")
            # if unreachable orient joints to target
            self.orient_to_target()
        else:
            print("reachable")
            # if reachable perform forward and backward reaching until tolerance is reached
            iter = 0
            distance = self.get_error()
            while distance > self.tolerance and iter< self.max_iter:
                self.backward()
                self.forward()
                iter+=1
                distance = self.get_error()
                print("iter",iter, distance)
        #print(self.positions)
        return self.positions

    def solve_with_constraints(self):
        if not self.target_is_reachable():
            print("unreachable")
            # if unreachable orient joints to target
            self.orient_to_target()
        else:
            print("reachable")
            # if reachable perform forward and backward reaching until tolerance is reached
            iter = 0
            distance = self.get_error()
            while distance > self.tolerance and iter < self.max_iter:
                self.backward()
                self.forward()
                if iter % 10 == 0:
                    self.apply_constraints_global()
                iter += 1
                distance = self.get_error()
                print("iter", iter, distance)
        # print(self.positions)
        return self.positions

    def apply_constraints(self):
        frame = self.get_joint_parameters()
        o = 3
        for idx, n in enumerate(self.joints):
            if self.skeleton.nodes[n].joint_constraint is not None:
                q = frame[o:o+4]
                frame[o:o + 4] = self.skeleton.nodes[n].joint_constraint.apply(q)
            o+=4

        self.set_positions_from_frame(frame, 0)


    def apply_constraints_global(self):
        frame = self.get_joint_parameters()
        o = 3
        for idx, n in enumerate(self.joints):
            if self.skeleton.nodes[n].joint_constraint is not None:
                pm = self.skeleton.nodes[n].parent.get_global_matrix(frame)[:3,:3]
                m = self.skeleton.nodes[n].get_global_matrix(frame, True)
                q = quaternion_from_matrix(m)
                q = self.skeleton.nodes[n].joint_constraint.apply_global(pm, q)
                frame[o:o + 4] = to_local_cos(self.skeleton, n, frame, q)
            o += 4
        self.set_positions_from_frame(frame, 0)


    def get_error(self):
        #print("ERROR",self.positions[-1], self.target)
        return np.linalg.norm(self.positions[-1] - self.target)

    def orient_to_target(self):
        for p_idx in range(0, self.n_points - 1):
            r = np.linalg.norm(self.target - self.positions[p_idx])
            l = self.distances[p_idx] / r
            self.positions[p_idx + 1] = (1 - l) * self.positions[p_idx] + l * self.target
        #print(self.target, self.positions)

    def backward(self):
        self.positions[-1] = np.array(self.target)
        for p_idx in range(self.n_points - 2, -1, -1):
            r = np.linalg.norm(self.positions[p_idx + 1] - self.positions[p_idx])
            if r > 0:
                l = self.distances[p_idx] / r
                self.positions[p_idx] = (1 - l) * self.positions[p_idx + 1] + l * self.positions[p_idx]
        #print("after backward", self.positions[0], self.positions[1])

    def forward(self):
        self.positions[0] = self.root_pos
        for p_idx in range(0, self.n_points - 1, 1):
            r = np.linalg.norm(self.positions[p_idx + 1] - self.positions[p_idx])
            if r > 0:
                l = self.distances[p_idx] / r
                self.positions[p_idx + 1] = l * self.positions[p_idx + 1] + (1 - l) * self.positions[p_idx]

    def get_joint_parameters(self):
        #print("get parameters", points[0])
        frame = np.zeros(len(self.joints)*4+3)
        o = 3
        prev_point = np.array([0,0,0])
        for idx, node in enumerate(self.joints):
            #print(node)
            target = self.positions[idx+1]-prev_point
            local_offset = np.array(self.skeleton.nodes[node].children[0].offset)
            target_len = np.linalg.norm(target)
            if target_len > 0:# FIXME workaround for exception
                #print(idx, target, offset, points[idx+1],prev_point)
                target /= target_len
                local_offset /= np.linalg.norm(local_offset)
                # 1. get global rotation
                q = quaternion_from_vector_to_vector(local_offset, target)
                # 2. bring global rotation into local coordinate system
                frame[o:o+4] = to_local_cos(self.skeleton, node, frame, q)
            else:
                frame[o:o+4] = [1,0,0,0]
            prev_point = self.positions[idx+1]
            o += 4
        return frame

    def get_joint_parameters_global(self):
        frame = np.zeros(len(self.joints)*4+3)
        o = 3
        #prev_point = np.array(self.root_pos)
        for idx in range(0, len(self.joints)-1):
            node = self.joints[idx]
            offset = np.array(self.skeleton.nodes[node].children[0].offset)
            offset /= np.linalg.norm(offset)
            if idx == 0:
                dir_vector = self.positions[idx+1]
                dir_vector_len = np.linalg.norm(dir_vector)
                if dir_vector_len > 0 and np.linalg.norm(offset) > 0:  # FIXME workaround for exception
                    # print(idx, target, offset, self.positions[idx+1],prev_point)
                    dir_vector /= dir_vector_len
                    q = quaternion_from_vector_to_vector(offset, dir_vector)
                    frame[o:o + 4] = q
                    #print("global", node, q)
                else:
                    print("work around", offset,dir_vector_len, node)
                    frame[o:o + 4] = [1, 0, 0, 0]
            else:
                #if idx+1 < len(self.positions):
                dir_vector = self.positions[idx+1] - self.positions[idx]
                #else:# use target pose
                #dir_vector = target_pos- self.positions[idx]
                dir_vector_len = np.linalg.norm(dir_vector)
                if dir_vector_len > 0 and np.linalg.norm(offset) > 0:# FIXME workaround for exception
                    #print(idx, target, offset, self.positions[idx+1],prev_point)
                    dir_vector /= dir_vector_len
                    q = quaternion_from_vector_to_vector(offset, dir_vector)
                    frame[o:o+4] = q
                    #print("global",node, q)
                else:
                    print("work around", offset,dir_vector_len, node)
                    frame[o:o+4] = [1,0,0,0]
            prev_point = self.positions[idx]
            o += 4
        #print(frame)
        return frame

    def get_global_positions(self):
        position_dict = dict()
        for idx in range(0, len(self.effectors)):
            position_dict[self.effectors[idx]] = self.positions[idx]
        return position_dict