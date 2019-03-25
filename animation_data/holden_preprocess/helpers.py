import numpy as np
import math

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# def softmax(X, theta=1.0, axis=None):
# 	"""
# 	Compute the softmax of each element along an axis of X.
#
# 	Parameters
# 	----------
# 	X: ND-Array. Probably should be floats.
# 	theta (optional): float parameter, used as a multiplier
# 		prior to exponentiation. Default = 1.0
# 	axis (optional): axis to compute values along. Default is the
# 		first non-singleton axis.
#
# 	Returns an array the same size as X. The result will sum to 1
# 	along the specified axis.
# 	"""
#
# 	# make X at least 2d
# 	y = np.atleast_2d(X)
#
# 	# find axis
# 	if axis is None:
# 		axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
#
# 	# multiply y against the theta parameter,
# 	y = y * float(theta)
#
# 	# subtract the max for numerical stability
# 	y = y - np.expand_dims(np.max(y, axis=axis), axis)
#
# 	# exponentiate y
# 	y = np.exp(y)
#
# 	# take the sum along the specified axis
# 	ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
#
# 	# finally: divide elementwise
# 	p = y / ax_sum
#
# 	# flatten if X was 1D
# 	if len(X.shape) == 1: p = p.flatten()
#
# 	return p

def generate_catmul_rom():
	# mapping follows p1
	# (0,1,2,3)
	base1 = np.array([[-0.5,  1.5, -1.5,  0.5], \
					  [   1, -2.5,    2, -0.5], \
					  [-0.5,    0,  0.5,    0], \
					  [   0,    1,    0,    0]])

	# (3, 0, 1, 2)
	base0 = np.array([[ 0.5, -0.5,  1.5, -1.5], \
					  [-0.5,    1, -2.5,    2], \
					  [   0, -0.5,    0,  0.5], \
					  [   0,    0,    1,    0]])
	# (2,3,0,1)
	base3 = np.array([[-1.5,  0.5, -0.5,  1.5], \
					  [   2, -0.5,    1, -2.5], \
					  [ 0.5,    0, -0.5,    0], \
					  [   0,    0,    0,    1]])

	# (1, 2, 3, 0)
	base2 = np.array([[ 1.5, -1.5,  0.5, -0.5], \
					  [-2.5,    2, -0.5,    1], \
					  [   0,  0.5,    0, -0.5], \
					  [   1,    0,    0,    0]])
	## mapping to match p1
	return np.array([base0, base1, base2, base3], dtype=np.float32)

def cubic(y0, y1, y2, y3, mu):
	return (
		(-0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3) * mu * mu * mu +
		(y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3) * mu * mu +
		(-0.5 * y0 + 0.5 * y2) * mu +
		(y1))

def euclidian_length(v):
	s = 0.0
	for a in v:
		s += a * a
	return math.sqrt(s)


# helpers:
def normalize(a):
	length = euclidian_length(a)
	if length < 0.000001:
		return np.array([0,0,0])
	return a / length


def glm_mix(v1, v2, a):
	return np.array(((1 - a) * v1 + a * v2))


def z_angle(v1):
	#v1 = normalize(v1)
	return math.atan2(v1[0], v1[-1])

def mix_directions(v1, v2, a):
	if v1.all(0) and v2.all(0):
		return v1
	v1 = normalize(v1)
	v2 = normalize(v2)
	omega = math.acos(max(min(np.dot(v1, v2), 1), -1))
	sinom = math.sin(omega)
	if sinom < 0.000001:
		return v1
	slerp = math.sin((1-a) * omega) / sinom * v1 + math.sin(a * omega) / sinom * v2
	return normalize(slerp)

# Angle in radians
def rot_around_z_3d(vector, angle, inverse = False):
	mat = np.array([
		[math.cos(angle), 0, math.sin(angle)],
		[0, 1, 0],
		[-math.sin(angle), 0, math.cos(angle)]
	])
	if inverse:
		mat = mat.transpose()
	return np.matmul(mat, vector)

def quat_to_mat(q):
	qr = q[3]
	qi = q[0]
	qj = q[1]
	qk = q[2]
	s = 1
	return np.array([
		[1 - 2 * s * (qj * qj + qk * qk), 2 * s * (qi * qj - qk * qr), 2 * s * (qi * qk + qj * qr)],
		[2 * s * (qi * qj + qk * qr), 1 - 2 * s * (qi * qi + qk * qk), 2 * s * (qj * qk - qi * qr)],
		[2 * s * (qi * qk - qj * qr), 2 * s * (qj * qk + qi * qr), 1 - 2 * s * (qi * qi + qj * qj)]
	])


def mat_to_quat(m):
	# print(m)
	qw = math.sqrt(1.0 + m[0][0] + m[1][1] + m[2][2]) / 2.0
	qx = (m[2][1] - m[1][2]) / (4 * qw)
	qy = (m[0][2] - m[2][0]) / (4 * qw)
	qz = (m[1][0] - m[0][1]) / (4 * qw)
	return np.array((qx, qy, qz, qw))

def global_to_local_pos(pos, root_pos, root_rot):
	return rot_around_z_3d(pos - root_pos, root_rot, inverse=True)  # self.char.joint_positions[i]#
