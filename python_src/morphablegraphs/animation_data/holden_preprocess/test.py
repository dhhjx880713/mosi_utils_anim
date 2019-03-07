
# import helpers.motion.BVH as BVH
# import helpers.motion.Animation as Animation
# from helpers.motion.Quaternions import Quaternions
# from helpers.motion.Pivots import Pivots
import scipy.ndimage.filters as filters


import morphablegraphs.animation_data.holden_preprocess.motion.BVH as BVH
import morphablegraphs.animation_data.holden_preprocess.motion.Animation as AnimationHolden
from morphablegraphs.animation_data.holden_preprocess.motion.Quaternions import Quaternions
from morphablegraphs.animation_data.holden_preprocess.motion.Pivots import Pivots


from morphablegraphs.animation_data.bvh import BVHReader
from morphablegraphs.animation_data.skeleton import Skeleton
from morphablegraphs.animation_data.skeleton_builder import SkeletonBuilder
from morphablegraphs.animation_data.quaternion_frame import QuaternionFrame
from morphablegraphs.animation_data.motion_vector import MotionVector

from morphablegraphs.animation_data.utils import convert_quaternion_frame_to_cartesian_frame, convert_euler_frames_to_quaternion_frames

from morphablegraphs.animation_data.SkeletonWrapper import Animation

import morphablegraphs.animation_data.utils as utils
import math

import numpy as np
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def load_data_bak(filename):
	bvhreader = BVHReader(filename)
	animated_joints = list(bvhreader.get_animated_joints())
	skelBuild = SkeletonBuilder()
	skel = skelBuild.load_from_bvh(bvhreader, animated_joints)
	mv = MotionVector()
	mv.from_bvh_reader(bvhreader)
	mv.skeleton = skel
	return mv

def load_src_motion(filename):
	bvh_reader = BVHReader(filename)
	motion_vector = MotionVector()
	motion_vector.from_bvh_reader(bvh_reader, False)
	animated_joints = list(bvh_reader.get_animated_joints())
	motion_vector.skeleton = SkeletonBuilder().load_from_bvh(bvh_reader, animated_joints=animated_joints)
	return motion_vector



#import cProfile, pstats
import json

def fail(text):
	print(bcolors.FAIL, text, bcolors.ENDC)

def success(text):
	print(bcolors.OKGREEN, text, bcolors.ENDC)


def twistRotations(anim, global_rotations):
	twists = np.array([0.0] * global_rotations.shape[0] * global_rotations.shape[1]).reshape(global_rotations.shape[0], global_rotations.shape[1])
	for f in range(len(global_rotations)):
		for b in range(len(global_rotations[f])):
			q = Quaternions(global_rotations[f][b])
			base_direction = np.array(q * anim.directions[b])[0]
			(swing, twist) = q.swing_twist_decomposition(base_direction)
			twist_angle = twist.angle_axis()[0][0]
			twists[f][b] = twist_angle
	return twists

#quatFrames = convert_euler_frames_to_quaternion_frames(bvhreader, mv.frames)
def test_compare_holden_SkeletonWrapper():
	data = "LocomotionFlat04_000.bvh"

	animHolden, names, _ = BVH.load(data)
	global_xforms = AnimationHolden.transforms_global(animHolden)

	print("holden loaded")

	anim = Animation(data, False)
	globals = anim.get_global_transforms()

	print("mine loaded")

	#danke
	if not np.all(np.fabs(globals - global_xforms) < 0.00001):
		fail("error: not equal global matrices")
	else:
		success("success: equal global matrices")


	global_positions = global_xforms[:, :, :3, 3] / global_xforms[:, :, 3:, 3]
	glob_jp = anim.get_global_joint_positions()
	if not np.all(np.fabs(global_positions - glob_jp) < 0.00001):
		fail("error: not equal global positions")
	else:
		success("success: equal global positions")

	global_rotations = Quaternions.from_transforms(global_xforms)

	my_global_rot = anim.get_global_joint_rotations()
	my_global_rot = utils.quaternions_from_matrix(my_global_rot)#np.array([np.array([utils.quaternion_normalized(utils.quaternion_from_matrix3(m, True)) for m in f]) for f in my_global_rot])

	if not np.all(np.fabs(np.array(global_rotations)- my_global_rot) < 0.001):
		fail("error: not equal global rotations")
	else:
		success("success: equal global rotations")


	sdr_l, sdr_r, hip_l, hip_r = 18, 25, 2, 7
	across = (
			(global_positions[:, sdr_l] - global_positions[:, sdr_r]) +
			(global_positions[:, hip_l] - global_positions[:, hip_r]))
	across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
	direction_filterwidth = 20
	forward = filters.gaussian_filter1d(
		np.cross(across, np.array([[0, 1, 0]])), direction_filterwidth, axis=0, mode='nearest')
	forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]
	root_rotation = Quaternions.between(forward,
										np.array([[0, 0, 1]]).repeat(len(forward), axis=0))[:, np.newaxis]
	root_rotation.normalize()
	f,r = anim.get_forward_directions_rotations()
	if not np.all(np.fabs(f - forward) < 0.00001):
		fail ("error: not equal forward vectors")
	else:
		success ("success: equal forward vectors")
	r2 = np.reshape(r, (r.shape[0], 1, 3, 3))
	r_ = utils.quaternions_from_matrix(np.reshape(r, (r.shape[0], 1, 3, 3)))#np.array([utils.quaternion_normalized(utils.quaternion_from_matrix3(f, True)) for f in r])
	if not np.all(np.fabs(np.array(root_rotation) - r_) < 0.001):
		fail ("error: not equal root rotations")
		if not np.all(np.fabs(np.array(root_rotation)**2 - r_**2) < 0.001):
			fail("		and not equal squared root rotations")
		else:
			success("		but equal squared root rotations")
	else:
		success ("success: equal root rotations")


	""" Local Space """

	# root_relevative positions
	local_positions = global_positions.copy()
	local_positions[:, :, 0] = local_positions[:, :, 0] - local_positions[:, 0:1, 0]
	local_positions[:, :, 2] = local_positions[:, :, 2] - local_positions[:, 0:1, 2]

	# rotate around root_rotation to make forward (0,0,1)
	local_positions = root_rotation[:-1] * local_positions[:-1]
	local_velocities = root_rotation[:-1] * (global_positions[1:] - global_positions[:-1])

	lp = anim.get_root_local_joint_position()
	if not np.all(np.fabs(np.reshape(lp, (lp.shape[0], lp.shape[1], 3)) - local_positions) < 0.00001):
		fail ("error: not equal local positions")
	else:
		success("success: equal local positions")

	lv = anim.get_root_local_joint_velocity()
	if not np.all(np.fabs(np.reshape(lv, (lv.shape[0], lv.shape[1], 3)) - local_velocities) < 0.00001):
		fail ("error: not equal local velocities")
	else:
		success("success: equal local velocities")

	local_rotations = root_rotation[:-1] * global_rotations[:-1]
	local_rotations = local_rotations.normalized()


	my_local_rot = anim.get_root_local_joint_rotations()
	my_local_rot = utils.quaternions_from_matrix(my_local_rot)#np.array([np.array([utils.quaternion_normalized(utils.quaternion_from_matrix3(m, True)) for m in f]) for f in my_local_rot])


	if not np.all(np.fabs(np.array(local_rotations)- my_local_rot) < 0.001):
		fail ("error: not equal local rotations")
		if not np.all(np.fabs(np.array(local_rotations)**2- my_local_rot**2) < 0.001):
			fail("		and not equal squared local rotations")
		else:
			success("		but equal squared local rotations")
	else:
		success("success: equal local rotations")

	twists = twistRotations(animHolden, local_rotations)
	twists[twists > math.pi] -= 2 * math.pi

	tw = anim.get_joint_swing_twist_rotation()

	# if not np.all(np.fabs(forward - animHolden.directions) < 0.00001):
	# 	fail ("error: not equal base directions")
	# else:
	# 	success("success: equal base directions")


	if not np.all(np.fabs(twists - tw) < 0.001):
		fail ("error: not equal local twists")
		if not np.all(np.fabs(twists**2 - tw**2) < 0.001):
			fail("		and not equal squared local twists")
		else:
			success("		but equal squared local twists")
	else:
		success("success: equal local twists")

	root_velocity = root_rotation[:-1] * (global_positions[1:, 0:1] - global_positions[:-1, 0:1])
	root_rvelocity = Pivots.from_quaternions(root_rotation[1:] * -root_rotation[:-1]).ps

	rot_rvel = anim.get_root_rotational_velocity()
	rot_vel = anim.get_root_velocity()

	if not np.all(np.fabs(root_velocity - rot_vel) < 0.001):
		fail ("error: not equal root vel")
	else:
		success("success: equal root vel")
	if not np.all(np.fabs(root_rvelocity - rot_rvel) < 0.001):
		fail ("error: not equal root rvel")
	else:
		success("success: equal root rvel")




	print(anim)

if __name__ =="__main__":
	test_compare_holden_SkeletonWrapper()