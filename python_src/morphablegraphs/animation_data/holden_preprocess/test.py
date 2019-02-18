
# import helpers.motion.BVH as BVH
# import helpers.motion.Animation as Animation
# from helpers.motion.Quaternions import Quaternions
# from helpers.motion.Pivots import Pivots


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

import numpy as np


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


#quatFrames = convert_euler_frames_to_quaternion_frames(bvhreader, mv.frames)
def run_test():
	data = "LocomotionFlat04_000.bvh"

	# pr = cProfile.Profile()
	# pr.enable()

	animHolden, names, _ = BVH.load(data)
	globalsHolden = AnimationHolden.transforms_global(animHolden)

	# pr.disable()
	# sortby = 'cumulative'
	# ps = pstats.Stats(pr).sort_stats(sortby)
	# print(ps.print_stats())

	global_rotations = Quaternions.from_transforms(globalsHolden)
	print("Holden loaded")

	# pr = cProfile.Profile()
	# pr.enable()

	# anim = Animation(data)
	# globals = anim.get_global_transforms()
	mv = load_src_motion(data)
	skel = mv.skeleton
	globals = []
	for f in range(len(mv.frames)):
		globals.append([])
		for bone_name in skel.get_joint_names():
			globals[f].append(skel.nodes[bone_name].get_global_matrix(mv.frames[f]))



	# pr.disable()
	# sortby = 'cumulative'
	# ps = pstats.Stats(pr).sort_stats(sortby)
	# print(ps.print_stats())


	if not np.all(np.fabs(globals - globalsHolden) < 0.00001):
		print("not equal global matrices")


	with open(data.replace(".bvh",".panim")) as f:
		d = json.load(f)
		frames = d["frames"]
		if not len(frames) == len(globalsHolden):
			print("not same frames with panim")

		for i in range(len(frames)):
			posPanim = [frames[i]["WorldPos"][0]["x"], frames[i]["WorldPos"][0]["y"], frames[i]["WorldPos"][0]["z"]]
			posAnim = globalsHolden[i][0][:3,3]
			if np.all(np.fabs(posPanim - posAnim) < 0.00001):
				print("not equal positions with panim: ", posPanim, posAnim)


	# mv = load_src_motion(data)
	# skel = mv.skeleton
	# print("skel loaded")
	#
	# # assert(skel.get_joint_names() == names)
	# if skel.get_joint_names() != names:
	# 	print("names do not match")
	# 	return
	#
	# print("MV loaded")
	# globtransf = []
	# #globeul = []
	# for bone_name in skel.get_joint_names():
	# 	globtransf.append(skel.nodes[bone_name].get_global_matrix(mv.frames[0]))
	# 	#globeul.append(skel.nodes[bone_name].get_global_matrix_from_euler_frame(mv.frames[0]))
	#
	# # check global positions
	# for i in range(0, len(globtransf)):
	# 	print(globals[0][i][:,3], globtransf[i][:,3])#, globeul[i][:,3])

	#assert(locpos[i] == anim.positions[0][i])

	#cart_frame = convert_quaternion_frame_to_cartesian_frame()

	#qf = QuaternionFrame(bvhreader)
	print(anim)

if __name__ =="__main__":
	run_test()