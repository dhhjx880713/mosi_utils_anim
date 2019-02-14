
# import helpers.motion.BVH as BVH
# import helpers.motion.Animation as Animation
# from helpers.motion.Quaternions import Quaternions
# from helpers.motion.Pivots import Pivots


import morphablegraphs.animation_data.holden_preprocess.motion.BVH as BVH
import morphablegraphs.animation_data.holden_preprocess.motion.Animation as Animation
from morphablegraphs.animation_data.holden_preprocess.motion.Quaternions import Quaternions
from morphablegraphs.animation_data.holden_preprocess.motion.Pivots import Pivots


from morphablegraphs.animation_data.bvh import BVHReader
from morphablegraphs.animation_data.skeleton import Skeleton
from morphablegraphs.animation_data.skeleton_builder import SkeletonBuilder
from morphablegraphs.animation_data.quaternion_frame import QuaternionFrame
from morphablegraphs.animation_data.motion_vector import MotionVector

from morphablegraphs.animation_data.utils import convert_quaternion_frame_to_cartesian_frame, convert_euler_frames_to_quaternion_frames


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




#quatFrames = convert_euler_frames_to_quaternion_frames(bvhreader, mv.frames)
def run_test():
	data = "LocomotionFlat01_000.bvh"

	anim, names, _ = BVH.load(data)
	globals = Animation.transforms_global(anim)
	print("Holden loaded")

	mv = load_src_motion(data)
	skel = mv.skeleton
	print("skel loaded")

	# assert(skel.get_joint_names() == names)
	if skel.get_joint_names() != names:
		print("names do not match")
		return

	print("MV loaded")
	globtransf = []
	#globeul = []
	for bone_name in skel.get_joint_names():
		globtransf.append(skel.nodes[bone_name].get_global_matrix(mv.frames[0]))
		#globeul.append(skel.nodes[bone_name].get_global_matrix_from_euler_frame(mv.frames[0]))

	# check global positions
	for i in range(0, len(globtransf)):
		print(globals[0][i][:,3], globtransf[i][:,3])#, globeul[i][:,3])

	#assert(locpos[i] == anim.positions[0][i])

	#cart_frame = convert_quaternion_frame_to_cartesian_frame()

	#qf = QuaternionFrame(bvhreader)
	print(skel)

if __name__ =="__main__":
	run_test()