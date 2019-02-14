
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



data = "LocomotionFlat01_000.bvh"

anim, names, _ = BVH.load(data)
globals = Animation.transforms_global(anim)
print("Holden loaded")

bvhreader = BVHReader(data)
skelBuild = SkeletonBuilder()
skel = skelBuild.load_from_bvh(bvhreader)
print("skel loaded")

assert(skel.get_joint_names() == names)

mv = MotionVector()
mv.from_bvh_reader(bvhreader)
print("MV loaded")

quatFrames = convert_euler_frames_to_quaternion_frames(bvhreader, mv.frames)

globtransf = []
globeul = []
for bone_name in skel.get_joint_names():
	globtransf.append(skel.nodes[bone_name].get_global_matrix(quatFrames[0]))
	globeul.append(skel.nodes[bone_name].get_global_matrix_from_euler_frame(mv.frames[0]))

# check global positions
for i in range(0, len(globtransf)):
	print(globals[0][i][:,3], globtransf[i][:,3], globeul[i][:,3])

#assert(locpos[i] == anim.positions[0][i])

#cart_frame = convert_quaternion_frame_to_cartesian_frame()

#qf = QuaternionFrame(bvhreader)
print(skel)