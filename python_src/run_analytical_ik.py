import os
from .morphablegraphs.animation_data import BVHReader, SkeletonBuilder, MotionVector
from .morphablegraphs.animation_data.skeleton_models import RAW_SKELETON_MODEL
from python_src.morphablegraphs.animation_data.motion_editing.motion_grounding import MotionGrounding
from python_src.morphablegraphs.motion_generator.algorithm_configuration import DEFAULT_ALGORITHM_CONFIG
from .morphablegraphs.animation_data.motion_editing.utils import add_heels_to_skeleton

LEFT_FOOT = "LeftFoot"
RIGHT_FOOT = "RightFoot"
RIGHT_TOE = "RightToeBase"
LEFT_TOE = "LeftToeBase"
RIGHT_KNEE = "RightLeg"
LEFT_KNEE = "LeftLeg"
RIGHT_HIP = "RightUpLeg"
LEFT_HIP = "LeftUpLeg"


def run_motion_editing(bvh_file):
    ik_chains = RAW_SKELETON_MODEL["ik_chains"]
    #right foot 98-152
    # , 167, 249, 330, 389
    bvh = BVHReader(bvh_file)
    #animated_joints = list(bvh.get_animated_joints())
    skeleton = SkeletonBuilder().load_from_bvh(bvh)# filter here
    mv = MotionVector()
    mv.from_bvh_reader(bvh, True) # filter here
    config = DEFAULT_ALGORITHM_CONFIG
    me = MotionGrounding(skeleton, config["inverse_kinematics_settings"], ik_chains)
    position = [10, 130, -40]
    #position = [10, 20, -40]
    direction = [1,0,0]
    me.add_constraint("RightHand", [0,100], position, direction)
    mv.frames = me.run(mv)
    print("export motion")
    mv.frames = skeleton.add_fixed_joint_parameters_to_motion(mv.frames)
    mv.export(skeleton, "out" + os.sep + "out")

if __name__ == "__main__":
    bvh_file = "foot_sliding_example.bvh"
    run_motion_editing(bvh_file)
