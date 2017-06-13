import numpy as np
from morphablegraphs.motion_generator.motion_editing import MotionGrounding, get_average_joint_position
from morphablegraphs.motion_generator.motion_editing.numerical_ik_exp import IKConstraintSet
from morphablegraphs.motion_generator.algorithm_configuration import AlgorithmConfigurationBuilder
from morphablegraphs.animation_data import BVHReader, Skeleton, MotionVector

LEFT_FOOT = "LeftFoot"
RIGHT_FOOT = "RightFoot"
RIGHT_TOE = "RightToeBase"
LEFT_TOE = "LeftToeBase"
RIGHT_KNEE = "RightLeg"
LEFT_KNEE = "LeftLeg"
RIGHT_HIP = "RightUpLeg"
LEFT_HIP = "LeftUpLeg"



def run_motion_editing(bvh_file):
    #right foot 98-152
    # , 167, 249, 330, 389
    bvh = BVHReader(bvh_file)
    #animated_joints = list(bvh.get_animated_joints())
    skeleton = Skeleton()
    skeleton.load_from_bvh(bvh ) # filter here
    mv = MotionVector()
    mv.from_bvh_reader(bvh, True) # filter here
    config = AlgorithmConfigurationBuilder().build()
    me = MotionGrounding(skeleton, config["inverse_kinematics_settings"])
    position = [-20,130,-40]
    me.add_constraint("LeftHand", position, [0,100])
    mv.frames = me.run(mv)
    print "export motion"
    mv.frames = skeleton.complete_motion_vector_from_reference(mv.frames)
    mv.export(skeleton, "out", "out")

if __name__ == "__main__":
    bvh_file = "foot_sliding_example.bvh"
    run_motion_editing(bvh_file)
