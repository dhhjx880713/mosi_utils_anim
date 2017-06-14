import numpy as np
from morphablegraphs.motion_generator.motion_editing import MotionGrounding, get_average_joint_position, get_average_joint_direction
from morphablegraphs.motion_generator.motion_editing.motion_grounding import IKConstraintSet
from morphablegraphs.motion_generator.algorithm_configuration import AlgorithmConfigurationBuilder
from morphablegraphs.animation_data import BVHReader, Skeleton, MotionVector
from morphablegraphs.motion_generator.motion_editing.constants import IK_CHAINS_RAW_SKELETON

LEFT_FOOT = "LeftFoot"
RIGHT_FOOT = "RightFoot"
RIGHT_TOE = "RightToeBase"
LEFT_TOE = "LeftToeBase"
RIGHT_KNEE = "RightLeg"
LEFT_KNEE = "LeftLeg"
RIGHT_HIP = "RightUpLeg"
LEFT_HIP = "LeftUpLeg"



def create_foot_plant_constraints_orig(skeleton, mv, me, joint_names, frame_range):
    positions = []
    for joint_name in joint_names:
        avg_p = get_average_joint_position(skeleton, mv.frames, joint_name, frame_range[0], frame_range[1])
        positions.append(avg_p)
    c = IKConstraintSet(frame_range, joint_names, positions)
    for idx in xrange(frame_range[0], frame_range[1]):
        if idx not in me._constraints.keys():
            me._constraints[idx] = []
        me._constraints[idx].append(c)
    return me


def create_foot_plant_constraints(skeleton, mv, me, joint_names, start_frame, end_frame):
    """ create a constraint based on the average position in the frame range"""
    for joint_name in joint_names:
        avg_p = get_average_joint_position(skeleton, mv.frames, joint_name, start_frame, end_frame)
        print joint_name, avg_p
        for idx in xrange(start_frame, end_frame):
            me.add_constraint(joint_name,(idx, idx + 1), avg_p)
    return me


def create_foot_plant_constraints2(skeleton, mv, me, joint_name, start_frame, end_frame):
    """ create a constraint based on the average position in the frame range"""

    avg_p = get_average_joint_position(skeleton, mv.frames, joint_name, start_frame, end_frame)
    avg_direction = None
    if len(skeleton.nodes[joint_name].children) > 0:
        child_joint_name = skeleton.nodes[joint_name].children[0].node_name
        avg_direction = get_average_joint_direction(skeleton, mv.frames, joint_name, child_joint_name, start_frame, end_frame)
    print joint_name, avg_p, avg_direction
    for idx in xrange(start_frame, end_frame):
        me.add_constraint(joint_name,(idx, idx + 1), avg_p, avg_direction)
    return me

def run_motion_editing(bvh_file):
    #right foot 98-152
    # , 167, 249, 330, 389
    bvh = BVHReader(bvh_file)
    animated_joints = list(bvh.get_animated_joints())
    skeleton = Skeleton()
    skeleton.load_from_bvh(bvh, animated_joints) # filter here
    mv = MotionVector()
    mv.from_bvh_reader(bvh) # filter here
    config = AlgorithmConfigurationBuilder().build()
    ik_chains = IK_CHAINS_RAW_SKELETON
    me = MotionGrounding(skeleton, config["inverse_kinematics_settings"], ik_chains)
    #free_joints = ["Hips","RightUpLeg", "LeftUpLeg", "RightLeg", "LeftLeg"]
    #me._ik.set_free_joints(free_joints)
    #frame_range = 98, 152
    #me = create_foot_plant_constraints2(skeleton, mv, me, [RIGHT_FOOT, RIGHT_TOE], frame_range)
    start_frame = 98
    end_frame = 140
    me = create_foot_plant_constraints2(skeleton, mv, me, RIGHT_FOOT, start_frame, end_frame)
    # problem you need to blend the hips joint otherwise it does not work, which is not really a good thing to do because it influences the entire body
    me.add_blend_range([RIGHT_FOOT,RIGHT_KNEE, RIGHT_HIP, "Hips"], (start_frame,end_frame-1))#the interpolation range must start at end_frame-1 because this is the last modified frame

    mv.frames = me.run(mv)
    print "export motion"
    #mv.frames = skeleton.complete_motion_vector_from_reference(mv.frames)
    mv.export(skeleton, "out\\foot_sliding", "out")

if __name__ == "__main__":
    bvh_file = "skeleton.bvh"
    bvh_file = "foot_sliding_example.bvh"
    #bvh_file = "no_blending.bvh"
    run_motion_editing(bvh_file)
