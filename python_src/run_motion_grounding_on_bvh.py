from morphablegraphs.animation_data import BVHReader, Skeleton, MotionVector
from morphablegraphs.motion_generator.algorithm_configuration import AlgorithmConfigurationBuilder
from morphablegraphs.animation_data.motion_editing import FootplantConstraintGenerator
from morphablegraphs.animation_data.motion_editing import MotionGrounding
from python_src.morphablegraphs.animation_data.motion_editing.constants import *
from python_src.morphablegraphs.animation_data.motion_editing.motion_grounding import IKConstraintSet
from python_src.morphablegraphs.animation_data.motion_editing.utils import get_average_joint_position, get_average_joint_direction, plot_joint_heights, add_heels_to_skeleton, get_joint_height, \
    save_ground_contact_annotation

LEFT_FOOT = "LeftFoot"
RIGHT_FOOT = "RightFoot"
RIGHT_TOE = "RightToeBase"
LEFT_TOE = "LeftToeBase"
RIGHT_KNEE = "RightLeg"
LEFT_KNEE = "LeftLeg"
RIGHT_HIP = "RightUpLeg"
LEFT_HIP = "LeftUpLeg"
LEFT_HEEL = "LeftHeel"
RIGHT_HEEL = "RightHeel"



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
    avg_direction = None
    for idx in xrange(start_frame, end_frame):
        me.add_constraint(joint_name,(idx, idx + 1), avg_p, avg_direction)
    return me



def run_motion_grounding(bvh_file, skeleton_type):
    skeleton_def = SKELETON_ANNOTATIONS[skeleton_type]
    source_ground_height = 100.0
    target_ground_height = 0.0
    bvh = BVHReader(bvh_file)
    animated_joints = list(bvh.get_animated_joints())
    skeleton = Skeleton()
    skeleton.load_from_bvh(bvh, animated_joints) # filter here
    mv = MotionVector()
    mv.from_bvh_reader(bvh) # filter here
    config = AlgorithmConfigurationBuilder().build()
    me = MotionGrounding(skeleton, config["inverse_kinematics_settings"], skeleton_def, use_analytical_ik=True)
    footplant_settings = {"window": 20, "tolerance": 1, "constraint_range": 10, "smoothing_constraints_window": 15}

    #joint_heights = get_joint_height(skeleton, mv.frames, FOOT_JOINTS)
    #plot_joint_heights(joint_heights)
    skeleton = add_heels_to_skeleton(skeleton, skeleton_def["left_foot"],
                                                 skeleton_def["right_foot"],
                                                 skeleton_def["left_heel"],
                                                 skeleton_def["right_heel"],
                                                 skeleton_def["heel_offset"])
    constraint_generator = FootplantConstraintGenerator(skeleton, skeleton_def, footplant_settings,
                                                        source_ground_height=source_ground_height,
                                                        target_ground_height=target_ground_height)
    constraints, blend_ranges = constraint_generator.generate(mv)
    #plot_constraints(constraints, ground_height)
    me.set_constraints(constraints)

    for joint_name, frame_ranges in blend_ranges.items():
        ik_chain = skeleton_def["ik_chains"][joint_name]
        for frame_range in frame_ranges:
            joint_names = [skeleton.root] + [ik_chain["root"], ik_chain["joint"], joint_name]
            me.add_blend_range(joint_names, tuple(frame_range))
    # problem you need to blend the hips joint otherwise it does not work, which is not really a good thing to do because it influences the entire body

    mv.frames = me.run(mv, target_ground_height)
    print "export motion"

    joint_heights = get_joint_height(skeleton, mv.frames, skeleton_def["foot_joints"])
    plot_joint_heights(joint_heights)
    mv.export(skeleton, "out\\foot_sliding", "out")

if __name__ == "__main__":
    bvh_file = "skeleton.bvh"
    bvh_file = "walk_001_1.bvh"
    #bvh_file = "walk_014_2.bvh"
    bvh_file = "game_engine_left_stance.bvh"
    #bvh_file = "no_blending.bvh"
    skeleton_type = "game_engine"
    run_motion_grounding(bvh_file, skeleton_type)
