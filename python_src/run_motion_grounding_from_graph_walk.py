import json
from morphablegraphs.motion_generator.motion_editing import MotionGrounding, get_average_joint_position, get_average_joint_direction
from morphablegraphs.motion_generator.motion_editing.motion_grounding import IKConstraintSet
from morphablegraphs.motion_generator.motion_editing import FootplantConstraintGenerator
from morphablegraphs.motion_model import MotionStateGraphLoader
from morphablegraphs.motion_generator.algorithm_configuration import AlgorithmConfigurationBuilder
from morphablegraphs.animation_data import BVHReader, Skeleton, MotionVector
from morphablegraphs.motion_generator.motion_editing.constants import IK_CHAINS_RAW_SKELETON
from morphablegraphs.motion_generator.graph_walk import GraphWalk
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




def run_motion_grounding(motion_graph_file, graph_walk_file):
    graph_walk_data = None
    loader = MotionStateGraphLoader()
    loader.set_data_source(motion_graph_file)
    graph = loader.build()
    with open(graph_walk_file) as in_file:
        graph_walk_data = json.load(in_file)
    if graph is None or graph_walk_data is None:
        return
    graph_walk = GraphWalk.from_json(graph, graph_walk_data)
    skeleton = graph.skeleton

    config = AlgorithmConfigurationBuilder().build()
    ik_chains = IK_CHAINS_RAW_SKELETON
    me = MotionGrounding(skeleton, config["inverse_kinematics_settings"], ik_chains, use_analytical_ik=True)
    footplant_settings = {"left_foot": "LeftFoot",
                          "right_foot": "RightFoot",
                          "left_toe": "LeftToeBase",
                          "right_toe": "RightToeBase",
                          "window": 35, "tolerance": 1}

    mv = graph_walk.convert_to_annotated_motion()

    constraint_generator = FootplantConstraintGenerator(skeleton, footplant_settings, ground_height=0, add_heels=True)
    #constraints, blend_ranges = constraint_generator.generate(mv)
    constraints, blend_ranges = constraint_generator.generate_from_graph_walk(mv)
    me.set_constraints(constraints)
    for joint_name in blend_ranges.keys():
        ik_chain = ik_chains[joint_name]
        joint_names = ["Hips"] + [ik_chain["root"], ik_chain["joint"], joint_name]
        for frame_range in blend_ranges[joint_name]:
            me.add_blend_range(joint_names, frame_range)


    mv.frames = me.run(mv)
    print "export motion"
    mv.export("out\\foot_sliding", "out", True)

if __name__ == "__main__":
    motion_graph_file = r"E:\projects\unity integration\model_data\motion_primitives_quaternion_PCA95_blender_1.2"
    graph_walk_file = "graph_walk.data"
    run_motion_grounding(motion_graph_file, graph_walk_file)
