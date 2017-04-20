import numpy as np
from morphablegraphs.animation_data import Skeleton, MotionVector, BVHReader, BVHWriter
from morphablegraphs.animation_data.retargeting import get_targets_from_motion, ROCKETBOX_TO_GAME_ENGINE_MAP
from morphablegraphs.animation_data.retargeting import get_new_frames_from_direction_constraints as get_new_frames_using_quaternion
from morphablegraphs.animation_data.retargeting_euler import get_new_euler_frames_from_direction_constraints as get_new_frames_using_euler
from morphablegraphs.animation_data.fbx_io import load_skeleton_and_animations_from_fbx, export_motion_vector_to_fbx_file


def retarget(src_skeleton, src_motion, target_skeleton, out_file, inv_joint_map=ROCKETBOX_TO_GAME_ENGINE_MAP, additional_rotation_map=None, frame_range=None, scale_factor=1.0, use_optimization=True, use_euler=False):
    targets = get_targets_from_motion(src_skeleton, src_motion.frames, inv_joint_map, additional_rotation_map=additional_rotation_map)
    if not use_euler:
        new_frames = get_new_frames_using_quaternion(target_skeleton, targets,
                                                     frame_range=frame_range,
                                                     scale_factor=scale_factor,
                                                     use_optimization=use_optimization)
        mv = MotionVector()
        mv.frames = new_frames

        mv.export(target_skeleton, ".", out_file, add_time_stamp=False)
    else:
        new_frames = get_new_frames_using_euler(target_skeleton,
                                    targets,
                                    frame_range=frame_range,
                                    scale_factor=scale_factor)
        BVHWriter(out_file, target_skeleton, new_frames, target_skeleton.frame_time)


def load_target_skeleton(file_path):
    skeleton = None
    if file_path.lower().endswith("fbx"):
        skeleton, mvs = load_skeleton_and_animations_from_fbx(file_path)
    elif file_path.lower().endswith("bvh"):
        target_bvh = BVHReader(file_path)
        animated_joints = list(target_bvh.get_animated_joints())
        skeleton = Skeleton()
        skeleton.load_from_bvh(target_bvh, animated_joints, add_tool_joints=False)
    return skeleton

def scale_skeleton(skeleton, scale_factor):
    for node in skeleton.nodes.values():
        node.offset = np.array(node.offset) * scale_factor

if __name__ == "__main__":

    src_file = "export15.bvh"
    target_file = "game_engine_target.bvh"
    out_file = "out_euler"
    frame_range = [0,1]
    use_optimization = False
    use_euler = True
    scale_factor = 10  # is applied on the root translation of the source and the offsets of the skeleton

    src_bvh = BVHReader(src_file)
    src_skeleton = Skeleton()
    src_skeleton.load_from_bvh(src_bvh, add_tool_joints=False)
    src_motion = MotionVector()
    src_motion.from_bvh_reader(src_bvh)

    target_skeleton = load_target_skeleton(target_file)
    if target_skeleton is not None:

        scale_skeleton(target_skeleton, scale_factor)

        retarget(src_skeleton, src_motion, target_skeleton, out_file,
                              inv_joint_map=ROCKETBOX_TO_GAME_ENGINE_MAP,
                              additional_rotation_map=None,
                              frame_range=frame_range,
                              scale_factor=1.0/scale_factor,
                              use_optimization=use_optimization,
                              use_euler=use_euler)
    else:
        print "Error: could not read target skeleton", target_file
