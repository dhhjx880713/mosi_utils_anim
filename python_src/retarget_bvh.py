from morphablegraphs.animation_data import Skeleton, MotionVector, BVHReader
from morphablegraphs.animation_data.retargeting import get_targets_from_motion, get_new_frames_from_direction_constraints, ROCKETBOX_TO_GAME_ENGINE_MAP


def retarget(src_skeleton, src_motion, target_skeleton, inv_joint_map=ROCKETBOX_TO_GAME_ENGINE_MAP, frame_range=None, scale_factor=1.0):
    targets = get_targets_from_motion(src_skeleton, src_motion.frames, inv_joint_map)
    new_frames = get_new_frames_from_direction_constraints(target_skeleton, src_skeleton,
                                                            src_motion.frames, targets,
                                                            frame_range=frame_range,
                                                              scale_factor=scale_factor)
    return new_frames


def export(skeleton, frames, out_file):
    mv = MotionVector()
    mv.frames = frames
    mv.export(skeleton, ".", out_file, add_time_stamp=False)


if __name__ == "__main__":
    src_file = "skeleton.bvh"
    target_file = "game_engine_target.bvh"
    out_file = "out"

    src_bvh = BVHReader(src_file)
    target_bvh = BVHReader(target_file)

    src_skeleton = Skeleton()
    src_skeleton.load_from_bvh(src_bvh, add_tool_joints=False)
    src_motion = MotionVector()
    src_motion.from_bvh_reader(src_bvh)

    animated_joints = list(target_bvh.get_animated_joints())
    target_skeleton = Skeleton()
    target_skeleton.load_from_bvh(target_bvh, animated_joints, add_tool_joints=False)

    scale_factor = 1.0/10  # is applied on the root translation of the source
    new_frames = retarget(src_skeleton, src_motion, target_skeleton, scale_factor=scale_factor)
    export(target_skeleton, new_frames, out_file)

