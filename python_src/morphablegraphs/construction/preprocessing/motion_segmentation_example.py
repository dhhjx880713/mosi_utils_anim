# encoding: UTF-8

from motion_segmentation import MotionSegmentation
# from motion_normalization import MotionNormalization
from morphablegraphs.animation_data import BVHReader, BVHWriter, Skeleton
from morphablegraphs.motion_analysis


def motion_cutting():
    elementary_action = 'screw'
    motion_primitive = 'retrieve'
    normalizer = MotionNormalization()
    retarget_folder = r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\Take_demotakes\screw'
    annotation_file = r'C:\repo\data\1 - MoCap\2 - Rocketbox retargeting\Take_demotakes\screw\key_frame_annotation.txt'
    normalizer.segment_motions(elementary_action,
                               motion_primitive,
                               retarget_folder,
                               annotation_file)
    normalizer.save_segments(
        r'C:\repo\data\1 - MoCap\3 - Cutting\elementary_action_screw\retrieve_new')


def feature_dection():
    test_file1 = r'C:\repo\data\1 - MoCap\1 - Rawdata\Take_walk\walk_001_1.bvh'
    test_file2 = r'C:\repo\data\1 - MoCap\1 - Rawdata\Take_walk\XYZ_rotation\walk_001_1.bvh'
    test_bvh1 = BVHReader(test_file1)
    test_bvh2 = BVHReader(test_file2)
    motion1 = Skeleton()
    motion1.load_from_bvh(test_bvh1)
    motion2 = Skeleton()
    motion2.load_from_bvh(test_bvh2)
    left_toe_pos1 = []
    left_toe_pos2 = []
    n_frames = len(test_bvh1)
    target_node = 'LeftToeBase'
    for i in range(n_frames):
        pos1 = motion1.nodes[target_node].get_global_position_from_euler(test_bvh1.frames[0])
        pos2 = motion2.nodes[target_node].get_global_position_from_euler(test_bvh2.frames[0])
        print(pos1)
        print(pos2)
        raw_input()


if __name__ == "__main__":
    # motion_cutting()
    feature_dection()