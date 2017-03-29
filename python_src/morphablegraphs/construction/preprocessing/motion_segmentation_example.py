# encoding: UTF-8

from motion_segmentation import MotionSegmentation
from motion_normalization import MotionNormalization


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


if __name__ == "__main__":
    motion_cutting()