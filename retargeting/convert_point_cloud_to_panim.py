from mosi_utils_anim.animation_data.panim import Panim
import numpy as np


def convert_point_cloud_to_panim(save_file: str, 
                                 point_cloud: np.ndarray, 
                                 skeleton: dict=None,  
                                 scale: float=1.0, 
                                 unity_format: bool=False):
    """Convert point cloud data to Panim format
    Arguments:
        save_file {str} -- save path
        point_cloud {numpy.array3d} -- n_frames * n_joints * 3

    """
    panim = Panim()
    panim.setMotionData(point_cloud)
    if skeleton is not None:
        panim.setSkeleton(skeleton)
    if unity_format:
        panim.convert_to_unity_format(scale=scale)
    else:
        panim.scale_motion(scale)
    panim.save(save_file)