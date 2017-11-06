# encoding: UTF-8
import collections
from .motion_dimension_reduction import MotionDimensionReduction
from ...utilities.io_helper_functions import load_json_file
from ..construction_algorithm_configuration import ConstructionAlgorithmConfigurationBuilder
from ...animation_data import BVHReader, Skeleton
import glob
import os
import sys
sys.path.append(os.path.dirname(__file__))



def get_aligned_data_folder(elementary_action,
                            motion_primitive,
                            repo_dir=None):
    if repo_dir is None:
        repo_dir = r'C:\repo'
    assert os.path.exists(repo_dir), ('Please configure morphablegraph repository directory!')
    data_folder = 'data'
    mocap_folder = '1 - Mocap'
    alignment_folder = '4 - Alignment'
    elementary_action_folder = 'elementary_action_' + elementary_action
    return os.sep.join([repo_dir,
                        data_folder,
                        mocap_folder,
                        alignment_folder,
                        elementary_action_folder,
                        motion_primitive])

def create_pseudo_timewarping(aligned_data_folder):
    bvhfiles = glob.glob(os.path.join(aligned_data_folder, '*.bvh'))
    timewarping_data = {}
    for item in bvhfiles:
        filename = os.path.split(item)[-1]
        bvhreader = BVHReader(item)
        frame_indices = list(range(len(bvhreader.frames)))
        timewarping_data[filename] = frame_indices
    return timewarping_data


def load_aligned_data(elementary_action, motion_primitive):
    aligned_data_folder = get_aligned_data_folder(elementary_action, motion_primitive)
    aligned_motion_data = {}
    timewarping_file = os.path.join(aligned_data_folder, 'timewarping.json')
    if not os.path.exists(timewarping_file):
        print("##############  cannot find timewarping file, create pseudo timewapring data")
        timewarping_data = create_pseudo_timewarping(aligned_data_folder)
    else:
        timewarping_data = load_json_file(timewarping_file)
    bvhfiles = glob.glob(os.path.join(aligned_data_folder, '*.bvh'))
    for filename, time_index in timewarping_data.items():
        aligned_motion_data[filename] = {}
        aligned_motion_data[filename]['warping_index'] = time_index
        filepath = os.path.join(aligned_data_folder, filename)
        assert filepath in bvhfiles, ('cannot find the file in the aligned folder')
        bvhreader = BVHReader(filepath)
        aligned_motion_data[filename]['frames'] = bvhreader.frames
    aligned_motion_data = collections.OrderedDict(sorted(aligned_motion_data.items()))
    return aligned_motion_data



class MotionDimensionReductionConstructor(object):

    def __init__(self, elementary_action, motion_primitive):
        self.elemetary_action = elementary_action
        self.motion_primitive = motion_primitive

    def load(self):
        aligned_motion_data = load_aligned_data(self.elemetary_action,
                                                self.motion_primitive)
        params = ConstructionAlgorithmConfigurationBuilder(self.elemetary_action,
                                                           self.motion_primitive)
        skeleton_file = r'../../../skeleton.bvh'
        bvhreader = BVHReader(skeleton_file)
        skeleton = Skeleton(bvhreader)
        return MotionDimensionReduction(aligned_motion_data,
                                        bvhreader,
                                        params)
