# encoding: UTF-8
"""
Automatically check timewarping file is still corresponding to the bvh files in the alignment folder
"""
from ...utilities import get_aligned_data_folder, load_json_file, write_to_json_file
from ...animation_data import BVHReader
import glob
import os


def check_meta_data(elementary_action,
                    motion_primitive):
    print(elementary_action, motion_primitive)
    aligned_folder = get_aligned_data_folder(elementary_action,
                                             motion_primitive)
    bvhfiles = glob.glob(os.path.join(aligned_folder, '*.bvh'))
    timewarping_data = load_json_file(os.path.join(aligned_folder, 'timewarping.json'))
    n_frames = len(timewarping_data[timewarping_data.keys()[0]])
    print('number of frames: ', n_frames)
    print 'number of files is: ' + str(len(bvhfiles))
    print 'number of indices is: ' + str(len(timewarping_data.keys()))
    filenames = []
    for item in bvhfiles:
        filename = os.path.split(item)[-1]
        filenames.append(filename)
        bvhreader = BVHReader(item)
        if len(bvhreader.frames) != n_frames:
            print(filename + ' has different number of frames!')
        # print filename
        if not filename in timewarping_data.keys():
            print filename + ' cannot be found!'
    for filename in timewarping_data.keys():
        # delete key in timewarping_data if the corresponding file does not exist
        if filename not in filenames:
            del timewarping_data[filename]
    write_to_json_file(os.path.join(aligned_folder, 'timewarping.json'), timewarping_data)


if __name__ == '__main__':
    elementary_action = 'walk'
    motion_primitive = 'endRightStance'
    check_meta_data(elementary_action,
                    motion_primitive)