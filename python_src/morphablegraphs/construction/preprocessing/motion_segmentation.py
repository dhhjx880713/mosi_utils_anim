# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 13:44:05 2015

@author: Han
"""

import os
import sys
ROOT_DIR = os.sep.join(['..'] * 2)
sys.path.append(ROOT_DIR)
from animation_data.bvh import BVHReader, BVHWriter
import json
from utilities.io_helper_functions import load_json_file
SERVICE_CONFIG_FILE = ROOT_DIR + os.sep + "config" + os.sep + "service.json"


class MotionSegmentation(object):

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.cutted_motions = {}
        self.annotation_label = {}
        self.bvhreader = None

    def segment_motions(self, elementary_action, primitive_type,
                        data_path, annotation_file):
        self.load_annotation(annotation_file)
        self.cut_files(elementary_action, primitive_type, data_path)

    def load_annotation(self, annotation_file):
        self.annotation_label = self._convert_to_json(annotation_file,
                                                      export=False)
        if self.verbose:
            print "Load %d files." % len(self.annotation_label.keys())

    def _check_motion_type(self, elementary_action, primitive_type, primitive_data):
        if primitive_data['elementary_action'] == elementary_action \
           and primitive_data['motion_primitive'] == primitive_type:
            return True
        else:
            return False

    def _get_annotation_information(self, data_path, filename, primitive_data):
        file_path = data_path + filename
        if not os.path.isfile(file_path):
            raise IOError(
                'cannot find ' +
                filename +
                ' in ' +
                data_path)
        start_frame = primitive_data['frames'][0]
        end_frame = primitive_data['frames'][1]
        filename_segments = filename[:-4].split('_')
        return start_frame, end_frame, filename_segments


    def cut_files(self, elementary_action, primitive_type, data_path):
        if not data_path.endswith(os.sep):
            data_path += os.sep
        if self.verbose:
            print "search files in " + data_path
        for filename, items in self.annotation_label.iteritems():
            for primitive_data in items:
                if self._check_motion_type(elementary_action, primitive_type, primitive_data):
                    print "find motion primitive " + elementary_action + '_' \
                          + primitive_type + ' in ' + filename
                    start_frame, end_frame, filename_segments = self._get_annotation_information(data_path, filename,
                                                                                                 primitive_data)
                    filename_segments[0] = elementary_action
                    cutted_frames = self._cut_one_file(data_path + filename,
                                                       start_frame,
                                                       end_frame)
                    outfilename = '_'.join(filename_segments) + \
                                  '_%s_%d_%d.bvh' % (primitive_type,
                                                     start_frame,
                                                     end_frame)
                    self.cutted_motions[outfilename] = cutted_frames
                else:
                    print "cannot find motion primitive " + elementary_action + '_' \
                          + primitive_type + ' in ' + filename

    def save_segments(self, save_path=None):
        if save_path is None:
            raise ValueError('Please give saving path!')
        if not save_path.endswith(os.sep):
            save_path += os.sep
        for outfilename, frames in self.cutted_motions.iteritems():
            save_filename = save_path + outfilename
            BVHWriter(save_filename, self.bvhreader, frames,
                      frame_time=self.bvhreader.frame_time,
                      is_quaternion=False)

    def _cut_one_file(self, input_file, start_frame, end_frame,
                      toe_modify=True):
        self.bvhreader = BVHReader(input_file)
        new_frames = self.bvhreader.frames[start_frame: end_frame]
        return new_frames

    def _convert_to_json(self, annotation_file, export=False):
        with open(annotation_file, 'rb') as input_file:
            annotation_data = {}
            current_motion = None
            for line in input_file:
                line = line.rstrip()
                if '.bvh' in line:
                    current_motion = line
                    annotation_data[current_motion] = []
                elif current_motion is not None and line != '' and line != '\n':
                    try:
                        line_split = line.split(' ')
                        tmp = {'elementary_action': line_split[0], 'motion_primitive': line_split[
                            1], 'frames': [int(line_split[2]), int(line_split[3])]}
                        annotation_data[current_motion].append(tmp)
                    except ValueError:
                        raise ValueError("Couldn't process line: %s" % line)
        if export:
            filename = os.path.split(annotation_file)[-1]
            with open(filename[:-4] + '.json', 'w+') as fp:
                json.dump(annotation_data, fp)
                fp.close()
        return annotation_data


def main():
    path_data = load_json_file(SERVICE_CONFIG_FILE)
    data_path = path_data['data_folder']
    elementary_action = 'walk'
    primitive_type = 'sidestepLeft'

    print data_path
    retarget_folder = data_path + os.sep + \
        r'2 - Rocketbox retargeting\Take_sidestep'
    cutting_folder = data_path + os.sep + r'3 - Cutting\test'
    annotation = retarget_folder + os.sep + 'key_frame_annotation.txt'
    motion_segmentor = MotionSegmentation()
    motion_segmentor.segment_motions(elementary_action,
                                     primitive_type,
                                     retarget_folder,
                                     annotation)
    motion_segmentor.save_segments(cutting_folder)


if __name__ == '__main__':
    main()
