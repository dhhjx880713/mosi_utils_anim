# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 13:44:05 2015

@author: Han
"""

import os
import sys
ROOT_DIR = os.sep.join(['..'] * 2)
sys.path.append(ROOT_DIR)
from utilities.bvh import BVHReader, BVHWriter
import json
from utilities.io_helper_functions import load_json_file
SERVICE_CONFIG_FILE = ROOT_DIR + os.sep + "config" + os.sep + "service.json"


class MotionSegmentation(object):

    def __init__(self, folder_path, annotation, save_path, verbose=False):
        if not folder_path.endswith(os.sep):
            folder_path += os.sep
        if not save_path.endswith(os.sep):
            save_path += os.sep
        if verbose:
            print "Looking for files in %s" % folder_path
        self.folder_path = folder_path
        self.annotation_file = annotation
        self.save_path = save_path
        self.verbose = verbose

    def segment_motions(self, elementary_action, primitive_type):
        self._load_annotation()
        self._cut_files(elementary_action, primitive_type)

    def _load_annotation(self):
        self.annotation_label= self._convert_to_json(self.annotation_file,
                                                     export=False)
        if self.verbose:
            print "Load &d files." % len(self.annotation_label.keys())

    def _cut_files(self, elementary_action, primitive_type):
        self.cut_motions = {}
        for filename, items in self.annotation_label.iteritems():
            for primitive_data in items:
                if primitive_data['elementary_action'] == elementary_action \
                   and primitive_data['motion_primitive'] == primitive_type:
                    print "find motion primitive " + elementary_action + '_' \
                          + primitive_type + ' in ' + filename
                    file_path = self.folder_path + filename
                    if not os.path.isfile(file_path):
                        raise IOError(
                            'cannot find ' +
                            filename +
                            ' in ' +
                            self.folder_path)
                    start_frame = primitive_data['frames'][0]
                    end_frame = primitive_data['frames'][1]
                    filename_segments = filename[:-4].split('_')
                    # repalce the first element, which should be elementary
                    # action name, by real elementary name
                    filename_segments[0] = elementary_action
                    outfilename = '_'.join(filename_segments) + \
                                  '_%s_%d_%d.bvh' % (primitive_type,
                                                     start_frame,
                                                     end_frame)
                    cutted_frames = self._cut_one_file(file_path,
                                                       start_frame,
                                                       end_frame)
                    self.cut_motions[outfilename] = cutted_frames

    def save_segments(self):
        for outfilename, frames in self.cut_motions.iteritems():
            save_filename = self.save_path + outfilename
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
    retarget_folder = data_path + os.sep + r'2 - Rocketbox retargeting\Take_sidestep'
    cutting_folder = data_path + os.sep + r'3 - Cutting\test'
    annotation = retarget_folder + os.sep + 'key_frame_annotation.txt'
    motion_segmentor = MotionSegmentation(retarget_folder, annotation,
                                          cutting_folder)
    motion_segmentor.segment_motions(elementary_action,
                                     primitive_type)  
    motion_segmentor.save_segments()            
                         

if __name__ == '__main__':
    main()
