# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:01:02 2016

@author: mamanns
"""

import os
import hashlib


def duplicated_file_detection():

    file_hashes = {}

    BVH_PATH = r"C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\turnLeftRightStance"

    for root, dirs, files in os.walk(BVH_PATH):
        for name in files:
            filepath = os.path.join(root, name)
            with open(filepath) as infile:
                hashres = hashlib.sha256(infile.read()).hexdigest()
                if hashres not in file_hashes:
                    file_hashes[hashres] = [name]
                else:
                    file_hashes[hashres] += [name]

    counter = 0
    for key in file_hashes:
        if len(file_hashes[key]) > 1:
            print file_hashes[key]
            delete_duplicated_files(file_hashes[key], BVH_PATH)
            counter += 1

    print counter, len(file_hashes)

def delete_duplicated_files(filenames, folder_path):
    """
    Delete the duplicated files in the list of filenames
    :param filenames: list of str
    :return:
    """
    segments = []
    for filename in filenames:
        segments.append(filename[:-4].split('_'))
    segments.sort(key=lambda x: int(x[4]))
    for seg in segments[:-1]:
        filename = '_'.join(seg) + '.bvh'
        if os.path.exists(os.path.join(folder_path, filename)):
            os.remove(os.path.join(folder_path, filename))

if __name__ == "__main__":
    duplicated_file_detection()