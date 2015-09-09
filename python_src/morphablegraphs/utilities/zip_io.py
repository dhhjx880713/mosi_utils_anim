# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 18:34:12 2015

@author: erhe01
"""
import zipfile
import json
import cPickle
import time

MORPHABLE_MODEL_FILE_ENDING = "mm.json"
MM_TYPE = "quaternion"


class ZipReader(object):
    def __init__(self, zip_file_path, pickle_objects=True):
        self.zip_file_path = zip_file_path
        self.zip_file = None
        self.pickle_objects = pickle_objects
        self.type_offset = len(MM_TYPE) + 1
        self.graph_data = None

    def get_graph_data(self):
        """ Extracts the data from the files stored in the zip file and
            returns it in a dictionary for easier parsing. The space partitioning
            data structure is also deserialized into an object.
            If pickle_objects is False the space partitioning is ignored.
        """
        data = {}
        self.zip_file = zipfile.ZipFile(self.zip_file_path, "r", zipfile.ZIP_DEFLATED)
        data["transitions"] = json.loads(self.zip_file.read("graph_definition.json"))
        structure_desc = self._read_zip_file_structure()
        self._construct_graph_data(structure_desc)
        data["subgraphs"] = self.graph_data
        self.zip_file.close()
        return data

    def _read_zip_file_structure(self):
        structure_desc = dict()
        for name in self.zip_file.namelist():
            splitted_names = name.split("/")
            if len(splitted_names) > 1:
                directory = splitted_names[0]
                file_name = splitted_names[1]
                if file_name.endswith(MORPHABLE_MODEL_FILE_ENDING):
                    if directory not in structure_desc.keys():
                        structure_desc[directory] = []
                    structure_desc[directory].append(file_name[:-8])
        return structure_desc

    def _construct_graph_data(self, structure_desc):
        self.graph_data = dict()
        for structure_key in structure_desc.keys():
            action_data_key = structure_key.split("_")[2]
            print "action key", action_data_key
            self.graph_data[action_data_key] = {}
            self.graph_data[action_data_key]["name"] = action_data_key
            meta_info_file = structure_key + "/meta_information.json"
            if meta_info_file in self.zip_file.namelist():
                self.graph_data[action_data_key]["info"] = json.loads(self.zip_file.read(meta_info_file))
            self.graph_data[action_data_key]["nodes"] = {}
            for mp in structure_desc[structure_key]:
                self._add_motion_primitive(action_data_key, structure_key, mp)

    def _add_motion_primitive(self, action_data_key, structure_key, mp):
        mp_data_key = (mp[:-self.type_offset]).split("_")[1]
        self.graph_data[action_data_key]["nodes"][mp_data_key] = {}
        self.graph_data[action_data_key]["nodes"][mp_data_key]["name"] = mp[:-self.type_offset]
        mm_string = self.zip_file.read(structure_key + "/" + mp + "_mm.json")
        mm_data = json.loads(mm_string)
        self.graph_data[action_data_key]["nodes"][mp_data_key]["mm"] = mm_data
        print "\t", mp
        statsfile = structure_key + "/" + (mp[:-self.type_offset] + ".stats")
        self._add_stats(action_data_key, mp_data_key, statsfile)
        space_partition_file = structure_key + "/" + mp + "_cluster_tree.pck"
        self._add_space_partioning_data_structure(action_data_key, mp_data_key, space_partition_file)

    def _add_stats(self, action_data_key, mp_data_key, statsfile):
        if statsfile in self.zip_file.namelist():
            stats_string = self.zip_file.read(statsfile)
            stats_data = json.loads(stats_string)
            self.graph_data[action_data_key]["nodes"][mp_data_key]["stats"] = stats_data

    def _add_space_partioning_data_structure(self, action_data_key, mp_data_key, space_partition_file):
        if space_partition_file in self.zip_file.namelist():
            space_partition = self.zip_file.read(space_partition_file)
            self.graph_data[action_data_key]["nodes"][mp_data_key]["space_partition"] = None
            if self.pickle_objects:
                space_partition_data = cPickle.loads(space_partition)
                self.graph_data[action_data_key]["nodes"][mp_data_key][
                    "space_partition"] = space_partition_data

def main():
    zip_path = "E:\\projects\\INTERACT\\repository\\data\\3 - Motion primitives\\motion_primitives_quaternion_PCA95.zip"
    print zip_path
    start = time.clock()
    zip_loader = ZipReader(zip_path)
    graph_data = zip_loader.get_graph_data()
    print graph_data["subgraphs"]["pick"]["nodes"].keys()
    print "finished reading data in", time.clock() - start, "seconds"
    # print  graph_data

if __name__ == "__main__":
    main()
