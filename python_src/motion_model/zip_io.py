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

def read_zip_file_structure(zip_file):
    structure_desc = {}
    for name in zip_file.namelist():
        splitted_names = name.split("/")
        if len(splitted_names) >1:
            #data = f.read(name)
            directory = splitted_names[0]
            file_name = splitted_names[1]
            if file_name.endswith(MORPHABLE_MODEL_FILE_ENDING):
                if directory not in structure_desc.keys():
                    structure_desc[directory] = []
                structure_desc[directory].append(file_name[:-8])

    return structure_desc


def construct_graph_data(zip_file, structure_desc, pickle_objects=False):
    graph_data = {}
    mm_type = "quaternion"
    type_offset = len(mm_type)+1
    for structure_key in structure_desc.keys():
        action_data_key = structure_key.split("_")[2]
        print "action key",action_data_key
        graph_data[action_data_key] = {}
        graph_data[action_data_key]["name"] = action_data_key
        meta_info_file = structure_key+"/meta_information.json"
        if meta_info_file in zip_file.namelist():
            graph_data[action_data_key]["info"] = json.loads(zip_file.read(meta_info_file))
        graph_data[action_data_key]["nodes"] = {}
        for mp in structure_desc[structure_key]:
            mp_data_key = (mp[:-type_offset]).split("_")[1]
            graph_data[action_data_key]["nodes"][mp_data_key] = {}
            graph_data[action_data_key]["nodes"][mp_data_key]["name"] = mp[:-type_offset]
            mm_string = zip_file.read(structure_key+"/"+mp+"_mm.json")
            mm_data = json.loads(mm_string)
            graph_data[action_data_key]["nodes"][mp_data_key]["mm"] = mm_data
            print "\t",mp
            statsfile = structure_key+"/"+(mp[:-type_offset]+".stats")#.split("_")[1]
        
    
            if statsfile in zip_file.namelist():
                #print statsfile
                #print "##############",statsfile
                stats_string = zip_file.read(statsfile)
                stats_data = json.loads(stats_string)
                graph_data[action_data_key]["nodes"][mp_data_key]["stats"] = stats_data
                #print graph_data[action_data_key]["nodes"][mp_data_key]["stats"]
      
                
            space_partition_file = structure_key+"/"+mp+"_cluster_tree.pck" 
            
            if space_partition_file in zip_file.namelist():
                #print space_partition_file
                space_partition = zip_file.read(space_partition_file)
                graph_data[action_data_key]["nodes"][mp_data_key]["space_partition"] = None
                if pickle_objects:
                    #space_partition_data = ClusterTree()#pickle.load(space_partition)
                    space_partition_data = cPickle.loads(space_partition)
                    #pickle.dump(space_partition_data,space_partition)
               
                    graph_data[action_data_key]["nodes"][mp_data_key]["space_partition"] = space_partition_data
    return graph_data

def read_graph_data_from_zip(zip_path,pickle_objects=True):
    """ Extracts the data from the files stored in the zip file and 
        returns it in a dictionary for easier parsing. The space partitioning 
        data structure is also deserialized into an object.
        If pickle_objects is False the space partitioning is ignored.
    """

    data = {}

    with zipfile.ZipFile(zip_path, "r",zipfile.ZIP_DEFLATED) as zip_file:

        data["transitions"] = json.loads(zip_file.read("graph_definition.json"))
        structure_desc = read_zip_file_structure(zip_file)
        data["subgraphs"] = construct_graph_data(zip_file, structure_desc, pickle_objects)
        return data
   
def main():
    zip_path = "E:\\projects\\INTERACT\\repository\\data\\3 - Motion primitives\\motion_primitives_quaternion_PCA95.zip"
    print zip_path
    start = time.clock()
    graph_data = read_graph_data_from_zip(zip_path)
    print graph_data["subgraphs"]["pick"]["nodes"].keys()
    print "finished reading data in",time.clock()-start,"seconds"
    #print  graph_data     
    
if __name__ == "__main__":
    main()

                    