# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:11:52 2015

Simple Motion Graphs command line interface for pipeline tests.
Note the loading of transition models can take up to 2 minutes

@author: erhe01
"""


import sys
import os
 # change working directory to the script file directory
dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(dirname)
import glob
import time
from controllable_morphable_graph import ControllableMorphableGraph, export_synthesis_result
from lib.helper_functions import global_path_dict, get_morphable_model_directory, get_transition_model_directory
from constrain_motion import generate_algorithm_settings
from lib.helper_functions import load_json_file

CONFIG_FILE = "config.json"
SKELETON_FILE = "lib" + os.sep + "skeleton.bvh"


def load_morphable_graph(use_transition_model=False, skeleton_file=SKELETON_FILE):
    mm_directory = get_morphable_model_directory()
    transition_directory = get_transition_model_directory()
    cmg = ControllableMorphableGraph(mm_directory,
                                     transition_directory,
                                     skeleton_file,
                                     use_transition_model)
    return cmg


def run_pipeline(input_file, output_dir, output_filename, config_file):
    """Creates an instance of the morphable graph and runs the synthesis
       algorithm with the input_file and standard parameters.
    """
    
    max_step = -1
    if os.path.isfile(config_file):
        options = load_json_file(config_file)
    else:
        options = generate_algorithm_settings()

    start = time.clock()
    morphable_graph = load_morphable_graph(use_transition_model=options["use_transition_model"])
    print "finished construction from file in",time.clock()-start,"seconds"
    
    verbose = False

    result_tuple = morphable_graph.synthesize_motion(input_file,options=options,
                                                      max_step=max_step,
                                                      verbose=verbose,
                                                      output_dir=output_dir,
                                                      output_filename=output_filename,
                                                      export=False)

    if result_tuple[0] != None:  # checks for quat_frames in result_tuple
        mg_input = load_json_file(input_file)
        export_synthesis_result(mg_input, output_dir, output_filename, morphable_graph.bvh_reader, *result_tuple, add_time_stamp=False)
    else:
        print "failed to generate motion data"

if __name__ == "__main__":
    """example call:
       mg_pipeline_interface.py
    """
    import warnings
    warnings.simplefilter("ignore")
    
    # set the path to the parent of the data directory 
    # TODO set as configuration file parameter
    global_path_dict["data_root"] = "E:\\projects\\INTERACT\\repository\\"
    
   
    
    # select input file as latest file from a fixed input directory
    local_path = os.path.dirname(__file__)
    globalpath = global_path_dict["data_root"] + r"BestFitPipeline\CNL-GUI\*.json"
    input_file = glob.glob(globalpath)[-1]
    
    # set output parameters to a fixed directory that is observed by an
    # animation server
    output_dir = global_path_dict["data_root"] + r"BestFitPipeline\_Results"
    output_filename = "MGresult"
    
    run_pipeline(input_file, output_dir, output_filename, CONFIG_FILE)
