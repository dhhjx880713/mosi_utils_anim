# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:11:52 2015

Simple Motion Graphs command line interface for pipeline tests.
Note the loading of transition models can take up to 2 minutes

@author: erhe01
"""


import os
 # change working directory to the script file directory
dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(dirname)
import glob
import time
from controllable_morphable_graph import load_morphable_graph, export_synthesis_result
from utilities.io_helper_functions import load_json_file
from constrain_motion import generate_algorithm_settings


ALGORITHM_CONFIG_FILE = "algorithm_config.json"
SERVICE_CONFIG_FILE = "service_config.json"

def run_pipeline(root_directory, input_file, output_dir, output_filename, config_file):
    """Creates an instance of the morphable graph and runs the synthesis
       algorithm with the input_file and standard parameters.
    """
    
    max_step = -1
    if os.path.isfile(config_file):
        options = load_json_file(config_file)
    else:
        options = generate_algorithm_settings()

    start = time.clock()
    morphable_graph = load_morphable_graph(root_directory, use_transition_model=options["use_transition_model"])
    print "finished construction from file in",time.clock()-start,"seconds"
    
    verbose = False

    result_tuple = morphable_graph.synthesize_motion(input_file,options=options,
                                                      max_step=max_step,
                                                      verbose=verbose,
                                                      output_dir=output_dir,
                                                      output_filename=output_filename,
                                                      export=False)

    if result_tuple[0] is not None:  # checks for quat_frames in result_tuple
        mg_input = load_json_file(input_file)
        export_synthesis_result(mg_input, output_dir, output_filename, morphable_graph.bvh_reader, *result_tuple, add_time_stamp=False)
    else:
        print "Error: Failed to generate motion data."

if __name__ == "__main__":
    """example call:
       mg_pipeline_interface.py
    """
    import warnings
    warnings.simplefilter("ignore")
    
    if os.path.isfile(SERVICE_CONFIG_FILE):
        service_config = load_json_file(SERVICE_CONFIG_FILE) 
        
        # select input file as latest file from a fixed input directory
        globalpath = service_config["input_dir"] + os.sep + "*.json"
        input_file = glob.glob(globalpath)[-1]
        
        run_pipeline(service_config["data_root"], input_file, service_config["output_dir"], service_config["output_filename"], ALGORITHM_CONFIG_FILE)
    else:
        print "Error: Could not read service config file", SERVICE_CONFIG_FILE