# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:11:52 2015

Simple Motion Graphs command line interface for pipeline tests.
Note the loading of transition models can take up to 2 minutes

@author: erhe01
"""
import sys
import os
import glob
from controllable_morphable_graph import ControllableMorphableGraph
from lib.helper_functions import get_morphable_model_directory, get_transition_model_directory
from constrain_motion import generate_algorithm_settings
from lib.helper_functions import load_json_file,write_to_json_file,export_quat_frames_to_bvh
CONFIG_FILE  = "config.json"

def run_pipeline(input_file, output_dir, output_filename, config_file):
    """Creates an instance of the morphable graph and runs the synthesis
       algorithm with the input_file and standard parameters.
    """
    mm_directory = get_morphable_model_directory()
    transition_directory = get_transition_model_directory()
    skeleton_path = "lib"+os.sep + "skeleton.bvh"
    use_transition_model = False

    cmg = ControllableMorphableGraph(mm_directory,
                                     transition_directory,
                                     skeleton_path,
                                     use_transition_model)
    verbose = False
    version = 3
    max_step = -1
    if os.path.isfile(config_file):
        options = load_json_file(config_file)
    else:
        options = generate_algorithm_settings()
    quat_frames,frame_annotation,action_list = cmg.synthesize_motion(input_file,options=options,
                                                                      max_step=max_step,
                                                                      version=version,verbose=verbose,
                                                                      output_dir=output_dir,
                                                                      out_name=output_filename,
                                                                      export=False)


    if quat_frames != None:
        # save frames + the annotation and actions to file
        input_constraints = load_json_file(input_file)
        write_to_json_file(output_dir + os.sep+output_filename+".json",input_constraints)
        write_to_json_file(output_dir + os.sep+output_filename + "_actions"+".json",action_list)
        frame_annotation["events"] = []
        for keyframe in action_list.keys():
            for event_desc in action_list[keyframe]:
                event = {}
                #if "joint" in action_list[keyframe]["parameters"].keys():
                event["jointName"] = event_desc["parameters"]["joint"]
                event_type = event_desc["event"]
                target = event_desc["parameters"]["target"]
                event[event_type] = target
            
                event["frameNumber"] = int(keyframe)
                
                frame_annotation["events"].append(event)
        write_to_json_file(output_dir + os.sep+output_filename + "_annotations"+".json",frame_annotation)
        export_quat_frames_to_bvh(output_dir,cmg.bvh_reader,quat_frames,prefix = output_filename,start_pose= None,time_stamp =False)
    else:
        print "failed to generate motion data"

if __name__ == "__main__":
    """example call:
       mg_pipeline_interface.py
    """
    import warnings
    warnings.simplefilter("ignore")
    os.chdir(sys.path[0])  # change working directory to the file directory
    local_path = os.path.dirname(__file__)
    globalpath = local_path + r"\..\..\BestFitPipeline\CNL-GUI\*.json"
    print globalpath
    input_file = glob.glob(globalpath)[-1]
    output_dir = local_path + r"\..\..\BestFitPipeline\_Results"
    output_filename = "MGresult"
    run_pipeline(input_file, output_dir, output_filename, CONFIG_FILE)
