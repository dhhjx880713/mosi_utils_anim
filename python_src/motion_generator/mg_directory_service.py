# -*- coding: utf-8 -*-
"""
Created on Thu May 07 11:33:10 2015

Simple Motion Graphs directory watch based service for pipeline tests.
Note the loading of transition models can take up to 2 minutes

@author: erhe01
"""
import sys
import os
import time
from lib.directory_observation import DirectoryObserverThread
from controllable_morphable_graph import ControllableMorphableGraph
from lib.helper_functions import get_morphable_model_directory, get_transition_model_directory
from constrain_motion import generate_algorithm_settings
from lib.helper_functions import load_json_file,write_to_json_file,export_quat_frames_to_bvh
CONFIG_FILE = "config.json"

class MorphableGraphService():
    def __init__(self,input_file_dir,output_dir,output_filename):
        self.input_file_dir = input_file_dir
        self.output_dir = output_dir
        self.output_filename = output_filename
        
        mm_directory = get_morphable_model_directory()
        transition_directory = get_transition_model_directory()
        skeleton_path = "lib"+os.sep + "skeleton.bvh"
        use_transition_model = False
        self.cmg =ControllableMorphableGraph(mm_directory,
                                     transition_directory,
                                     skeleton_path,
                                     use_transition_model)      
        self.verbose = False
        self.algorithm_version = 3
        self.max_step = -1
  
        if os.path.isfile(CONFIG_FILE):
            self.options = load_json_file(CONFIG_FILE)
        else:
            self.options = generate_algorithm_settings()
        self.observer_thread = DirectoryObserverThread(input_file_dir,pattern="*.json",callback=self.process_file)
         
    def process_file(self,input_file):
        quat_frames,frame_annotation,action_list = self.cmg.synthesize_motion(input_file,options=self.options,
                                                                                max_step=self.max_step,
                                                                                  version=self.algorithm_version,verbose=self.verbose,
                                                                                  output_dir=self.output_dir,
                                                                                  out_name=self.output_filename,
                                                                                  export=False)
                                                                              
        if quat_frames is not None:
            # save frames + the annotation and actions to file
            input_constraints = load_json_file(input_file)
            write_to_json_file(self.output_dir + os.sep+self.output_filename+".json",input_constraints)
            write_to_json_file(self.output_dir + os.sep+self.output_filename + "_actions"+".json",action_list)
            export_quat_frames_to_bvh(self.output_dir,self.cmg.bvh_reader,quat_frames,prefix = self.output_filename,start_pose= None,time_stamp =False)
        else:
            print "failed to generate motion data"
    
        return

    def run(self):
        self.observer_thread.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
                self.observer_thread.observer.stop()
        self.observer_thread.join()
        return
        
        
def main(input_file_dir,output_dir,output_filename):
    mgs = MorphableGraphService(input_file_dir,output_dir,output_filename)
    mgs.run()
    return
 
if __name__ == "__main__":
    """example call:
       mg_directory_service.py  "input_dir" "output_dir" "filename"
     """
    
    os.chdir(sys.path[0])  # change working directory to the file directory
    args = sys.argv
    if len(args) >= 3:
        input_file_dir= args[1]
        if input_file_dir.count('*') >= 1:
            import glob
            print input_file_dir
            input_file_dir = glob.glob(input_file_dir)[-1]
        output_dir = args[2]
        if len(args) >= 4:
            output_filename = args[3]
        else:
            output_filename = "result"
   
        main(input_file_dir,output_dir,output_filename)
   
