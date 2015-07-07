# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 17:33:59 2015
Get average velocity and  bounding box  over training data
Identify outliers in samples

@author: erhe01
"""
import sys
import os
import time
import collections
from copy import copy
import numpy as np
import datetime
import json
#add the motion primitive directory to the python path variable
sys.path.append(os.sep.join(["..", "2 - motion_primitive"]))
from motion_primitive import MotionPrimitive
from lib.bvh import BVHReader, BVHWriter
from helper_functions import *
from evaluation_methods import *
from motion_parameter_conversion import convert_quaternion_to_euler
import rpy2


ROOT_DIR = os.sep.join([".."] * 3)

def calculate_average_velocity_from_directory(directory_path):
    """
    Calculate average joint velocity over training samples in parameter space.
    The parameter space is defined as a vector of Euler Angles + cartesian
    coordinates for the root joint.
    """
    avg_motion_velocities = []
    for item in gen_file_paths(directory_path):
        bvh_reader = BVHReader(item)
        if len(bvh_reader.keyframes) > 2:
            vector = calculate_avg_motion_velocity_from_bvh(bvh_reader, True)
            avg_motion_velocities.append(vector)

    #calculate overall velocity + variance
    #initialize data structure
    overall_velocity = collections.OrderedDict()
    for node_name in avg_motion_velocities[0].keys():
        overall_velocity[node_name] = collections.OrderedDict()
        print node_name
        for c in avg_motion_velocities[0][node_name].keys():
            print "c", c
            overall_velocity[node_name][c] = {"avg": 0.0, "var":0.0}

    #calculate the average: sum over all and divide by N
    #,  where N is the number of motion samples
    for avg_motion_velocity in avg_motion_velocities:
        for node_name in avg_motion_velocity.keys():
            for c in avg_motion_velocity[node_name].keys():
                 overall_velocity[node_name][c]["avg"] += \
                                     avg_motion_velocity[node_name][c]["avg"]
    N = len(avg_motion_velocities)
    for node_name in overall_velocity.keys():
        for c in overall_velocity[node_name].keys():
            overall_velocity[node_name][c]["avg"] /= N

    # calculate empirical variance: get squared difference to mean and
    # divide by N-1, the number of motion samples
    for avg_motion_velocity in avg_motion_velocities:
        for node_name in avg_motion_velocity.keys():
            for c in avg_motion_velocity[node_name].keys():
                var = (overall_velocity[node_name][c]["avg"] - \
                         avg_motion_velocity[node_name][c]["avg"])**2
                overall_velocity[node_name][c]["var"] = var

    for node_name in overall_velocity.keys():
        for c in overall_velocity[node_name].keys():
            overall_velocity[node_name][c]["var"] /=  N-1
    print overall_velocity
    return overall_velocity

def calculate_parameter_bounding_box_from_directory(directory_path):
    """
    calculate joint parameter bounding box over training samples
    """
    parameter_bounding_boxes = []
    for item in gen_file_paths(directory_path):
        bvh_reader = BVHReader(item)
        bb = calculate_parameter_bounding_box(bvh_reader)
        parameter_bounding_boxes.append(bb)

    #calculate overall bounding box
    #overall_bb = parameter_bounding_boxes[0]
    overall_bb = collections.OrderedDict()#{}
    for node_name in  parameter_bounding_boxes[0].keys():
        overall_bb[node_name] =collections.OrderedDict()
        for c in parameter_bounding_boxes[0][node_name].keys():
            overall_bb[node_name][c] = {"min":np.inf,"max":-np.inf}


  
    for bb in parameter_bounding_boxes:
        for node_name in bb.keys():
            for c in bb[node_name].keys():
                update_bb_value(overall_bb[node_name][c], \
                                bb[node_name][c]["min"])
                update_bb_value(overall_bb[node_name][c], \
                                bb[node_name][c]["max"])
    print overall_bb
    return overall_bb
    
    
def calculate_parameter_bounding_box_from_directory2(directory_path):
    """
    calculate joint parameter bounding box over training samples
    """
    
    overall_bb = None
    for item in gen_file_paths(directory_path):
        bvh_reader = BVHReader(item)
        if overall_bb:
            check_parameter_bounding_box(bvh_reader,bvh_reader.keyframes, overall_bb,0, True,True)
        else:
            overall_bb = calculate_parameter_bounding_box(bvh_reader)
    print overall_bb
    return overall_bb
  

def calculate_cartesian_pose_bounding_box_from_directory(directory_path):
    """
    calculate cartesian bounding box over training samples
    """

    bounding_boxes = []
    for item in gen_file_paths(directory_path):
        bvh_reader = BVHReader(item)
        bb = calculate_cartesian_pose_bounding_box(bvh_reader)
        bounding_boxes.append(bb)


    #calculate overall bounding box
    #overall_bb = bounding_boxes[0]
    overall_bb = collections.OrderedDict()
    for c in ["X", "Y", "Z"]:
        overall_bb[c] = {"min":np.inf,"max":-np.inf}
  
    for bb in bounding_boxes:
        for c in bb.keys():
            update_bb_value(overall_bb[c], bb[c]["min"])
            update_bb_value(overall_bb[c], bb[c]["max"])

    print overall_bb
    return overall_bb

def calculate_average_motion_energy_from_directory(directory_path):
    """
    calculate average energy vector over training samples
    """
    motion_energy_vectors =[]
    for item in gen_file_paths(directory_path):
        bvh_reader = BVHReader(item)
        bb = calculate_avg_motion_velocity_from_bvh(bvh_reader, False)
        motion_energy_vectors.append(bb)

    #sum over all
    motion_energy_vector_sum = motion_energy_vectors[0]
    i = 1#directly copy the first one to keep the structure
    while i < len(motion_energy_vectors):
        for node_name in motion_energy_vectors[i].keys():
            for c in motion_energy_vectors[i][node_name].keys():
                e = motion_energy_vectors[i][node_name][c]["energy"]
                motion_energy_vector_sum[node_name][c]["energy"] += e

        i += 1

    #divide by N the number of motion samples
    N = len(motion_energy_vectors)
    for node_name in motion_energy_vector_sum.keys():
        for c in motion_energy_vector_sum[node_name].keys():
            motion_energy_vector_sum[node_name][c]["energy"] /=  N

    return motion_energy_vector_sum



def create_parameter_bounding_box_from_training_data(input_dir, \
                                        elementary_action, motion_primitive, \
                                        recalculate=False):
                                            
    """
    Creates a file path based on the input parameters and tries to load a
    parameter bounding box for a motion primitive.
    If the file does not exists a directory path is generated and a new
    parameter bounding box is calculated using the BVH files found in this 
    directory. The resulting bounding box is then saved to the previously 
    generated file path in order to be loaded automatically on the next run.
    """                               
    file_name = input_dir+ "\\"+elementary_action+"_"+motion_primitive+".bb"

    if os.path.isfile(file_name) and not recalculate:
        print "read "+file_name
        in_f = open(file_name, "rb")
        #http://stackoverflow.com/questions/6921699/can-i-get-json-to-load-into-an-ordereddict-in-python
        pose_bb = json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(in_f.read())
        in_f.close()
    else:
        directory_path = get_input_data_folder(elementary_action,motion_primitive, sub_dir_name="4 - Alignment")
        assert os.path.isdir(directory_path)
        print "write "+file_name

        pose_bb = calculate_parameter_bounding_box_from_directory2(directory_path)
        bb_string = json.dumps(pose_bb)
        out_f = open(file_name, "wb")
        out_f.write(bb_string)
        out_f.close()

    return pose_bb



def calculate_avg_velocity_from_training_data(input_dir, \
                                             elementary_action, \
                                             motion_primitive, \
                                             recalculate=False):
    """
    Creates a file path based on the input parameters and tries to load a
    the average velocity for the pose paramaters for a motion primitive.
    If the file does not exists a directory path is generated and the avaerage 
    velocity is calculated using the BVH files found in this 
    directory. The resulting data is then saved to the previously 
    generated file path in order to be loaded automatically on the next run.
    """
    file_name =input_dir+ "\\"+elementary_action+"_"+motion_primitive+".vel"
    if os.path.isfile(file_name) and not recalculate:
        print "read "+file_name
        in_f = open(file_name, "rb")
        #http://stackoverflow.com/questions/6921699/can-i-get-json-to-load-into-an-ordereddict-in-python
        pose_velocity = json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(in_f.read())
        in_f.close()
    else:
        directory_path = get_input_data_folder(elementary_action,motion_primitive)
        assert os.path.isdir(directory_path)
        print "write "+file_name

        pose_velocity = calculate_average_velocity_from_directory(directory_path)
        pose_velocity_string = json.dumps(pose_velocity)
        out_f = open(file_name, "wb")
        out_f.write(pose_velocity_string)
        out_f.close()
    return pose_velocity



def calculate_cartesian_pose_bounding_box_from_aligned_training_data(input_dir, \
                                        elementary_action, motion_primitive, \
                                        recalculate=False):

    """
    Creates a file path based on the input parameters and tries to load a
    cartesian bounding box for a motion primitive.
    If the file does not exists a directory path is generated and a new
    cartesian bounding box is calculated using the BVH files found in this 
    directory. The resulting bounding box is then saved to the previously 
    generated file path in order to be loaded automatically on the next run.
    """
    file_name = input_dir+"\\"+ elementary_action+"_"+motion_primitive+".cbb"
    if os.path.isfile(file_name) and not recalculate:
        print "read "+file_name
        in_f = open(file_name, "rb")
        #http://stackoverflow.com/questions/6921699/can-i-get-json-to-load-into-an-ordereddict-in-python
        cbb = json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(in_f.read())
        in_f.close()
    else:
        directory_path = get_input_data_folder(elementary_action, \
                                motion_primitive, sub_dir_name="4 - Alignment")
        assert os.path.isdir(directory_path)
        print "write "+file_name

        cbb = calculate_cartesian_pose_bounding_box_from_directory(directory_path)
        cbb_string = json.dumps(cbb)
        out_f = open(file_name, "wb")
        out_f.write(cbb_string)
        out_f.close()
    return cbb

def save_motion_to_bvh_file(out_dir, prefix, elementary_action ,motion_primitive,\
                                                       bvh_reader, quat_frames):

    """
    Generates filename based on parameters and writes the motion to
    """
    filename = out_dir+"\\"+prefix+elementary_action+"_"+motion_primitive+"_"+\
              unicode(datetime.datetime.now().strftime("%d%m%y_%H%M%S"))+".bvh"
    BVHWriter(filename, bvh_reader, quat_frames, bvh_reader.frame_time, \
                                                            is_quaternion=True)


def report(out_dir, elementary_action ,motion_primitive, outliers,\
                         n_samples, epsilons, duration, write_output=True):

    """
    Prints the result to console and if xlswriter is installed creates an Excel
    report file
    """
    filename = out_dir + "\\"+"report"+ \
             unicode(datetime.datetime.now().strftime("%d%m%y_%H%M%S"))+".xlsx"
    print len(outliers)," outliers from", n_samples, "samples"
    bb_count =0
    cbb_count = 0
    velocity_count = 0
    exception_count = 0
    for o in outliers:
        if "bb" in o["test_results"].keys() and o["test_results"]["bb"]:
            bb_count+=1
        if "cbb" in o["test_results"].keys() and o["test_results"]["cbb"] :
            cbb_count+=1
        if "velocity" in o["test_results"].keys() and o["test_results"]["velocity"]:
            velocity_count+=1
        if "exception" in o["test_results"].keys() and o["test_results"]["exception"]:
            exception_count+=1
#    bb_count =len([o for o in outliers if o["test_results"]["bb"]])
#    cbb_count =len([o for o in outliers if o["test_results"]["cbb"] ])
#    velocity_count =len([o for o in outliers if o["test_results"]["velocity"]])
#    exception_count =len([o for o in outliers if o["test_results"]["exception"]])
    print "parameter bounding_box:" ,bb_count
    print "cartesian bounding_box:" ,cbb_count
    print "velocity:" ,velocity_count
    print "exceptions:" ,exception_count
    print "took", n_samples, "samples in", duration,"seconds"

    if write_output:
        try:
            import xlsxwriter

            print filename
            workbook = xlsxwriter.Workbook(filename)
            worksheet = workbook.add_worksheet()
            width = 30
            worksheet.set_column(0, 0, width)
            worksheet.set_column(0, 1, width)
            worksheet.set_column(0, 2, width)
            worksheet.set_column(0, 3, width)
            worksheet.set_column(0, 4, width)
            worksheet.set_column(0, 5, width)

            row = 0
            worksheet.write(row, 0, "elementary action")
            worksheet.write(row, 1, elementary_action)
            worksheet.write(row, 2, "motion primitive")
            worksheet.write(row, 3,   motion_primitive)

            row += 1
            worksheet.write(row, 0, "number of samples")

            worksheet.write(row, 1, n_samples)
            worksheet.write(row, 2, "total number of outliers")
            worksheet.write(row, 3, len(outliers))
            worksheet.write(row, 4, "time in seconds")
            worksheet.write(row, 5, duration)

            row += 1
            worksheet.write(row, 0, "bounding_box")
            worksheet.write(row, 1, bb_count)
            worksheet.write(row, 2, "epsilon")
            worksheet.write(row, 3, epsilons["bb"])
            row += 1
            worksheet.write(row, 0, "cartesian_bounding_box")
            worksheet.write(row, 1, cbb_count)
            worksheet.write(row, 2, "epsilon")
            worksheet.write(row, 3, epsilons["cbb"])
            row += 1
            worksheet.write(row, 0, "velocity")
            worksheet.write(row, 1, velocity_count)
            worksheet.write(row, 2, "epsilon")
            worksheet.write(row, 3, epsilons["velocity"])

            row += 1
            worksheet.write(row, 0, "exceptions")
            worksheet.write(row, 1, exception_count)

            workbook.close()

        except IOError as exception:
            print "I/O error({0}): {1}".format(exception.errno, \
                                               exception.strerror)
        except:
            print sys.exc_info()[0]
            print "If python cannot find the xlswriter module, \
                   type 'pip install xlsxwriter' into the command line"


def get_mm_directory(morphable_model_type,elementary_action):
    """
    Generates the path string for the given morphable model in the repository
    """
    data_dir_name = "data"
    process_step_dir_name = "3 - Motion primitives"

    mm_dir = os.sep.join([ROOT_DIR,
                              data_dir_name,
                              process_step_dir_name,
                              "motion_primitives_"+morphable_model_type,
                              elementary_action])
    return mm_dir

def save_s_vectors_to_file(output_dir, elementary_action, motion_primitive, s_vectors):
    """
    Dumps a list of low dimensional s-vectors from a Motion Primitive
    to a json file
    """
    motion_name = elementary_action+"_"+motion_primitive
    filename = os.sep.join([output_dir, motion_name+"_s_vectors"+\
    unicode(datetime.datetime.now().strftime("%d%m%y_%H%M%S"))+".json"])
    print "save s-vectors to file",filename
    with open(filename, 'wb') as outfile:
        output = {}
        output["motiline on_name"] = motion_name
        output["motion_data"] = [s.tolist() for s in s_vectors]
        json.dump(output, outfile)
        outfile.close()


def run_tests(elementary_action, motion_primitive, bvh_reader, mp,\
              test_configuration, n_samples, output_dir):
    """
    Runs a list of tests for a given number of samples of a morphable model 
    to check whether the samples are close to the original training data set. The result is 
    stored in an Excel report file.

    Parameters
    ----------
    * test_configuration: Tuple
    \tA tuple with tree values containing the dictionaries for the
    test_functions, parameters and epsilons (error threshold)

    """
    print "start"
    test_functions, parameters,  epsilons = test_configuration
    start = time.time()
    outliers =[]

    for i in xrange(n_samples):
        print "sample",i

        sample_s = mp.sample(return_lowdimvector=True)
        test_results = {}
        try:
          
          quat_frames = mp.back_project(sample_s).get_motion_vector()
     
          euler_frames = convert_quaternion_to_euler(quat_frames[:].tolist()) 
          for key in test_functions.keys():
                test_results[key] = not test_functions[key](bvh_reader,
                                                      euler_frames, \
                                                      parameters[key], \
                                                      eps=epsilons[key])
        except rpy2.rinterface.RRuntimeError as e:
            print e
            #exceptions.append({"id":i,"s_vector":sample_s})
            for key in test_functions.keys():
                test_results[key] = False
            test_results={"exception":True}
            outlier ={"id":i, \
                    "test_results":test_results}
            outliers.append(outlier)
             #ignore sample                                            
        if np.any(test_results.values()):
            if not "exception" in test_results.keys():
                #export motion if there was no exception
                prefix = ""
                for test in test_results.keys():
                    if test_results[test]:
                        prefix += test+"_"
                save_motion_to_bvh_file(output_dir, prefix, elementary_action,\
                                       motion_primitive, bvh_reader, quat_frames)

            outlier ={"id":i, \
                    "test_results":test_results}
            outliers.append(outlier)

    duration = time.time()-start
    report(output_dir, elementary_action, motion_primitive, outliers,\
            n_samples, epsilons, duration)






def export_good_s_vectors(elementary_action, motion_primitive, bvh_reader, mp,\
              test_configuration, n_samples, output_dir):
    """
    Generates n_samples good samples from a morphable model and saves the 
    s-vectors to an output file. In order to check whether the samples are 
    good, i.e. close to the original training data set, a list of tests is run 
    for each sample.

    Parameters
    ----------
    * test_configuration: Tuple
    \tA tuple with tree values containing the dictionaries for the
    test_functions, parameters and epsilons (error threshold))

    """
    print "start"
    test_functions, parameters,  epsilons = test_configuration
    start = time.time()
    outliers =[]
    s_vectors = []
    count = 0
    i = 0
    while count < n_samples: #generate good samples
        print "sample",i,count

        sample_s = mp.sample(return_lowdimvector=True)
        quat_frames = mp.back_project(sample_s).get_motion_vector()
        
        euler_frames = convert_quaternion_to_euler(quat_frames[:].tolist())
        test_results ={}
        for key in test_functions.keys():
            test_results[key] = not test_functions[key](bvh_reader,
                                                  euler_frames, \
                                                  parameters[key], \
                                                  eps=epsilons[key])
        if np.any(test_results.values()):
            #export motion
            prefix = ""
            for test in test_results.keys():
                if test_results[test]:
                    prefix += test+"_"
            save_motion_to_bvh_file(output_dir, prefix, elementary_action,\
                                   motion_primitive, bvh_reader, quat_frames)

            outlier ={"id":i, \
                    "test_results":test_results}
            outliers.append(outlier)
        else:
            s_vectors.append(sample_s)
            count+=1
        i+=1
    duration = time.time()-start
    save_s_vectors_to_file(output_dir, elementary_action, motion_primitive,
                           s_vectors)
    report(output_dir, elementary_action, motion_primitive, outliers,\
            i+1, epsilons, duration)



def verify_evaluation_using_test_data(elementary_action,motion_primitive,test_configuration):
    
    """
    Verify parameters using the original training data
    """
    
    #load cutted data
    directory_path = get_input_data_folder(elementary_action, motion_primitive)
    cutted_euler_frames_list = []
    for item in gen_file_paths(directory_path):
        bvh_reader_c = BVHReader(item)    
        if len(bvh_reader_c.keyframes) > 2:
            cutted_euler_frames_list.append(np.array(bvh_reader_c.keyframes) )    
    
    #load aligned data
    directory_path = get_input_data_folder(elementary_action, motion_primitive, sub_dir_name="4 - Alignment")
    
    aligned_euler_frames_list = []
    for item in gen_file_paths(directory_path):
        bvh_reader_a = BVHReader(item)    
        if len(bvh_reader_a.keyframes) > 2:
            aligned_euler_frames_list.append(np.array(bvh_reader_a.keyframes) )
        


    test_functions, parameters,  epsilons = test_configuration
    start = time.time()
    outliers =[]
    i = 0
    n1 = len(aligned_euler_frames_list)
    n2 = len(cutted_euler_frames_list)
    n = min(n1,n2)
    print "n",n1,n2,n

    while i < n: #generate good samples
        print "sample",i
        
        euler_frames = aligned_euler_frames_list[i]# convert_quaternion_to_euler(quat_frames[:].tolist())
        test_results = {}
        test_results["velocity"] = not test_functions["velocity"](bvh_reader_c, cutted_euler_frames_list[i], \
                                                         parameters["velocity"],\
                                                         epsilons["velocity"],bip_present=True) #
        test_results["cbb"] = not test_functions["cbb"](bvh_reader_a, \
                                                  euler_frames, \
                                                  parameters["cbb"], \
                                                  eps=epsilons["cbb"], \
                                                  bip_present=True)

        test_results["bb"] = not test_functions["bb"](bvh_reader_a, euler_frames, parameters["bb"], epsilons["bb"],True)

#        for key in test_functions.keys():
#            test_results[key] = not test_functions[key](bvh_reader,
#                                                  euler_frames, \
#                                                  parameters[key], \
#                                                  eps=epsilons[key], \
#                                                  bip_present=True)
        if np.any(test_results.values()):

            outlier ={"id":i, \
                    "test_results":test_results}
            outliers.append(outlier)
   
        i+=1
        

    duration = time.time()-start

    bb_count =len([o for o in outliers if o["test_results"]["bb"]])
    cbb_count =len([o for o in outliers if o["test_results"]["cbb"] ])
    velocity_count =len([o for o in outliers if o["test_results"]["velocity"]])
    exception_count = 0
    print "parameter bounding_box:" ,bb_count
    print "cartesian bounding_box:" ,cbb_count
    print "velocity:" ,velocity_count
    print "exceptions:" ,exception_count
    print "duration", duration
    return


def verify_bounding_box(elementary_action, motion_primitive, bb_type):
    print "start"
    if bb_type == "cbb":
        directory_path = get_input_data_folder(elementary_action,motion_primitive, sub_dir_name="4 - Alignment")
    else:
        directory_path = get_input_data_folder(elementary_action,motion_primitive)

    bounding_boxes = []
    for item in gen_file_paths(directory_path):
        print "load",item
        bvh_reader = BVHReader(item)    
        
        #euler_frames_list.append(np.array(bvh_reader.keyframes))
        if bb_type == "cbb":
            bb = calculate_cartesian_pose_bounding_box(bvh_reader)
        else:
            bb = calculate_parameter_bounding_box(bvh_reader)
        bounding_boxes.append(bb)
    print "check if bounding box is correct"
    #calculate overall bounding box
    if bb_type == "cbb":
        overall_bb = collections.OrderedDict()
        for c in ["X", "Y", "Z"]:
            overall_bb[c] = {"min":np.inf,"max":-np.inf}
      
        for bb in bounding_boxes:
            for c in bb.keys():
                update_bb_value(overall_bb[c], bb[c]["min"])
                update_bb_value(overall_bb[c], bb[c]["max"])
        cbb = calculate_cartesian_pose_bounding_box_from_directory(directory_path)
        result =  overall_bb == cbb
        print overall_bb,cbb
    else:
         overall_bb = collections.OrderedDict()#{}
         for node_name in  bounding_boxes[0].keys():
            overall_bb[node_name] =collections.OrderedDict()
            for c in bounding_boxes[0][node_name].keys():
                overall_bb[node_name][c] = {"min":np.inf,"max":-np.inf}
    
         for bb in bounding_boxes:
            for node_name in bb.keys():
                for c in bb[node_name].keys():
                    update_bb_value(overall_bb[node_name][c], \
                                    bb[node_name][c]["min"])
                    update_bb_value(overall_bb[node_name][c], \
                                    bb[node_name][c]["max"])
         pose_bb = calculate_parameter_bounding_box_from_directory(directory_path)
         result =  overall_bb == pose_bb
         print overall_bb,pose_bb
    print result
    return result

def verify_bounding_box2(elementary_action, motion_primitive, bb_type, update= False , pose_bb=None):
    print "start"
    if bb_type == "cbb":
        directory_path = get_input_data_folder(elementary_action,motion_primitive, sub_dir_name="4 - Alignment")
    else:
        directory_path = get_input_data_folder(elementary_action,motion_primitive)

    euler_frames_list = []
    for item in gen_file_paths(directory_path):
        print "load",item
        bvh_reader = BVHReader(item)    
        euler_frames_list.append(np.array(bvh_reader.keyframes))

    print "check if bounding box is correct"
    result= True
    #calculate overall bounding box
    if bb_type == "cbb":
        eps = 10
        cbb = calculate_cartesian_pose_bounding_box_from_directory(directory_path)
        i = 0         
        for euler_frames in euler_frames_list:
            i += 1
            print i
            if not check_cartesian_pose_bounding_box(bvh_reader, euler_frames, cbb, eps, True):
                result =  False
                break

    else:
        eps = 10
        if not pose_bb:
            pose_bb = calculate_parameter_bounding_box_from_directory(directory_path)
        pose_bb2 = copy(pose_bb) 
        if update:
            i = 0        
            for euler_frames in euler_frames_list:
                i += 1
                print i
                #if not 
                check_parameter_bounding_box(bvh_reader, euler_frames, pose_bb2, eps, True, True)
        i = 0
        for euler_frames in euler_frames_list:
            i += 1
            print i
            if not check_parameter_bounding_box(bvh_reader, euler_frames, pose_bb2, eps, True, False):
                result = False
                break
                           
             
    print result
    #print pose_bb==pose_bb2
    return result 
    
def generate_test_configuration(output_dir, elementary_action, 
                                motion_primitive, recalculate=False):
    """

    Returns
    ---------------
    A tuple with tree values containing the dictionaries for the
    test_functions, parameters and epsilons (error threshold)

    """
    epsilons = {}
    epsilons["bb"]= 10# 50# use 10 for walk
    epsilons["cbb"] = 5
    epsilons["velocity"] = 1

    parameters = {}
    parameters["bb"]  = create_parameter_bounding_box_from_training_data(output_dir,\
                                                                        elementary_action,\
                                                                        motion_primitive, \
                                                                        recalculate)
    parameters["cbb"] = calculate_cartesian_pose_bounding_box_from_aligned_training_data(output_dir,\
                                                                                elementary_action,\
                                                                                motion_primitive, \
                                                                                recalculate= False)
    parameters["velocity"] = calculate_avg_velocity_from_training_data(output_dir,elementary_action,\
                                                                        motion_primitive, \
                                                                        recalculate)


    test_funcs = {}
    test_funcs["bb"] = check_parameter_bounding_box
    test_funcs["velocity"] = check_average_velocity
    test_funcs["cbb"] = check_cartesian_bounding_box
    return test_funcs,parameters,epsilons

def calculate_boundary_data(elementary_action,motion_primitive):
    """Calculate pose bb, cartesian bb, and avg velocity
    """
    cutted_directory_path =  get_input_data_folder(elementary_action,motion_primitive)
    aligned_directory_path = get_input_data_folder(elementary_action,motion_primitive, sub_dir_name="4 - Alignment")
    print "input dir",cutted_directory_path
    print "input dir",aligned_directory_path
    
    assert os.path.isdir(aligned_directory_path) and os.path.isdir(cutted_directory_path)
   
    dictionary  = {}
    dictionary["pose_bb"] = calculate_parameter_bounding_box_from_directory2(aligned_directory_path)
    dictionary["pose_velocity"]  = calculate_average_velocity_from_directory(cutted_directory_path)
    dictionary["cartesian_bb"] = calculate_cartesian_pose_bounding_box_from_directory(cutted_directory_path)
    
    return dictionary

def calculate_bounding_boxes():
    """Calculates bounding boxes for all motion primitives
    """
    morphable_model_root = get_morphable_model_directory()

    
    for key in next(os.walk(morphable_model_root))[1]:
            elementary_action_directory = morphable_model_root+os.sep+key
            elementary_action = key
            elementary_action_name = key.split("_")[-1]
            print elementary_action_directory

            for root, dirs, files in os.walk(elementary_action_directory):
                for file_name in files:#for each morphable model 
              
                #print file_name
                    if file_name.endswith("mm.json") :
                        motion_primitive =file_name.split("_")[1]
                        print  motion_primitive
                        boundary_data = calculate_boundary_data(elementary_action,motion_primitive)
                        boundary_string = json.dumps(boundary_data)
                        file_name = elementary_action_directory+"\\"+ elementary_action_name+"_"+motion_primitive+".stats"
                        out_f = open(file_name, "wb")
                        out_f.write(boundary_string)
                        out_f.close()
    return


def run_evaluation():
    """
    Performs the evaluation of a motion primtive based on the training data by
    checking if samples break the pose parameter bounding box or their
    average velocity is too far away from the training data.
    """
    bvh_reader = BVHReader("skeleton.bvh")
    elementary_action= 'elementary_action_carry'
    #motion_primitive = 'rightStance'
    motion_primitive = 'leftStance'
    #motion_primitive = 'beginLeftStance'
    #motion_primitive = 'beginRightStance'
    #motion_primitive = 'endLeftStance'
    #motion_primitive = 'endRightStance'
    #elementary_action= 'elementary_action_pick'
    #motion_primitive = 'first'
    mm_type = "quaternion_PCA95" #"motion_primitives_quaternion_PCA95"
                       

    
    
    mm_file_directory = get_mm_directory(mm_type, elementary_action)
    #mm_file = mm_file_directory+"\\" +"carry_rightStance_quaternion_mm.json"
    mm_file = mm_file_directory+"\\" +"carry_leftStance_quaternion_mm.json"
    #mm_file = mm_file_directory+"\\" +"carry_beginLeftStance_quaternion_mm.json"
    #mm_file = mm_file_directory+"\\" +"carry_beginRightStance_quaternion_mm.json"
    #mm_file = mm_file_directory+"\\" +"carry_endLeftStance_quaternion_mm.json"
    #mm_file = mm_file_directory+"\\" +"carry_endRightStance_quaternion_mm.json"
    #mm_file = mm_file_directory+"\\" +"pick_first_quaternion_mm.json"
        
    mp = MotionPrimitive(mm_file)
    print mm_file
    output_dir = get_output_folder(elementary_action, motion_primitive, mm_type)
    n_samples =  1000
    recalculate = True
    mode = "verify"# , "export_bad_samples" # "export_good_samples"

    test_configuration = generate_test_configuration(output_dir, \
                                                     elementary_action,\
                                                     motion_primitive,\
                                                     recalculate)
  
    if mode == "verify":
        bb_type = "bb"#"cbb"
#        #check for keys
#        pose_bb = test_configuration[1]["bb"]
#        directory_path = get_input_data_folder(elementary_action,motion_primitive)
#        pose_bb2 = calculate_parameter_bounding_box_from_directory(directory_path)
#        print pose_bb["Hips"].keys()
#        print pose_bb2["Hips"].keys() 
#        verify_bounding_box2(elementary_action, motion_primitive,bb_type,False,pose_bb)
#        verify_bounding_box2(elementary_action, motion_primitive,bb_type,False,pose_bb2)
        #verify_bounding_box2(elementary_action, motion_primitive,bb_type,False)
        verify_evaluation_using_test_data(elementary_action, motion_primitive,test_configuration)
    elif mode == "export_good_samples":
        export_good_s_vectors(elementary_action, motion_primitive, bvh_reader, mp,\
                  test_configuration , n_samples, output_dir)
    elif mode == "export_bad_samples":
        run_tests(elementary_action, motion_primitive, bvh_reader, mp,\
                  test_configuration , n_samples, output_dir)

def main():
    
    calculate_bounding_boxes()
    #run_evaluation()
    


if __name__=='__main__':
    main()
