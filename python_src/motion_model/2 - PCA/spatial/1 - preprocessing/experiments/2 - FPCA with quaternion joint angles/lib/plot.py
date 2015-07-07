# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:10:21 2014

@author: erhe01

"""
import numpy as np   
import matplotlib.pyplot as plt
from quaternion_frame import QuaternionFrame
from cgkit.cgtypes import quat
import glob
from bvh import BVHReader
    
def _plot_joint(bvh_reader,joint_name,fig,sub_plot_coordinate = (1,1,1),filter_values = True):
    """Plots the quaternion values of one joint over all frames using the specified subplot"""
     
    if joint_name in bvh_reader.node_names.keys():
        values = []
        #last_quat_value = [-0.25,-0.25,-0.25,-0.25]
        #last_quat_value = [0.25,0.25,0.25,0.25]
        last_quat_value = None      
        last_quat_set = False
        for i in xrange(len(bvh_reader.keyframes) ):
            quat_frame = QuaternionFrame(bvh_reader,i,filter_values)
            quat_value =  np.array(quat_frame[joint_name])
            if last_quat_set and filter_values:
                #http://physicsforgames.blogspot.de/2010/02/quaternions.html
                #get dot product to see if they are far away from each other
                dot = quat_value[0]*last_quat_value[0] + quat_value[1]*last_quat_value[1] + quat_value[2]*last_quat_value[2] + quat_value[3]*last_quat_value[3]
    
                #if they are far away then flip the sign
                if dot < 0:
                    quat_value = [-v for v in quat_value]               
                    
            values.append(quat_value)
            last_quat_value = quat_value
            last_quat_set = True
           
        ax= fig.add_subplot(*sub_plot_coordinate)
        ax.set_title(joint_name)
        ax.plot(values) 
        
def plot_bvh_values(bvh_reader,max_columns = 3,filter_values = False):
    """Plots the quaternion values of all joints over all frames using several subplots"""
    
    fig = plt.figure()
    #joints = ["RightHand","RightFoot","LeftShoulder","LeftHand","LeftFoot","LeftLeg","RightLeg","RightShoulder"]
    joints = [key for key in bvh_reader.node_names.keys()  
                if not bvh_reader.node_names[key].isEndSite()
                   and not key.startswith("Bip")]
    number_of_plots = len(joints)
    
    #find number of rows and columns
    rows = int(number_of_plots/max_columns)
    remainder = number_of_plots%max_columns
    if rows == 0.0:
        columns = number_of_plots
        rows = 1
    else:
        columns = max_columns
    if remainder > 0:#add additional row if there is a remainder
        rows+=1    
    [_plot_joint(bvh_reader,joints[idx],fig,(rows,columns,idx),filter_values) 
                                for idx in xrange(number_of_plots) ]
                                    



def matrix_to_euler(matrix,rotation_order):
    '''
    returns the angles in a three dimensional vector where the x/y/z component of the vector represents the rotation around the x/y/z axis independent of the rotation order
    rotation order is needed to extract the correct angles
    when the angles are converted into a matrix or a quaternion or saved as a bvh format string the rotation order needs to be applied
    TODO use better and faster code by Ken Shoemake in Graphic Gems 4, p.222
    http://thehuwaldtfamily.org/jtrl/math/Shoemake,%20Euler%20Angle%20Conversion,%20Graphic%27s%20Gems%20IV.pdf
    https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2012/07/euler-angles1.pdf
    '''
    from math import degrees

    if rotation_order[0] =='X':
          if rotation_order[1] =='Y':
              vector = matrix.toEulerXYZ()
          elif rotation_order[1] =='Z':
              vector = matrix.toEulerXZY()
    elif rotation_order[0] =='Y':
        if rotation_order[1] =='X':
             vector = matrix.toEulerYXZ()
        elif rotation_order[1] =='Z':
             vector = matrix.toEulerYZX()  
    elif rotation_order[0] =='Z': 
        if rotation_order[1] =='X':
            vector = matrix.toEulerZXY()  
        elif rotation_order[1] =='Y': 
            vector = matrix.toEulerZYX()  
    return [degrees(vector[0]),degrees(vector[1]),degrees(vector[2])] 
   

def get_motion_string(bvh_reader,filter_values = False):
    rotation_order = ['X','Y','Z']
    motion_string = ""
    last_quat_frame = None
    for frame_idx in xrange(len(bvh_reader.keyframes) ):

        root_translation =bvh_reader.keyframes[frame_idx][0:3]
        #print frame_idx,root_translation
        motion_string+= str(root_translation[0])+"\t"+str(root_translation[1])+"\t"+str(root_translation[2])+"\t"
        
        quat_frame = QuaternionFrame()
        quat_frame.fill(bvh_reader,frame_idx)
        for joint_name in bvh_reader.node_names:
          
            if joint_name in quat_frame.keys():
                quat_value = quat_frame[joint_name]
                if filter_values and last_quat_frame != None:
                
                    #http://physicsforgames.blogspot.de/2010/02/quaternions.html
                    #get dot product to see if they are far away from each other
                    #dot = quat_value[0]*last_quat_frame[joint_name][0] + quat_value[1]*last_quat_frame[joint_name][1] + quat_value[2]*last_quat_frame[joint_name][2] + quat_value[3]*last_quat_frame[joint_name][3]
                    dot = quat_value[0] + quat_value[1] + quat_value[2] + quat_value[3]#dot product with [1,1,1,1]
                                   
                    #if they are then flip the sign
                    if dot < 0:
                        flipped_quat_value = [-v for v in quat_value]         
                    else:
                        flipped_quat_value = [v for v in quat_value]                    
                    #signs = np.sign(quat_value)
    #                quat_value = [quat_value[i]*signs[i] if signs[i]!= 0 else 
    #                                                        quat_value[i]
    #                                                        for i in xrange(4)]      
                
                else:
                    flipped_quat_value = quat_value
#                signs = np.sign(quat_value)
#                print signs
#                flipped_quat_value = [quat_value[j]*signs[j] if signs[j]!= 0 else quat_value[j]
#                                                                    for j in xrange(4)] 
                flipped_quaternion = quat(flipped_quat_value).normalize()
                m = flipped_quaternion.toMat3()
                euler_angles2  = matrix_to_euler(m,rotation_order)
                #else:
                #get euler angles without flip
                quaternion = quat(quat_value).normalize()
                m = quaternion.toMat3()
                euler_angles1  = matrix_to_euler(m,rotation_order)
                    
                print map(round,euler_angles1) ,map(round,euler_angles2)
                print map(round,euler_angles1) == map(round,euler_angles2)
               
                motion_string += str(euler_angles1[0])+"\t"+str(euler_angles1[1])+"\t"+str(euler_angles1[2])      
                motion_string +="\t"
            elif not bvh_reader.node_names[joint_name].isEndSite():#if it is a finger add 0 0 0 as angles
                motion_string += "0\t0\t0"      
                motion_string +="\t"
        motion_string +="\n"
        last_quat_frame = quat_frame
    return motion_string      
   
def get_frames(bvh_reader,joint_name,filter_values = True):
    values = []
    if joint_name in bvh_reader.node_names.keys():
      
        last_quat_value = None      
        last_quat_set = False
        for i in xrange(len(bvh_reader.keyframes) ):
            quat_frame = QuaternionFrame(bvh_reader,i,filter_values)
            quat_value =  np.array(quat_frame[joint_name])
            if last_quat_set and filter_values:
                #http://physicsforgames.blogspot.de/2010/02/quaternions.html
                #get dot product to see if they are far away from each other
                dot = quat_value[0]*last_quat_value[0] + quat_value[1]*last_quat_value[1] + quat_value[2]*last_quat_value[2] + quat_value[3]*last_quat_value[3]
        
                #if they are far away then flip the sign
                if dot < 0:
                    quat_value = np.array([-v for v in quat_value]               )
                    
            values.append(quat_value)
            last_quat_value = quat_value
            last_quat_set = True
            
    return np.array(values)

def get_mean_parameters(animation_list,joint_name,filter_values = True):
    ''' reads all animations in the input dir and calculates the mean animation 
        parameters of the given joint
    '''
    number_of_frames = 91#len(bvh_reader.keyframes)
    sum_of_frames = np.zeros((number_of_frames,4) ) 
    i=0
    for bvh_reader in animation_list:
        print i
        frames = get_frames(bvh_reader,joint_name,True)
        i+=1
        print frames.shape
        sum_of_frames += np.array(frames)
        
    print sum_of_frames
    mean = sum_of_frames /number_of_frames
    print mean
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(mean)
    
def get_animation_list(input_dir):
    animation_list = []
    for filepath in glob.glob(input_dir):
        print filepath
        bvh_header = BVHReader(filepath)
        bvh_header.fhandle.close()
        animation_list.append(bvh_header)
    return animation_list
                   
def main():
    
    #filepath = "test/walk_001_1_rightStance_86_128.bvh"
    input_dir = r'C:\Users\hadu01\MG++\repo\data\1 - MoCap\4 - Alignment\elementary_action_pick\first\*.bvh'
    filepath = glob.glob(input_dir)[40]
    print filepath
    
    bvh_reader = BVHReader(filepath)
    #quat_frame = QuaternionFrame(bvh_reader,0,True)
    #print quat_frame.values()
    plot_bvh_values(bvh_reader,3,True)
    
    joint_name = "LeftShoulder"
    #frames = get_frames(bvh_reader,joint_name,True)
    #print frames
#    \et_mean_parameters(get_animation_list(input_dir),joint_name,True)
        #motion_string =  get_motion_string(bvh_reader,True)
    #    with  open('motionstring.bvh', 'wb') as outfile:
    #       outfile.write(motion_string)
    #    outfile.close()
#    print "done"
    number_of_frames = len(bvh_reader.keyframes)
    sum_of_frames = np.zeros((number_of_frames,4) ) 
    for filepath in glob.glob(input_dir):
        bvh_reader = BVHReader(filepath)
        frames = get_frames(bvh_reader,"LeftShoulder",True)
    sum_of_frames += np.array(frames)
    
    print sum_of_frames
    mean = sum_of_frames /number_of_frames
    print mean
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(mean)
   

            
   

if __name__ == '__main__':
    main()
