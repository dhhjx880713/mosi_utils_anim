# -*- coding: utf-8 -*-
#===============================================================================
# author: Erik Herrmann (DFKI GmbH, FB: Agenten und Simulierte Realit�t)
# last update: 2.4.2014
#===============================================================================
import os
import re
import numpy as np
from cgkit.cgtypes import *
from AnimationEngine import JointData


class BVHSkeletonReader():
    '''
    #code loosely based on the BVH LOADER v1.0 for modo 601 by Lukasz Pazera 2012 from http://community.thefoundry.co.uk/discussion/topic.aspx?f=4&t=69994
    '''

    def loadFile(self,filePath,animationData):
        animationData.filePath = filePath
        splittedPath = os.path.split(filePath)
        animationData.name = splittedPath[len(splittedPath)-1]
        #read file
        try:
            fo = open(animationData.filePath)

            bvh = fo.readlines()
            fo.close()
        except:
            print("failed loading file "+animationData.filePath)
            return

        #parse file

        line_index = self.readHierarchy(bvh,animationData)
        if line_index > 0:
            #search for the start of the motion data
            line_index = 0
            while line_index < len(bvh):
                if re.match("(.*)MOTION(.*)",bvh[line_index]):
                    break
                line_index +=1
            #read motion data
            self.readAllAnimationFrames(animationData, bvh,line_index)
            animationData.loadedCorrectly =True
            #animationData.calculateBoundingBoxFromStructure()
        return line_index > 0


    def readHierarchy(self,bvh,animationData):
        line_index = 0

        if re.match("(.*)HIERARCHY(.*)", bvh[line_index]):
            jointCount = 0#counter for real joints ignoring endsites, needed for optimization of parameters to assiociate a joint with its eigenvector
            #add root joint
            line_index = 1
            animationData.root.type = JointData.TYPE_ROOT
            root_line = bvh[line_index]
            name_and_type = root_line.split()

            animationData.root.name = name_and_type[1]
            animationData.root.level = animationData.currentLevel
            animationData.root.number = jointCount
            jointCount+=1
             #find offset
            while line_index < len(bvh):
                if re.match("(.*)OFFSET(.*)",bvh[line_index]):
                    break
                line_index +=1
            offset_line = (bvh[line_index]).split() 
            animationData.root.setOffset(float(offset_line[1]),float(offset_line[2]),float(offset_line[3]),)
            #animationData.setRootOffset(float(offset_line[1]),float(offset_line[2]),float(offset_line[3]))   
                      
             #find channels
            while line_index < len(bvh):
                if re.match("(.*)CHANNELS(.*)",bvh[line_index]):
                    break
                line_index +=1

            self.setJointChannelOrder(bvh[line_index],animationData.root)
           # animationData.setRootChannelOrder(bvh[line_index])
            #jointOrder.append(animationData.root.name)
            animationData.joints[animationData.root.name] = animationData.root

            #add child joints
            while line_index < len(bvh):

               line = bvh[line_index]
               if re.match("(.*)JOINT(.*)|(.*)End Site(.*)", line):

                   animationData.currentLevel +=1
                   jointCount = self.addChildJointFrameData(animationData, bvh,line_index,jointCount)

               elif re.match("(.*)}(.*)", line):
                   animationData.currentLevel -=1

               elif re.match("(.*)MOTION(.*)",line):
                   break;
               line_index +=1
            #print(len(animationData.joints))     
            return line_index

        else:
            return line_index

    def addChildJointFrameData(self,animationData, bvh,line_index,jointCount):
        joint = JointData.JointFrameData()
         #type and name
        name_and_type = (bvh[line_index]).split()
        if name_and_type[0] != "End":#JOINT with name and children
           joint.type = JointData.TYPE_JOINT   
           joint.name = name_and_type[1]
           #print(joint.name)  
           joint.level = animationData.currentLevel
           

           #find offset "OFFSET    X     Y    Z
           while line_index < len(bvh):
               if re.match("(.*)OFFSET(.*)",bvh[line_index]):
                   break
               line_index +=1

           #print(bvh[line_index])
           offset_line = (bvh[line_index]).split() 
           joint.setOffset(float(offset_line[1]),float(offset_line[2]),float(offset_line[3]))
           #find channels
           while line_index < len(bvh):
               if re.match("(.*)CHANNELS(.*)",bvh[line_index]):
                   break
               line_index +=1
           self.setJointChannelOrder(bvh[line_index],joint)
           joint.number = jointCount
           jointCount+=1
        else:#END SITE without name or children
           joint.type = JointData.TYPE_END
           joint.level = animationData.currentLevel
           joint.numberOfChannels = 0
           #find offset "OFFSET    X     Y    Z
           while line_index < len(bvh):
               if re.match("(.*)OFFSET(.*)",bvh[line_index]):
                   break
               line_index +=1

           offset_line = (bvh[line_index]).split()
           joint.setOffset(float(offset_line[1]),float(offset_line[2]),float(offset_line[3]))
           joint.number = -1 # endsites don't have any parameters needed for the optimization so they need to be ignored
        joint_index = len(animationData.joints)-1
        #note animationData joints has to be a ordered dictionary
        for key in reversed(animationData.joints.keys()):#while joint_index >=0:#search for parent joint and the new joint to its child list
           if animationData.joints[key].level < animationData.currentLevel :
               animationData.joints[key].addChild(joint)
               joint.parent = animationData.joints[key]
               break
           joint_index -=1
        #add joint to overall joint list
        #jointOrder.append(joint)
        if joint.type == JointData.TYPE_END:
            joint.name = joint.parent.name+"EndSite"+str(len(joint.parent.children))
        animationData.joints[joint.name] = joint

        return jointCount
    #reads the BVH file after the "MOTION"-line. 
    #takes the bvh file returned by file.readlines() and the startLine (= "MOTION"-line) as parameters              
    def readAllAnimationFrames(self,animationData, bvh,startLine):
        line_index = startLine+1
        
        #read number of frames and frame time
        num_frames = bvh[line_index].split()

        animationData.numberOfFrames =float(num_frames[1])
        line_index += 1

        frame_time = bvh[line_index].split()
        animationData.frameTime = float(frame_time[2])
        
        animationData.maxAnimationTime = animationData.frameTime*animationData.numberOfFrames
        
        #read frames
        #max_line_number = line_index + self.numberOfFrames
        line_index += 1
        print "reading " + str(animationData.numberOfFrames)+" frames ..."
        line_index = self.readAnimationFramesFromLines(bvh,line_index,animationData.numberOfFrames,animationData)
        print "done reading at line"+str(line_index) 
        return line_index
#         while line_index < max_line_number:
#             self.parseFrameLine(bvh[line_index])
#             line_index +=1
            #print("here it goes inside"+str(line_index))
        #print("done")  
        
        #print("done")


    def readAnimationFramesFromLines(self,lines,offset,length,animationData):

        index = offset
        len = offset+length
        while index < len:
            #print(index)
            self.parseFrameLine(lines[index],animationData)
            index +=1
        return index
        #print("\n")

    def parseFrameLine(self, line,animationData):
        #add channel to joint based on parsed sequence
        channels = line.split()
        #animationData.addRootFrame(channels)
        joint_index = 0 # dont skip root
        channel_offset = joint_index
        #print("channels "+ str(len(channels)))
        #print("joints "+str(len(animationData.joints)))
        for joint in animationData.joints.itervalues():# < len(animationData.joints):
            if joint.numberOfChannels == 3:
                channel_offset = self.addRotationFrame(joint,channels,channel_offset)
            elif joint.numberOfChannels ==  6:
                channel_offset = self.addTransformationFrame(joint,channels,channel_offset)
            elif joint.numberOfChannels ==  0:# necessary when more than one root exist
                self.addZeroFrame(joint,channels)
                #channel_offset+= 1
            joint_index +=1



    def setJointChannelOrder(self, channelLine,joint):
        channels = channelLine.split()
        if channels[1] =='3' :#only rotation 
            #self.rotOrder = []
            channelIndex = 2
            joint.numberOfChannels = 3
            while channelIndex < len(channels):
                if channels[channelIndex] == "Xrotation":
                    joint.rotOrder.append('X') 
                elif channels[channelIndex] == "Yrotation":
                    joint.rotOrder.append('Y') 
                elif channels[channelIndex] == "Zrotation":
                    joint.rotOrder.append('Z') 
                channelIndex+=1  
        elif channels[1] =='6' :#translation and rotation
            joint.numberOfChannels = 6
            #self.rotOrder = []
            offset = 2
            channelIndex = 0
            while offset+channelIndex < len(channels):
                if channels[offset+channelIndex] == "Xposition":
                    joint.posOrder.append('X') 
                elif channels[offset+channelIndex] == "Yposition":
                    joint.posOrder.append('Y') 
                elif channels[offset+channelIndex] == "Zposition":
                    joint.posOrder.append('Z') 
                channelIndex+=1
                
            offset = 5
            channelIndex = 0
            while offset+channelIndex < len(channels):
                if channels[offset+channelIndex] == "Xrotation":
                    joint.rotOrder.append('X') 
                elif channels[offset+channelIndex] == "Yrotation":
                    joint.rotOrder.append('Y') 
                elif channels[offset+channelIndex] == "Zrotation":
                    joint.rotOrder.append('Z') 
                channelIndex+=1
        return

# 
#     def setRootChannelOrder(self, channelLine,joint):
#         channels = channelLine.split()
#         if channels[1] =='6' :#only translation and rotation supported
#             #self.rootPosOrder = []
#             #self.rootRotOrder = []
#             offset = 2
#             channelIndex = 0
#             while offset+channelIndex < len(channels):
#                 if channels[offset+channelIndex] == "Xposition":
#                     joint.posOrder.append('X') 
#                 elif channels[offset+channelIndex] == "Yposition":
#                     joint.posOrder.append('Y') 
#                 elif channels[offset+channelIndex] == "Zposition":
#                     joint.posOrder.append('Z') 
#                 channelIndex+=1
#                 
#             offset = 5
#             channelIndex = 0
#             while offset+channelIndex < len(channels):
#                 if channels[offset+channelIndex] == "Xrotation":
#                     joint.rotOrder.append('X') 
#                 elif channels[offset+channelIndex] == "Yrotation":
#                     joint.rotOrder.append('Y') 
#                 elif channels[offset+channelIndex] == "Zrotation":
#                     joint.rotOrder.append('Z') 
#                 channelIndex+=1
#         
#         return


    def addRotationFrame(self, joint, channels, startIndex):
        #print("current offset "+str(offset))
        #startIndex = 3 + offset *3
        #print("current start_index "+str(startIndex))
        frame = JointData.JointFrame()
        angles =[]
        channelIndex = startIndex
        while (channelIndex-startIndex)< len(joint.rotOrder):
            angles.append(float(channels[channelIndex]))
            channelIndex+=1
        frame.setRotationByEulerAngles(angles, joint.rotOrder)
        frame.EulerAnglesDeg = vec3(angles)
        joint.frames.append(frame)
        return channelIndex
        #print(joint.rotOrder)



    def addTransformationFrame(self,root,channels,startIndex):
        frame = JointData.JointFrame()

        currentAxis = 0
        channelIndex = startIndex
        while currentAxis < len(root.posOrder):  
            if root.posOrder[currentAxis] == 'X':
                frame.translation.x = float(channels[channelIndex])
            elif root.posOrder[currentAxis] == 'Y':
                frame.translation.y = float(channels[channelIndex])
            elif root.posOrder[currentAxis] == 'Z':
                frame.translation.z = float(channels[channelIndex])
            currentAxis+=1
            channelIndex+=1


        currentAxis = 0
        startIndex = channelIndex
        angles=[]
        while (channelIndex-startIndex)< len(root.rotOrder):
            angles.append(float(channels[channelIndex]))
            channelIndex+=1
        frame.setRotationByEulerAngles(angles, root.rotOrder)
        frame.EulerAnglesDeg = vec3(angles)
        root.frames.append(frame)

        return channelIndex
    
    def addZeroFrame(self,joint,channels):
        frame = JointData.JointFrame()
        joint.frames.append(frame)
    
 