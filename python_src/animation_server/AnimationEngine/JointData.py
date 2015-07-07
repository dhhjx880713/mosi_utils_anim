# -*- coding: utf-8 -*-
#===============================================================================
# author: Erik Herrmann (DFKI GmbH, FB: Agenten und Simulierte Realit�t)
# last update: 4.2.2014
#===============================================================================
import copy
import math
from Utilities import CustomMath
from cgkit.cgtypes import *
import numpy as np

# constants
TYPE_NONE     = 0
TYPE_ROOT    = 1
TYPE_JOINT     = 2
TYPE_END    = 3
STR_NONE    = ''


class Keyframe():
    def __init__(self):
        self.animationIndex = -1
        self.frameNumber = -1


class JointFrame(object):
    def __init__(self):
        self.EulerAnglesDeg = vec3()
        #self.rotationMatrix = CustomMath.get4x4IdentityMatrix() 
        self.rotationQuaternion = quat() # CustomMath.Quaternion(0,0,0,0)
        self.translation = vec3()
        
    def getRotationMatrix(self):
        #return self.rotationQuat.toMatrix() #self.rotationMatrix
        return  CustomMath.CGkitMatToNumpyArray4x4(self.rotationQuaternion.toMat4())# np.array(self.rotationQuaternion.toMat4().toList(), np.float32).reshape(4,4)
    
    
    def setRotationByEulerAngles(self,angles,rotOrder):
        '''
        applies 3d rotation around the X Y Z axes 
        angles: an ordered list of length 3 with the rotation angles given in degrees  
        rotOrder: a list that specifies the rotation axes of the angle with corresponding index in the angles list
        '''
        currentAxis = 0
        #print angles
        self.rotationQuaternion = None
        while currentAxis < len(rotOrder):
            deg = angles[currentAxis]
            if rotOrder[currentAxis] == 'X':
                #frame.EulerAnglesDeg.x = deg
                tempQuaternion = quat()#CustomMath.Quaternion(0,0,0,0)
                tempQuaternion.fromAngleAxis(math.radians(deg), vec3(1.0,0.0,0.0))
                if self.rotationQuaternion != None:
                    self.rotationQuaternion = self.rotationQuaternion*tempQuaternion #quaternion.multiply(tempQuaternion)
                else:
                    self.rotationQuaternion = tempQuaternion
                #frame.rotationMatrix = np.dot(CustomMath.getRotationAroundXAxis(math.radians(deg)), frame.rotationMatrix)
            elif rotOrder[currentAxis] == 'Y':
                #frame.EulerAnglesDeg.y = deg
                tempQuaternion = quat()#CustomMath.Quaternion(0,0,0,0)
                tempQuaternion.fromAngleAxis(math.radians(deg), vec3(0.0,1.0,0.0))
                if self.rotationQuaternion != None:
                    self.rotationQuaternion = self.rotationQuaternion*tempQuaternion #quaternion.multiply(tempQuaternion)
                else:
                    self.rotationQuaternion = self.rotationQuaternion
                #frame.rotationMatrix = np.dot(CustomMath.getRotationAroundYAxis(math.radians(deg)), frame.rotationMatrix)
            elif rotOrder[currentAxis] == 'Z':
                #frame.EulerAnglesDeg.z = deg
                tempQuaternion = quat()#CustomMath.Quaternion(0,0,0,0)
                tempQuaternion.fromAngleAxis(math.radians(deg), vec3(0.0,0.0,1.0))
                if self.rotationQuaternion != None:
                    self.rotationQuaternion = self.rotationQuaternion*tempQuaternion #quaternion.multiply(tempQuaternion)
                else:
                    self.rotationQuaternion = tempQuaternion
                #frame.rotationMatrix = np.dot(CustomMath.getRotationAroundZAxis(math.radians(deg)), frame.rotationMatrix)
            currentAxis+=1
        return    
    
    def getEulerAngles(self,rotOrder):
        '''
        returns the angles independent of the given joint rotation order as x y z component
        therefore it has to be reordered according to the rotation order before it can be saved as a bvh string
        '''
        return CustomMath.quaternionToEuler3(self.rotationQuaternion,rotOrder)
    
    def getEulerAnglesOrdered(self,rotOrder):
        '''
        returns the angles in the given joint rotation order
        note: the corresponding axis of an angle can be read from the rotation order
        ordered like this it can be directly saved as a BVH string
        '''
        temp =self.getEulerAngles(rotOrder) 
        count = 0
        eulerAnglesDeg = vec3(0)
        for axis in rotOrder:
            if axis =='X':
                eulerAnglesDeg[count] = temp.x
            elif axis == 'Y':
                eulerAnglesDeg[count] =temp.y
            elif axis == 'Z':
                eulerAnglesDeg[count] =temp.z
            count +=1
        return eulerAnglesDeg
       

#             
#     def switchAxisValues(self,index1,index2,index3,rotOrder):
# 
#         #switch angles
#         eulerAngles = self.getEulerAngles(rotOrder)
#         newEulerAngles =[0,0,0]
#         newEulerAngles[index1] = eulerAngles[index2]
#         newEulerAngles[index2] = eulerAngles[index1]
#         newEulerAngles[index3] = eulerAngles[index3]
#         self.setRotationByEulerAngles(eulerAngles, rotOrder)
  
    #degOffset contains the euler angles that are also contained in the quaternion q so  they dont have to be calculated for every frame
    # todo get angles from quaternion when saving so this is not necessary anymore
    def rotateByQuaternion(self,q):
        self.rotationQuaternion = q * self.rotationQuaternion
        self.translation  = q.rotateVec( self.translation)
        
    def transformByMatrix(self,transformationMatrix):
        rotationMatrix = self.rotationQuaternion.toMat4()
        self.rotationQuaternion.fromMat(transformationMatrix*rotationMatrix)
        self.translation  =transformationMatrix* self.translation
        return
        


  
        
class JointDescription(object):
    def __init__(self):
        self.name = ""
        self.level = -1
        self.number  = -1#gives reading order from the BVH file that is also used as order for eigen vectors
        self.type = TYPE_NONE
        self.parent = None
        self.children =[]
        self.offset = vec3(0.0) 
        self.offsetMatrix = CustomMath.getTranslationMatrix(self.offset)
        self.numberOfChannels = 0
        self.rotOrder =[]
        self.posOrder =[]
        
               
    def setOffset(self,x,y,z):
        self.offset.x = x               # position offset as in bvh file   
        self.offset.y = y        
        self.offset.z = z
        self.offsetMatrix = CustomMath.getTranslationMatrix(self.offset) 
        return
     
     
    def getFrameLength(self):
        '''
        returns recursively the number of parameters of one pose
        '''
        length = self.numberOfChannels
        for child in self.children:
            length+= child.getFrameLength()
        return length



class JointFrameData (JointDescription):
    def __init__(self) :
        JointDescription.__init__(self)
        #self.size = 0
        self.weight = 1#used for distance calculation
        
        #self.numberOfFrames = 0
        self.frames = []
        
    def copyAttributesFromOtherInstance(self,original):
        self.offset = original.offset
        self.offsetMatrix = original.offsetMatrix
        self.rotOrder = original.rotOrder
        self.posOrder = original.posOrder
        self.type = original.type
        self.name = original.name
        self.level = original.level
        self.numberOfChannels = original.numberOfChannels
      
    def copyDataFromOtherInstance(self,original,jointList,firstFrame, lastFrame):
        jointList[original.name]= self
        JointFrameData.__init__(self)#stupid error
        self.copyAttributesFromOtherInstance(original)
        self.parent = original.parent
        if self.type == TYPE_JOINT:
            self.frames= []
            index = firstFrame#0
            #lastFrame = len(original.frames)
            #print self.name
            #print len(original.frames)
            #print original.type
            while index <= lastFrame:
                self.frames.append(copy.deepcopy(original.frames[index]))
                #print(index)
                index += 1
            self.children = []
            for originalChild in original.children:
                child = JointFrameData()
                child.copyDataFromOtherInstance(originalChild,jointList,firstFrame, lastFrame)
                self.children.append(child)
        
    def copyFromController(self,originalController,animationId,parent,jointList,firstFrame,lastFrame):
        jointList[originalController.name] = self
        JointFrameData.__init__(self)#stupid error
        self.copyAttributesFromOtherInstance(originalController)
        self.parent = parent
        self.frames= []
        index = firstFrame#0
        #lastFrame = len(jointController.frameData[animationId].frames)
        #print self.name
        if self.type == TYPE_JOINT and animationId in originalController.frameData.keys():
            while index <= lastFrame:
                self.frames.append(copy.deepcopy(originalController.frameData[animationId].frames[index]))
                #print(index)
                index += 1
            self.children = []
            for childController in originalController.children:
                child = JointFrameData()
                child.copyFromController(childController,animationId,self,jointList,firstFrame,lastFrame)
                self.children.append(child)
  


    def addChild(self,child):
        self.children.append(child)    
        
    def printChildren(self):
        level_string = ""
       
        i = 0
        while i <self.level:
            level_string += " "
            i+= 1
            
        i = 0
        rotorderstr = ""
        while i < len(self.rotOrder):
            rotorderstr += self.rotOrder[i]
            i+=1
            
        print(level_string  +  str(self.level) + self.name+": "+rotorderstr)
        
        
        for child in self.children:
            child.printChildren()

    