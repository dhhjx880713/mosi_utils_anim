# -*- coding: utf-8 -*-
#===============================================================================
# authors: Erik Herrmann (DFKI GmbH, FB: Agenten und Simulierte Realität) general data structure, Han Du (Daimler AG/DFKI GmbH) interface for morphable model
# last update: 24.4.2014
#===============================================================================
import sys
import math
import collections
from math import *
from cgkit.cgtypes import quat,vec3
from JointData import *
from IO import AnimationReader
from IO import AnimationWriter
from copy import deepcopy
#from IO.Configuration import *

class SkeletonAnimationData (object):
    '''
    Stores an animation in tree of joints. Each joint stores its own frames.
    '''
    def __init__(self,properties = None):

        self.joints = collections.OrderedDict()#{}
        self.root =JointFrameData()
        self.currentLevel = 0
        self.numberOfFrames = 0
        self.frameTime = 0.0
        self.maxAnimationTime = 0.0
        self.scaleFactor = 1.0
        self.loadedCorrectly = False
        self.name = ""
        self.filePath = ""


    def buildFromBVHFile(self,filePath):
        AnimationReader.BVHSkeletonReader().loadFile(filePath,self)
        return


    def copyDataFromOtherInstance(self, originalAnimationData, firstFrame =-1, lastFrame =-1):
        '''
        copies a the properties and a range of the frames from another instance
        '''
        if firstFrame ==-1 or  lastFrame ==-1:
            firstFrame, lastFrame  = 0,originalAnimationData.numberOfFrames-1
            self.numberOfFrames = originalAnimationData.numberOfFrames
        else:
            self.numberOfFrames = lastFrame-firstFrame

        self.loadedCorrectly = originalAnimationData.loadedCorrectly
        self.name = originalAnimationData.name
        self.filePath = originalAnimationData.filePath
        self.scaleFactor = originalAnimationData.scaleFactor
        self.currentLevel =  originalAnimationData.currentLevel
        
        self.frameTime = originalAnimationData.frameTime
        self.maxAnimationTime = originalAnimationData.maxAnimationTime

        self.root = JointFrameData()
        self.root.copyDataFromOtherInstance(originalAnimationData.root,self.joints,firstFrame, lastFrame)

    def copyFromController(self, controller,animationId,firstFrame =-1, lastFrame =-1):
        '''
        copies a the properties and a range of the frames from an instance of the SkeletonAnimationController
        '''
        if firstFrame ==-1 or  lastFrame ==-1:
            self.numberOfFrames = len(controller.rootController.frameData[animationId].frames)
            firstFrame, lastFrame  = 0,self.numberOfFrames-1
#         else:
#             if firstFrame > 0 :#the first frame is the first index so the number of frames are
#                 self.numberOfFrames = lastFrame-(firstFrame-1)
#             else:
#                 firstFrame = 0
#                 self.numberOfFrames = lastFrame-firstFrame


        self.loadedCorrectly = controller.loadedCorrectly
        self.name = controller.name
        self.filePath = controller.filePath
        self.scaleFactor = controller.scaleFactor

        self.frameTime = controller.frameTime
        self.maxAnimationTime = self.frameTime*self.numberOfFrames

        self.root = JointFrameData()
        #copy the rest from the framedata because the children are still preserved TODO will be optimized
        #self.root.copyDataFromOtherInstance(controller.rootController.frameData[index],self.joints)
        self.root.copyFromController(controller.rootController,animationId,self.joints,firstFrame, lastFrame)
        self.numberOfFrames = len(self.root.frames)
        return



    def printHierarchy(self):
        self.root.printChildren()

    def printself(self):
        print("name: "+self.name)
        for joint in self.joints.itervalues():
            print joint.rotOrder
            #joint.printself()



    def cutBeforeFrame(self,frameNumber):
        newNumberOfFrames = self.root.cutBeforeFrame(frameNumber)
        if newNumberOfFrames>-1:
            self.numberOfFrames =newNumberOfFrames
            self.maxAnimationTime = self.frameTime*self.numberOfFrames
        return newNumberOfFrames

    def cutAfterFrame(self,frameNumber):
         newNumberOfFrames = self.root.cutAfterFrame(frameNumber)
         if newNumberOfFrames>-1:
            self.numberOfFrames =newNumberOfFrames
            self.maxAnimationTime = self.frameTime*self.numberOfFrames

         return newNumberOfFrames


    def saveToFile(self,filePath, usingQuaternion = True):
        '''
        saves the animation as a BVH file
        '''
        AnimationWriter.BVHSkeletonWriter().saveAnimationDataToFile(self, filePath,usingQuaternion)



    def saveFrameToFile(self, frameNumber):
        AnimationWriter.BVHSkeletonWriter().saveAnimationDataFrameToFile(self,frameNumber)

    def savePointCloud(self,filePath,frameNumber):
        try:
            f = open(filePath,"wb")
            numOfJoints = len(self.joints)
            formatString = "# .PCD v.7 - Point Cloud Data file format \n"+"VERSION .7\n"+ "FIELDS x y z \n"+"SIZE 4 4 4 \n"+"TYPE F F F  \n"+"COUNT 1 1 1  \n"+"WIDTH "+str(numOfJoints)+" \n"+"HEIGHT 1 \n"+"VIEWPOINT 0 0 0 1 0 0 0 \n"+"POINTS "+str(numOfJoints)+" \n"+"DATA ascii \n"
            f.write(formatString)
            for joint in self.joints.itervalues():
                position = joint.getAbsolutePosition(frameNumber)
                f.write(str(position.x)+" "+str(position.y)+" "+str(position.z)+"\n")
            f.close()
        except:
            etype, evalue, etb = sys.exc_info()
            evalue = etype("Failed to save file: %s" % evalue)
            print(evalue)
        return

    def getRelativeEndRootPosition(self):
        return self.getRelativeRootPositionAtFrame(self.numberOfFrames-1)

    def getRelativeRootPositionAtFrame(self,frameNumber):
        transformation = self.getRelativeRootTransformationAtFrame(frameNumber)
        #print absoluteJointTransformation
        #NOTE cgkit seems to handle the multiplication of a 4x4 matrix with a vec3 by automatically projecting the vec3 to a vec4 by adding w=1 and then changes it back to vec3.
        #the result in tests was at least the same
        if transformation != None:
            position = CustomMath.numpyArrayToCGkit4x4(transformation) * vec3(0.0,0.0,0.0)#extract the translation from the matrix
            return position

    def getRelativeEndRootTransformation(self):
        return self.getRelativeRootTransformationAtFrame(self.numberOfFrames-1)

    def getRelativeRootTransformationAtFrame(self,frameNumber):
            if self.loadedCorrectly:
                frameNumber = int(frameNumber)#TypeError: list indices must be integers, not float - weird symptom of larger bug
                return self.root.getRelativeTransformation(frameNumber)
            else:
                return None

    def getRelativeEndRootRotation(self):
        return self.getRelativeRootRotationAtFrame(self.numberOfFrames-1)

    def getRelativeRootRotationAtFrame(self,frameNumber):
         if self.loadedCorrectly:
            return self.root.frames[frameNumber].getEulerAngles(self.root.rotOrder)#self.root.frames[frameNumber].EulerAnglesDeg
         else:
            return None

    def getRelativeEndRootTranslation(self):
        frameNumber = int( self.numberOfFrames-1)
        return self.getRelativeRootTranslationAtFrame(frameNumber)


    def getRelativeRootTranslationAtFrame(self,frameNumber):
        if self.loadedCorrectly:
            return self.root.offset +self.root.frames[frameNumber].translation
        else:
            return None
#     #takes a cgtypes.mat4
#     def transformAnimationFrames(self,transformationMatrix):
#         self.root.transformFrames(transformationMatrix)
#         return

    def transformAnimationFrames(self,rotationMatrix,translation):
        print "translation",translation
        tempTranslationMatrix =CustomMath.getTranslationMatrix(translation)
        tempTransformation = np.dot(rotationMatrix,tempTranslationMatrix)#M = T * R
        tramformationMatrix =CustomMath.numpyArrayToCGkit4x4(tempTransformation)
        self.root.transformAnimationFrames(tramformationMatrix)
        return

    #takes cgkit.cgtypes.vec3
    def rotateAnimationFramesByEulerAngles(self,angles):
        q = CustomMath.getQuaternionFromEuler(angles, self.root.rotOrder)
        self.rotateAnimationFramesByQuaternion(q)
        return

    def rotateAnimationFramesByMatrix(self,rotationMatrix):
        q = quat()
        q.fromMat(rotationMatrix)
        self.root.rotateAnimationFramesByQuaternion(q)
        return

    def rotateAnimationFramesByQuaternion(self,q):
        self.root.rotateAnimationFramesByQuaternion(q)


    def translateAnimationFrames(self,translation):
        self.root.translateAnimationFrames(translation)


    def getOffsetVector(self):
        offsetVector =[]
        for frame in range(int(self.numberOfFrames)):
            for joint in self.joints.itervalues():
                 if joint.type != TYPE_END:
                    offsetVector.append(joint.frames[frame])
        return offsetVector

    def getOffsetVector2(self):
        offsetVector =[]
        for frame in range(int(self.numberOfFrames)):
            for joint in self.joints.itervalues():
                 if joint.type == TYPE_JOINT:
                    offsetVector.append(joint.frames[frame].getEulerAngles(joint.rotOrder))
                 elif joint.type == TYPE_ROOT:
                    offsetVector.append((joint.frames[frame].translation,joint.frames[frame].getEulerAngles(joint.rotOrder)))
        return offsetVector

    def getMaxLevel(self):
        maxLevel = 0
        for joint in self.joints.itervalues():
             if joint.level > maxLevel:
                 maxLevel = joint.level
        return maxLevel




    def transformToConstraintPoint(self, contactPoint):
        contactPointPosition = vec3(0)
        counter = 0
        for joint in self.joints.itervalues():
            if joint.name == contactPoint:
                contactPointPosition = joint.getAbsolutePosition(0)
                break
            counter += 1
        if counter == len(self.joints):
            raise('skeleton does not contain key joint')
        contactPointPosition.y = 0.0
        print "contactPoint: " + str(contactPointPosition)
        for index in range(int(self.numberOfFrames)):
            diff = contactPointPosition - joint.getAbsolutePosition(index)
            self.root.translateAnimationFrame( index, diff)


    def filterRotationOutliers(self,jointName,axis):
#         for frame in self.root.frames:
#             frame.translation = vec3(0.0)
        if jointName in self.joints.keys():
            joint = self.joints[jointName]
            axisIndex =joint.rotOrder.index(axis)
            for i in range(int(self.numberOfFrames)):
                eulerAngles = joint.frames[i].getEulerAngles(joint.rotOrder)
                eulerAngles[axisIndex] = abs(eulerAngles[axisIndex])
                joint.frames[i].setRotationByEulerAngles(eulerAngles,joint.rotOrder)




    def deleteJoint(self,jointName):
        '''
        removes a joint from the skeleton hierarchy and appends its children to its parent.
        Note end effectors or the root joint can not be removed using this method
        '''
        if jointName in self.joints.keys() and self.joints[jointName].type == TYPE_JOINT:
            parent = self.joints[jointName].parent
            children = self.joints[jointName].children
            for i in range(len(parent.children)):
                if parent.children[i].name == jointName:
                    del parent.children[i]
            parent.children += children
            for c in parent.children:
                c.parent = parent
            del self.joints[jointName]
            #todo use changeRelativeCoordinateSystem to let the endeffectors end up at the same position


    def getChainFromRootToJoint(self,jointName,onlyJoints):
        '''
        returns a list containing the Joint objects (JointFrameData/RootJointFrameData) starting from the root and ending at the joint
        '''
        jointList = []
        if jointName in self.joints.keys():
            #print jointName
            currentJoint = self.joints[jointName]
            #add joints traversing along the parent relationship until the root is found
            while  currentJoint.parent != None:
#                 i= 0
#                 while i < len(currentJoint.frames):
#                     currentJoint.frames[i].setRotationByEulerAngles(vec3(0),currentJoint.rotOrder)
#                     print "angles",currentJoint.frames[i].getEulerAngles(currentJoint.rotOrder)
#                     i+= 1
                if currentJoint.type == TYPE_JOINT or not onlyJoints:
                    jointList.append(currentJoint)
                currentJoint = currentJoint.parent
            #add also the root to the list
            jointList.append(self.root)

            jointList.reverse()
            #print "reversed"
            #print jointList
        return jointList

    def getChainOfJointOrderFromRootToEndEffector(self,jointName):
        '''
        retrieves the list of joints in a kinemantik chain to the endeffector and  returns a list containing the tuples with joint
        names and their order in the bvh file or the frame line vector  starting from the root and ending at the joint
        '''
        jointNameList = []
        #jointKeys=[]
        if jointName in self.joints.keys():
            #print jointName
            currentJoint = self.joints[jointName]

            #add joints traversing along the parent relationship until the root is found
            while  currentJoint.parent != None:
                if currentJoint.type == TYPE_JOINT:
                    #print currentJoint.name, currentJoint.type
                    jointNameList.append ((currentJoint.name,currentJoint.number,3,currentJoint.level,currentJoint.offset))#add name, bvh reading order, number of parameters
                elif  currentJoint.type == TYPE_END:
                    print "add ",jointName
                    jointNameList.append ((currentJoint.name,currentJoint.number,0,currentJoint.level,currentJoint.offset))#add name, bvh reading order, number of parameters
                currentJoint = currentJoint.parent
            #add also the root to the list
            jointNameList.append((self.root.name, self.root.number,6,self.root.level,self.root.offset))#add name, bvh reading order, number of parameters
            jointNameList.reverse()
            #print jointNameList
        return jointNameList


    def getFullJointNameList(self,onlyJoints = False):
        '''
        returns a list containing the tuples with joint names and their order in the bvh file
        '''
        jointNameList = []
        #jointKeys=[]
        for jointName in self.joints.keys():
            #print jointName
            joint = self.joints[jointName]
            if joint.type == TYPE_JOINT:
                jointNameList.append ((jointName,joint.number,3))#add name, bvh reading order, number of parameters
            elif joint.type == TYPE_ROOT:
                jointNameList.append((joint.name, joint.number,6))#add name, bvh reading order, number of parameters
            elif joint.type == TYPE_END and not onlyJoints:
                jointNameList.append((joint.name, joint.number,0))#add name, bvh reading order, number of parameters
            #print jointNameList
        return jointNameList

    def getAbsolutePositionOfJointIteratively(self,jointName,frameIndex):
        jointList= self.getChainFromRootToJoint(jointName,onlyJoints = False)
        i  = 0
        matrix = CustomMath.get4x4IdentityMatrix()
        while i < len(jointList)-1:
             local = jointList[i].getRelativeTransformation(frameIndex)
             matrix = np.dot(local,matrix)
             i+=1

        offset = np.array([jointList[i].offset.x,jointList[i].offset.y,jointList[i].offset.z,1.0])
        absolute_position = np.dot(np.transpose(matrix),offset)
        return absolute_position


    def prepareJointChainListMap(self):
        chainList = collections.OrderedDict()
        for jointName in self.joints.keys():
            chainList[jointName] = self.getChainOfJointOrderFromRootToEndEffector(jointName)
        return chainList


    def fillParameterList(self,jointName,frameIndex,onlyJoints = False):
        '''
        returns a tuple with six arrays, each containing a list of values for one of the rotational degrees of freedom or one value of the offset for a joint in the sequence from the root to the target joint
        '''
        if frameIndex < self.numberOfFrames:
            jointList= self.getChainFromRootToJoint(jointName,onlyJoints)

            ax = []
            ay= []
            az= []
            thx= []
            thy= []
            thz= []
            #print "length",len(jointList)
    #         i = 0
    #         while i < len(jointList):
    #             print jointList[i].name
    #             i+=1
            i = 0
            #print "create array"
            while i < len(jointList):
                #if i < len(jointList):
                    print jointList[i].name, frameIndex

                    if jointList[i].type == TYPE_ROOT:#root notes also have a translation
                        #absolutePosition = jointList[i].getAbsolutePosition(frameIndex)
                        ax.append(jointList[i].offset.x +jointList[i].frames[frameIndex].translation.x)
                        ay.append(jointList[i].offset.y +jointList[i].frames[frameIndex].translation.y)
                        az.append(jointList[i].offset.z +jointList[i].frames[frameIndex].translation.z)
                    else:
                        ax.append(jointList[i].offset.x)
                        ay.append(jointList[i].offset.y)
                        az.append(jointList[i].offset.z)
                    if jointList[i].type != TYPE_END:#end sites dont have angles
                        angles = jointList[i].frames[frameIndex].getEulerAngles(jointList[i].rotOrder)
                        print jointList[i].name, "angles",angles.x,angles.y,angles.z
                        thx.append(radians(angles.x))
                        thy.append(radians(angles.y))
                        thz.append(radians(angles.z))

                    i+=1

#             print "done"
#             print "parameters"
#             print "joint", jointName,",frame ",frameIndex
#             print "ax",ax
#             print "ay",ay
#             print "az",az
#
#             print "thx",thx
#             print "thy",thy
#             print "thz",thz
            return ax,ay,az,thx,thy,thz
        else:
            return None




    def switchAxisValues(self,axis1,axis2):
        for joint in self.joints.itervalues():
            joint.switchAxisValues(axis1, axis2)
        return


    def switchAxisValuesOfOffset(self,axis1,axis2):
        for joint in self.joints.itervalues():
            joint.switchAxisValuesOfOffset(axis1, axis2)
        return










#####################################################################








    def getFrameLineVector(self, frameNumber, weighted = 0, usingQuaternion = True):
        vector = np.array(0)
        maxLevel = self.getMaxLevel()
        for joint in self.joints.itervalues():
            jointWeight = (maxLevel + 1.0 - joint.level)/(maxLevel + 1.0)
            if joint.type == TYPE_ROOT:

                if weighted == 1:

                    vector = np.array(joint.frames[frameNumber].translation)
                    if usingQuaternion:

                        vector = np.concatenate((vector,joint.frames[frameNumber].getEulerAnglesOrdered(joint.rotOrder)  * jointWeight))

                    else:
                        vector = np.concatenate((vector, joint.frames[frameNumber].EulerAnglesDeg * jointWeight))
                else:
                    vector = np.array(joint.frames[frameNumber].translation)
                    if usingQuaternion:
                        vector = np.concatenate((vector, joint.frames[frameNumber].getEulerAnglesOrdered(joint.rotOrder) ))
                    else:
                        vector = np.concatenate((vector, joint.frames[frameNumber].EulerAnglesDeg))
#                     angles = joint.frames[frameNumber].getEulerAnglesOrdered(joint.rotOrder)
#                     vector = np.concatenate((vector, [angles.y,]))
            elif joint.type == TYPE_JOINT:


                if weighted == 1:
                    if usingQuaternion:
                        vector = np.concatenate((vector, joint.frames[frameNumber].getEulerAnglesOrdered(joint.rotOrder)  * jointWeight))
                    else:
                        vector = np.concatenate((vector, joint.frames[frameNumber].EulerAnglesDeg * jointWeight))
                else:
                    if usingQuaternion:
                        vector = np.concatenate((vector, joint.frames[frameNumber].getEulerAnglesOrdered(joint.rotOrder) ))
                    else:
                        vector = np.concatenate((vector, joint.frames[frameNumber].EulerAnglesDeg))
        return vector




    def getFrameLineVectorLength(self):
        length = 0
        #print len(self.joints)
        for joint in self.joints.itervalues():
            length+= joint.numberOfChannels
        return length


    def setNumberOfFrames(self, newNumberOfFrames):
        '''
        /brief  sets the number of frames for this AnimationData
        '''
        if self.numberOfFrames > newNumberOfFrames:
            for joint in self.joints.itervalues():
                if joint.type == TYPE_ROOT or joint.type == TYPE_JOINT:
                    joint.frames = joint.frames[:newNumberOfFrames]


        elif self.numberOfFrames < newNumberOfFrames:
            for joint in self.joints.itervalues():
                if joint.type == TYPE_ROOT or joint.type == TYPE_JOINT:
                    lastFrame = joint.frames[-1]
                    diff = newNumberOfFrames - int(self.numberOfFrames)
                    joint.frames.extend([deepcopy(lastFrame)
                                         for i in xrange(diff)])

        self.numberOfFrames = float(newNumberOfFrames)
        return




    def getFramesData(self, weighted = 0, usingQuaternion = True):
        '''
        /brief store motion data of BVH file as a matrix
        '''
        framesData = []
        for i in range(int(self.numberOfFrames)):
            framesData.append(self.getFrameLineVector(i, weighted, usingQuaternion))
        return np.array(framesData)
        # self.framesData = np.array(framesData)

    def getFramesDataInVector(self, weighted = 0, usingQuaternion = True):
        # concantenating all the motion data of each frame as a long vector
        motionData = self.getFramesData(weighted, usingQuaternion)
        motionVector = np.ravel(motionData)
        self.motionVector = motionVector
        return motionVector

    def fromVectorToMotionData(self, vector, weighted = 0, usingQuaternion = True):
        '''
        /brief  convert a high dimensional vector to BVH motion data. calls fromVectorToFrameData
        '''
        lengthOfLine = self.getFrameLineVectorLength()
        vectorLength =(lengthOfLine * self.numberOfFrames)
        if len(vector) != lengthOfLine * self.numberOfFrames:
            # If the length is not the same, adjust framelen of this animationdata
            self.setNumberOfFrames(len(vector) / lengthOfLine)
        #print "length of vector is: " + str(len(vector))
        print "save animation to file",len(vector),self.numberOfFrames,lengthOfLine
        frameData = vector.reshape( int(self.numberOfFrames), lengthOfLine)
        #print frameData.shape

        for i in range(int(self.numberOfFrames)):
            #print "length of frame data: " + str(len(frameData[i,:]))
            self.fromVectorToFrameData(frameData[i,:], i, weighted, usingQuaternion)

    def fromVectorToFrameData(self, vector, frameIndex, weighted = 0, usingQuaternion = True):
        '''
        /brief convert a data vector to one frame in BVH file
        '''

        lengthOfLine = self.getFrameLineVectorLength()
#         print "testing !"
#         print lengthOfLine

        if len(vector) != lengthOfLine:
            raise ValueError(" The length of data vector is wrong! ")
        maxLevel = self.getMaxLevel()
        index = 0

        for joint in self.joints.itervalues():
            if joint.numberOfChannels ==  6:
                translation = vec3(vector[index:index+3])
                joint.frames[frameIndex].translation = translation
                # update index
                index = index + 3
                if weighted == 1:

                    jointWeight = (maxLevel + 1.0 - joint.level)/(maxLevel + 1.0)
                    rotation = vec3(vector[index:index+3])/jointWeight
                    #joint.frames[frameIndex].EulerAnglesDeg = rotation
                    #print "input", rotation
                    if usingQuaternion:
                        joint.frames[frameIndex].setRotationByEulerAngles(rotation,joint.rotOrder)
                    else:
                        joint.frames[frameIndex].EulerAnglesDeg = rotation
                        joint.frames[frameIndex].setRotationByEulerAngles(rotation,joint.rotOrder)
                    #print "output", joint.frames[frameIndex].getEulerAngles(joint.rotOrder)
                    # update index
                    index = index + 3
                else:
                    rotation =  vec3(vector[index:index+3])
#                     yAngle = vector[index]

                    #joint.frames[frameIndex].EulerAnglesDeg = rotation
#                     originalData = joint.frames[frameIndex].getEulerAnglesOrdered(joint.rotOrder)
#                     rotation.x = originalData.x
#                     rotation.z = originalData.z
#                     rotation = vec3(originalData.x, yAngle, originalData.z)
                    if usingQuaternion:
                        joint.frames[frameIndex].setRotationByEulerAngles(rotation,joint.rotOrder)
                    else:
                        joint.frames[frameIndex].EulerAnglesDeg = rotation
                        joint.frames[frameIndex].setRotationByEulerAngles(rotation,joint.rotOrder)

                    # update index
                    index = index + 3
#                     index = index + 1

            elif joint.numberOfChannels == 3:
                if weighted == 1:
                    jointWeight = (maxLevel + 1.0 - joint.level)/(maxLevel + 1.0)
                    rotation = vec3(vector[index:index+3])/jointWeight
                    #print "input", rotation
                    #joint.frames[frameIndex].EulerAnglesDeg = rotation
                    #print rotation
                    if usingQuaternion:
                        joint.frames[frameIndex].setRotationByEulerAngles(rotation,joint.rotOrder)
                    else:
                        joint.frames[frameIndex].EulerAnglesDeg = rotation
                        joint.frames[frameIndex].setRotationByEulerAngles(rotation,joint.rotOrder)
                    #print "output", joint.frames[frameIndex].getEulerAngles(joint.rotOrder)
                    # update index
                    index = index + 3
                else:
                    rotation =  vec3(vector[index:index+3])
                    #joint.frames[frameIndex].EulerAnglesDeg = rotation
                    #print rotation
                    if usingQuaternion:
                        joint.frames[frameIndex].setRotationByEulerAngles(rotation,joint.rotOrder)
                    else:
                        joint.frames[frameIndex].EulerAnglesDeg = rotation
                        joint.frames[frameIndex].setRotationByEulerAngles(rotation,joint.rotOrder)
                    # update index
                    index = index + 3


############################################################################################################################################################
#deprecated


    def getFrameLineVectorOld(self, frameNumber, weighted = 1):
        vector = np.array(0)
        maxLevel = self.getMaxLevel()
        for joint in self.joints.itervalues():
            jointWeight = (maxLevel + 1.0 - joint.level)/(maxLevel + 1.0)
            if joint.type == TYPE_ROOT:

                if weighted == 1:

                    vector = np.array(joint.frames[frameNumber].translation)
                    vector = np.concatenate((vector,joint.frames[frameNumber].getEulerAnglesOrdered(joint.rotOrder)  * jointWeight))
                else:
                    vector = np.array(joint.frames[frameNumber].translation)
                    vector = np.concatenate((vector, joint.frames[frameNumber].getEulerAnglesOrdered(joint.rotOrder) ))
            elif joint.type == TYPE_JOINT:


                if weighted == 1:
                    vector = np.concatenate((vector, joint.frames[frameNumber].getEulerAnglesOrdered(joint.rotOrder)  * jointWeight))
                else:

                    vector = np.concatenate((vector, joint.frames[frameNumber].getEulerAnglesOrdered(joint.rotOrder) ))
        return vector
    def getFramesDataOld(self, weighted = 1):
        '''
        /brief store motion data of BVH file as a matrix
        '''
        framesData = []
        for i in range(int(self.numberOfFrames)):
            framesData.append(self.getFrameLineVector(i, weighted))
        return np.array(framesData)
        # self.framesData = np.array(framesData)

    def getFramesDataInVectorOld(self, weighted = 1):
        # concantenating all the motion data of each frame as a long vector
        motionData = self.getFramesData(weighted)
        motionVector = np.ravel(motionData)
        self.motionVector = motionVector
        return motionVector

    def fromVectorToMotionDataOld(self, vector, weighted = 1):
        '''
        /brief  convert a high dimensional vector to BVH motion data. calls fromVectorToFrameData
        '''
        lengthOfLine = self.getFrameLineVectorLength()
        #print lengthOfLine
#        vectorLength =(lengthOfLine * self.numberOfFrames)
#        if len(vector) != lengthOfLine * self.numberOfFrames:
#            raise ValueError(" The length of data vector is wrong! "+str(len(vector))+" - "+str(vectorLength))

        frames = len(vector) / lengthOfLine
        self.setNumberOfFrames(frames)

        #print "length of vector is: " + str(len(vector))
        frameData = vector.reshape( int(self.numberOfFrames), lengthOfLine)
        #print frameData.shape

        for i in range(int(self.numberOfFrames)):
            #print "length of frame data: " + str(len(frameData[i,:]))
            self.fromVectorToFrameData(frameData[i,:], i, weighted)

    def fromVectorToFrameDataOld(self, vector, frameIndex, weighted = 1):
        '''
        /brief convert a data vector to one frame in BVH file
        '''

        lengthOfLine = self.getFrameLineVectorLength()
#         print "testing !"
#         print lengthOfLine

        if len(vector) != lengthOfLine:
            raise ValueError(" The length of data vector is wrong! ")
        maxLevel = self.getMaxLevel()
        index = 0

        for joint in self.joints.itervalues():
            if joint.type ==  TYPE_ROOT:
                translation = vec3(vector[index:index+3])
                joint.frames[frameIndex].translation = translation
                # update index
                index = index + 3
                if weighted == 1:

                    jointWeight = (maxLevel + 1.0 - joint.level)/(maxLevel + 1.0)
                    rotation = vec3(vector[index:index+3])/jointWeight
                    #joint.frames[frameIndex].EulerAnglesDeg = rotation
                    #print "input", rotation
                    joint.frames[frameIndex].setRotationByEulerAngles(rotation,joint.rotOrder)
                    #print "output", joint.frames[frameIndex].getEulerAngles(joint.rotOrder)
                    # update index
                    index = index + 3
                else:
                    rotation =  vec3(vector[index:index+3])

                    #joint.frames[frameIndex].EulerAnglesDeg = rotation
                    joint.frames[frameIndex].setRotationByEulerAngles(rotation,joint.rotOrder)

                    # update index
                    index = index + 3

            elif joint.type == TYPE_JOINT:
                if weighted == 1:
                    jointWeight = (maxLevel + 1.0 - joint.level)/(maxLevel + 1.0)
                    rotation = vec3(vector[index:index+3])/jointWeight
                    #print "input", rotation
                    #joint.frames[frameIndex].EulerAnglesDeg = rotation
                    #print rotation
                    joint.frames[frameIndex].setRotationByEulerAngles(rotation,joint.rotOrder)
                    #print "output", joint.frames[frameIndex].getEulerAngles(joint.rotOrder)
                    # update index
                    index = index + 3
                else:
                    rotation =  vec3(vector[index:index+3])
                    #joint.frames[frameIndex].EulerAnglesDeg = rotation
                    #print rotation
                    joint.frames[frameIndex].setRotationByEulerAngles(rotation,joint.rotOrder)
                    # update index
                    index = index + 3