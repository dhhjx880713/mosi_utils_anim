# -*- coding: utf-8 -*-
#===============================================================================
# author: Erik Herrmann (DFKI GmbH, FB: Agenten und Simulierte Realit�t)
# last update: 2.4.2014
#===============================================================================
from AnimationEngine import JointData
import sys

class BVHSkeletonWriter():

    def saveAnimationDataToFile(self,animationData,filePath, usingQuaternion = True):
        try:
            f = open(filePath,"wb")
            bvhString = self.getBVHString(animationData, usingQuaternion = usingQuaternion)
            #print bvhString
            f.write(bvhString)
            f.close()
        except:
            etype, evalue, etb = sys.exc_info()
            evalue = etype("Failed to save file: %s" % evalue)
            print(evalue)

    def saveAnimationDataFrameToFile(self, animationData,frameNumber):
        try:
            tempFilename = animationData.filePath[:-4]
            tempFilename = tempFilename + "_" + str(frameNumber) + ".bvhframe"
            print("saving frame "+str(frameNumber)+ " to "+tempFilename)
            f = open(tempFilename,"wb")
            #todo save bvh skeleton structure
            f.write(self.getSkeletonStructureString(animationData)+"\n")
            #save frame parameters
            f.write(self.getFrameLine(animationData,frameNumber))
            f.close()

        except:
            etype, evalue, etb = sys.exc_info()
            evalue = etype("Failed to save file: %s" % evalue)
            print(evalue)


    def saveAnimationFromControllerToFile(self,filePath,controller,animationId, usingQuaternion = True):
        try:
            f = open(filePath,"wb")
            #bvhString = self.getBVHString(animationData)
            bvhString =  self.getSkeletonStructureStringFromController(controller)+"\n"

            bvhString += "MOTION\n"#needed so the hierarchy reader still works
            bvhString += self.getFramesStringFromController(controller, animationId, usingQuaternion)

            f.write(bvhString)
            f.close()
        except:
            etype, evalue, etb = sys.exc_info()
            evalue = etype("Failed to save file: %s" % evalue)
            print(evalue)

    def saveAnimationFrameFromControllerToFile(self, controller,animationId,frameNumber):
            try:
                tempFilename = controller.filePath[:-4]
                tempFilename = tempFilename + "_" + str(frameNumber) + ".bvhframe"
                print("saving frame "+str(frameNumber)+ " to "+tempFilename)
                f = open(tempFilename,"wb")
                #save bvh skeleton structure
                f.write(self.getSkeletonStructureStringFromController(controller)+"\n")
                #save frame parameters
                f.write(self.getFrameLineFromController(controller,animationId,frameNumber))
                f.close()

            except:
                etype, evalue, etb = sys.exc_info()
                evalue = etype("Failed to save file: %s" % evalue)
                print(evalue)



    def getBVHString(self,animationData, usingQuaternion = True):
       #print "test"
        bvhString =  self.getSkeletonStructureString(animationData)+"\n"
       # print "test2"
        #print bvhString
        bvhString += "MOTION\n"#needed so the hierarchy reader still works
        print animationData.numberOfFrames
        bvhString += self.getFramesString(animationData, usingQuaternion = usingQuaternion)
        return bvhString

    def getSkeletonStructureString(self,animationData):
        structureDesc = "HIERARCHY"+"\n"
        structureDesc += self.getRootStructureString(animationData.root)
        return structureDesc

    def getSkeletonStructureStringFromController(self,controller):
        structureDesc = "HIERARCHY"+"\n"
        structureDesc += self.getRootStructureString(controller.rootController)
        return structureDesc

    def getJointStructureString(self,joint):
        structureDesc = ""
        tabString=""
        tempLevel = 0
        while tempLevel < joint.level:
                tabString +="\t"
                tempLevel+=1
        if joint.type == JointData.TYPE_JOINT:
            structureDesc +=tabString+ "JOINT "+joint.name+"\n"
        else:
            structureDesc += tabString+"End Site"+"\n"
        structureDesc += tabString+"{"+"\n"
        structureDesc += tabString + "\t"  + "OFFSET " +"\t "+ str(joint.offset.x) +"\t "+str(joint.offset.y) +"\t "+str(joint.offset.z)+"\n"

        if joint.type == JointData.TYPE_JOINT:
            structureDesc += tabString +"\t"  + "CHANNELS "+str(joint.numberOfChannels)+" "
            if joint.numberOfChannels ==6:
                for axis in joint.posOrder:
                    if axis =='X':
                        structureDesc+="Xposition "
                    elif axis =='Y':
                        structureDesc+="Yposition "
                    elif axis =='Z':
                        structureDesc+="Zposition "
            if joint.numberOfChannels >=3:
                for axis in joint.rotOrder:
                    if axis =='X':
                        structureDesc+="Xrotation "
                    elif axis =='Y':
                        structureDesc+="Yrotation "
                    elif axis =='Z':
                        structureDesc+="Zrotation "
            structureDesc+="\n"
            for child in joint.children:
                structureDesc+= self.getJointStructureString(child)

        structureDesc += tabString+"}"+"\n"
        return structureDesc


    def getRootStructureString(self,root):
        structureDesc = ""
        structureDesc += "ROOT "+root.name+"\n"
        structureDesc +="{"+"\n"
        structureDesc += "\t"  + "OFFSET " +"\t "+ str(root.offset.x) +"\t "+str(root.offset.y) +"\t "+str(root.offset.z) +"\n"
        structureDesc += "\t"  + "CHANNELS "+str(root.numberOfChannels)+" "
        if root.numberOfChannels ==6:
            for axis in root.posOrder:
                if axis =='X':
                    structureDesc+="Xposition "
                elif axis =='Y':
                    structureDesc+="Yposition "
                elif axis =='Z':
                    structureDesc+="Zposition "
        if root.numberOfChannels >=3:
            for axis in root.rotOrder:
                if axis =='X':
                    structureDesc+="Xrotation "
                elif axis =='Y':
                    structureDesc+="Yrotation "
                elif axis =='Z':
                    structureDesc+="Zrotation "
        structureDesc+="\n"

        for child in root.children:
            structureDesc+= self.getJointStructureString(child)
        structureDesc += "}"
        return structureDesc

    def getFrameLine(self,animationData,frameNumber, usingQuaternion = True):
        line = ""

        #note animationData joints has to be a ordered dictionary
        for joint in animationData.joints.itervalues():
             #root parameters
            if joint.numberOfChannels == 6:
                #print(animationData.name)
                line += self.getFrameTransformationString(joint.frames[frameNumber],joint.posOrder,joint.rotOrder, usingQuaternion = usingQuaternion)
             # joint parameters
            elif joint.numberOfChannels == 3:
                line += self.getFrameRotationString(joint.frames[frameNumber],joint.rotOrder, usingQuaternion = usingQuaternion)
        return line



    def getFramesString(self,data, usingQuaternion = True):
        framesString = "Frames: "+str(data.numberOfFrames)+"\n"
        framesString +="Frame Time: "+str(data.frameTime)+"\n"
            #save frame parameters to file
        frameNumber =0
        while frameNumber < data.numberOfFrames:
              framesString +=self.getFrameLine(data,frameNumber, usingQuaternion = usingQuaternion)+"\n"
              frameNumber+=1
        return framesString


    def getFrameRotationString(self,frame,rotOrder, usingQuaternion = True) :
        if usingQuaternion:
            EulerAnglesDeg = frame.getEulerAnglesOrdered(rotOrder)
        else:
            EulerAnglesDeg = frame.EulerAnglesDeg
        line = ""
        line+=str(EulerAnglesDeg.x)+"\t"
        line+=str(EulerAnglesDeg.y)+"\t"
        line+=str(EulerAnglesDeg.z)+"\t"
#         EulerAnglesDeg = CustomMath.quaternionToEuler3(frame.rotationQuaternion,rotOrder)
#         line = ""
#         for axis in rotOrder:
#             if axis =='X':
#                 line+=str(EulerAnglesDeg.x)
#             elif axis == 'Y':
#                 line+=str(EulerAnglesDeg.y)
#             elif axis == 'Z':
#                 line+=str(EulerAnglesDeg.z)
#             line+="\t"
        return line


    def getFrameTransformationString(self,frame,posOrder,rotOrder, usingQuaternion = True):

        line = ""
        for axis in posOrder:
            if axis =='X':
                line+=str(frame.translation.x)
            elif axis == 'Y':
                line+=str(frame.translation.y)
            elif axis == 'Z':
                line+=str(frame.translation.z)
            line+="\t"

        #EulerAnglesDeg = CustomMath.quaternionToEuler3(frame.rotationQuaternion,rotOrder)
        if usingQuaternion:
            EulerAnglesDeg = frame.getEulerAnglesOrdered(rotOrder)
        else:
            EulerAnglesDeg = frame.EulerAnglesDeg       
        line+=str(EulerAnglesDeg.x)+"\t"
        line+=str(EulerAnglesDeg.y)+"\t"
        line+=str(EulerAnglesDeg.z)+"\t"

#         for axis in rotOrder:
#             if axis =='X':
#                 line+=str(EulerAnglesDeg.x)
#             elif axis == 'Y':
#                 line+=str(EulerAnglesDeg.y)
#             elif axis == 'Z':
#                 line+=str(EulerAnglesDeg.z)
#             line+="\t"

        return line

    def getFrameLineFromController(self,controller,animationId,frameNumber, usingQuaternion = True):
        if animationId in controller.rootController.frameData.keys():
            line = ""

            for joint in controller.jointControllers.itervalues():
                 #root parameters
                if joint.numberOfChannels == 6:
                    line += self.getFrameTransformationString(joint.frameData[animationId].frames[frameNumber], joint.posOrder,joint.rotOrder)
                 # joint parameters
                elif joint.numberOfChannels == 3:
                     line += self.getFrameRotationString(joint.frameData[animationId].frames[frameNumber], joint.rotOrder, usingQuaternion)
            return line



    #return self.rootController.saveFrameToFile(self.currentAnimationId,self.currentFrameNumber)
    def getFramesStringFromController(self,controller,animationId, usingQuaternion = True):
        if animationId in controller.rootController.frameData.keys():
            numberOfFrames = len(controller.rootController.frameData[animationId].frames)
            framesString = "Frames: "+str(numberOfFrames)+"\n"
            framesString +="Frame Time: "+str(controller.frameTime)+"\n"
            #save frame parameters to file
            frameNumber =0
            while frameNumber < numberOfFrames:
                  framesString +=self.getFrameLineFromController(controller,animationId,frameNumber, usingQuaternion)+"\n"
                  frameNumber+=1

            return framesString


class AnimationStateMachineWriter():

    def saveStateMachineToFile(self,stateMachine,filePath):
         print "save to "+filePath
         try:
             fo = open(filePath,'wb')
             if stateMachine.loadedCorrectly:

                 #use skeletonstructure string from root and ignore the offset
                 bvhWriter = BVHSkeletonWriter()
                 bvhLine =bvhWriter.getSkeletonStructureStringFromController(stateMachine) #self.getSkeletonStructureString()#[12:]
                 #print(bvhLine)
                 fo.write(bvhLine + "\n")
                 fo.write("PROPERTIES \n")
                 fo.write("Frame Time: "+str(stateMachine.frameTime)+"\n")
                 fo.write("\n")
                 fo.write("STATES \n")
                 fo.write("Number: "+str(len(stateMachine.vertList))+"\n")
                 for state in stateMachine.vertList:
                     print state
                     fo.write("Name: "+str(state))
                     fo.write(" IdleState: ")
                     if stateMachine.defaultState == state:
                        fo.write(str(True)+"\n")
                     else:
                         fo.write(str(False)+"\n")

                     index = stateMachine.vertList[state].getAnimationId()
                     frameString =bvhWriter.getFramesStringFromController(stateMachine,index) #stateMachine.getFramesString(index)
                     fo.write(frameString)
                     fo.write("\n")

                     fo.write("Transitions: ")

                     for transition in stateMachine.vertList[state].getConnections():
                          #print "window size"+str( stateMachine.vertList[state].connectedTo[transition]['windowSize'])
                          windowSize = stateMachine.vertList[state].getWindowSizeOfTransition(transition)#connectedTo[transition]['windowSize']
                          fo.write(transition+":"+ str(windowSize) + "\t")
                          #print "this"

                     fo.write("\n\n\n")

             fo.close()
         except:
             return

    def saveStateAnimationToFile(self,stateMachine,stateKey,filePath):
        try:
             fo = open(filePath,'wb')
             if stateMachine.loadedCorrectly:
                 bvhWriter = BVHSkeletonWriter()
                 fo.write(bvhWriter.getSkeletonStructureStringFromController(stateMachine)+"\n")
                 fo.write("MOTION\n")#needed so the hierarchy reader still works
                 animationId= stateMachine.vertList[stateKey].getAnimationId()
                 fo.write(bvhWriter.getFramesStringFromController(stateMachine, animationId))
             fo.close()
        except:
             return
        return


