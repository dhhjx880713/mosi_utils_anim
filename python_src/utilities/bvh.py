#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

BVH
===

Biovision file format classes for reading and writing.
BVH Reader by Martin Manns
BVH Writer by Erik Herrmann

"""

from collections import OrderedDict
import collections
import numpy as np
from math import degrees
from cgkit.cgtypes import quat #TODO replace with transformations.py

def create_filtered_node_name_map(bvh_reader):
    """
    creates dictionary that maps node names to indices in a frame vector
    without "Bip" joints
    """
    node_name_map = collections.OrderedDict()
    j = 0
    for node_name in bvh_reader.node_names:
        if not node_name.startswith("Bip") and \
            "children" in bvh_reader.node_names[node_name].keys():
            node_name_map[node_name] = j
            j += 1

    return node_name_map

def get_joint_weights(bvh_reader, node_name_map=None):
    """ Gives joints weights according to their distance in the joint hiearchty
       to the root joint. The further away the smaller the weight.
    """
#    max_level = bvh_reader.max_level+1.0
    if node_name_map:

#        weights = [np.exp(max_level - bvh_reader.node_levels[node_name]/max_level) for node_name in node_name_map.keys()]
        weights = [np.exp(-bvh_reader.node_names[node_name]["level"]) for node_name in node_name_map.keys()]
    else:
#        weights = [np.exp(max_level - bvh_reader.node_levels[node_name]/max_level) for node_name in bvh_reader.node_names.keys()]
        weights = [np.exp(-bvh_reader.node_levels[node_name]["level"]) for node_name in bvh_reader.node_names.keys()]
    return weights
    
class BVHReader(object):
    """Biovision file format class

    Parameters
    ----------
     * infile: string
    \t path to BVH file that is loaded initially

    """

    def __init__(self, infilename=""):
       

       
        self.node_names = OrderedDict()
        self.node_channels = []
        self.parent_dict = {}
        self.frame_time = None
        self.frames = None
        self.root = ""  # needed for the bvh writer
        if infilename != "":
            infile = open(infilename,"rb")
            self.read(infile)
            self.max_level = max([node["level"] for node in
                                  self.node_names.values()
                                  if "level" in node.keys()])
            self.parent_dict = self._get_parent_dict()
        infile.close()

    def _read_skeleton(self, infile):
        """Reads the skeleton part of a BVH file"""

        parents = []
        level = 0
        name = None

        for line in infile:
            if "{" in line:
                parents.append(name)
                level += 1

            if "}" in line:
                level -= 1
                parents.pop(-1)
                if level == 0:
                    break

            line_split = line.strip().split()

            if line_split:
                if line_split[0] == "ROOT":
                    name = line_split[1]
                    self.root = name
                    self.node_names[name] = {"children": [], "level": level, "channels":[]}

                elif line_split[0] == "JOINT":
                    name = line_split[1]
                    self.node_names[name] = {"children": [], "level": level, "channels":[]}
                    self.node_names[parents[-1]]["children"].append(name)

                elif line_split[0] == "CHANNELS":
                    for channel in line_split[2:]:
                        self.node_channels.append((name, channel))
                        self.node_names[name]["channels"].append(channel)

                elif line_split == ["End", "Site"]:
                    name += "_" + "".join(line_split)
                    self.node_names[name] = {"level": level}
                    #also the end sites need to be adde as children
                    self.node_names[parents[-1]]["children"].append(name)

                elif line_split[0] == "OFFSET" and name in self.node_names.keys():
                    offset = [float(x) for x in line_split[1:]]
                    self.node_names[name]["offset"] = offset
                    


    def _read_frametime(self, infile):
        """Reads the frametime part of a BVH file"""

        for line in infile:
            if line.startswith("Frame Time:"):
                self.frame_time = float(line.split(":")[-1].strip())
                break

    def _read_frames(self, infile):
        """Reads the frames part of a BVH file"""

        frames = []
        for line in infile:

            line_split = line.strip().split()
            frames.append(map(float, line_split))

        self.frames = np.array(frames)

    def read(self, infile):
        """Reads BVH file infile

        Parameters
        ----------
         * infile: Filelike object, optional
        \tBVH file

        """

        for line in infile:
            if line.startswith("HIERARCHY"):
                break

        self._read_skeleton(infile)

        for line in infile:
            if line.startswith("MOTION"):
                break

        self._read_frametime(infile)
        self._read_frames(infile)
        
    def _get_parent_dict(self):
        """Returns a dict of node names to their parent node's name"""

        parent_dict = {}

        for node_name in self.node_names:
            if "children" in self.node_names[node_name].keys():
                for child_node in self.node_names[node_name]["children"]:
                    parent_dict[child_node] = node_name

        return parent_dict
        
    def get_angles(self, *node_channels):
        """Returns numpy array of angles in all frames for specified channels

        Parameters
        ----------
         * node_channels: 2-tuples of strings
        \tEach tuple contains joint name and channel name
        \te.g. ("hip", "Xposition")

        """

        indices = [self.node_channels.index(nc) for nc in node_channels]
        return self.frames[:, indices]
        
    def gen_all_parents(self, node_name):
        """Generator of all parents' node names of node with node_name"""

        while node_name in self.parent_dict:
            parent_name = self.parent_dict[node_name]
            yield parent_name
            node_name = parent_name


class BVHWriter(object):
    """Write BVH files.     
    """

    def __init__(self,filename, bvh_reader, frame_data,frame_time,is_quaternion=False):
        """ Saves an input motion defined either as an array of euler or quaternion 
            frame vectors as a BVH file. 
            * filename: name of the created bvh file.If no file is provided the 
                        string can still be returned for unit tests
            * bvh_reader: bvh data struced needed to copy the hierarchy
            * frame_data: array of motion vectors, either as euler or quaternion
            * frame_time: time in seconds for the display of each keyframe
            * is_quaternion: defines wether the frame_data is quaternion data 
                            or euler data
        """
        self.bvh_reader = bvh_reader
        self.frame_data = frame_data
        self.frame_time = frame_time
        self.is_quaternion = is_quaternion
        if filename != None:
            self._write(filename,self.generate_bvh_string())
            
       
       
    def _write(self, filename,bvh_string):
        """ Write the hierarchy string and the frame parameter string to file
        """
        if filename[-4:] == '.bvh':
            filename = filename
        else:
            filename = filename + '.bvh'
        fp = open(filename, 'wb')
        fp.write(bvh_string)
        fp.close()
    
  
    def generate_bvh_string(self):

          bvh_string = self._generate_hierarchy_string(self.bvh_reader.root,self.bvh_reader.node_names)+"\n"
          bvh_string += self._generate_frame_parameter_string(self.frame_data,self.bvh_reader.node_names,self.frame_time, self.is_quaternion)
          return bvh_string
          
    def _generate_hierarchy_string(self,root,node_names):
        """ Initiates the recursive generation of the skeleton structure string
            by calling _generate_joint_string with the root joint
        """
        hierarchy_string = "HIERARCHY\n"
        hierarchy_string += self._generate_joint_string(root,node_names,0)
        return hierarchy_string
     
          
    def _generate_joint_string(self,joint,node_names, joint_level):
        """ Recursive traversing of the joint hierarchy to create a 
            skeleton structure string in the BVH format
        """
        joint_string = ""
        temp_level = 0 
        tab_string=""
        while temp_level < joint_level:
            tab_string +="\t"
            temp_level+=1
             
        #determine joint type
        if joint_level == 0:
            joint_string +=tab_string+ "ROOT "+joint+"\n"
        else:
            if joint != "End Site" and "channels" in node_names[joint].keys():
                joint_string +=tab_string+ "JOINT "+joint +"\n"
            else:
                joint_string += tab_string+"End Site"+"\n"
              
        #open bracket add offset  
        joint_string += tab_string+"{"+"\n"
        joint_string += tab_string + "\t"  + "OFFSET " +"\t "+ \
                str(node_names[joint]["offset"][0]) +"\t "+str(node_names[joint]["offset"][1]) \
                +"\t "+str(node_names[joint]["offset"][2])+"\n"
                
        if joint != "End Site" and "channels" in node_names[joint].keys():
            #channel information
            channels = node_names[joint]["channels"]
            joint_string += tab_string +"\t"  + "CHANNELS "+str(len(channels))+" "
            for tok in channels:
                joint_string += tok+" "
            joint_string+="\n"
            
            joint_level +=1
            # recursive call for all children
            for child in node_names[joint]["children"]:
                joint_string += self._generate_joint_string(child, node_names,
                                                            joint_level)
        
        #close the bracket
        joint_string += tab_string+"}"+"\n"  
        return joint_string
        
        
    def _generate_frame_parameter_string(self,frame_data,node_names,frame_time ,is_quaternion = False):
        """ Converts the joint parameters for a list of frames into the BVH file representation. 
            Note: for the toe joints of the rocketbox skeleton a hard set value is used
            * frame_data: array of motion vectors, either as euler or quaternion
            * node_names: OrderedDict containing the nodes of the skeleton accessible by their name
            * frame_time: time in seconds for the display of each keyframe
            * is_quaternion: defines wether the frame_data is quaternion data 
                            or euler data
        """

        # convert to euler frames if necessary
        if not is_quaternion:
            skip_joints = True
            if len(frame_data[0]) == len([n for n in node_names if "children" in node_names[n].keys() ])*3 +3:
                skip_joints = False
            if not skip_joints:
                euler_frames = frame_data
            else:
                euler_frames = []
                for frame in frame_data:
                      euler_frame = frame[:3]
                      joint_idx = 0
                      for node_name in node_names:  # go through the node names to
                                                  # to append specific data
                        if "children" in node_names[node_name].keys(): #ignore end sites completely
                            if not node_name.startswith("Bip") or not skip_joints:
                                if node_name in ["Bip01_R_Toe0","Bip01_L_Toe0"]:
                                    #special fix for unused toe parameters
                                    euler_frame = np.concatenate((euler_frame,([90.0,-1.00000000713e-06,75.0	])), axis = 0)
                                else:
                                    #print node_name
                                    i =joint_idx*3 +3 # get start index in the frame vector
                                    if node_names[node_name]["level"] == 0:
                                        channels = node_names[node_name]["channels"][3:]
                                    else:
                                        channels = node_names[node_name]["channels"]
                                        
                                    euler_frame = np.concatenate((euler_frame,frame[i:i+3]), axis =0  )
                                joint_idx+=1
                            else:
                                if node_name in ["Bip01_R_Toe0","Bip01_L_Toe0"]:
                                    #special fix for unused toe parameters
                                    euler_frame = np.concatenate((euler_frame,([90.0,-1.00000000713e-06,75.0	])), axis = 0)
                                else:
                                    euler_frame = np.concatenate((euler_frame,([0,0,0])), axis = 0)  # set rotation to 0
    
                      euler_frames.append(euler_frame)        
        else:
            #check whether or not "Bip" frames should be ignored
            skip_joints = True
            if len(frame_data[0]) == len([n for n in node_names if "children" in node_names[n].keys() ])*4 +3:
                skip_joints = False
            euler_frames = []
            for frame in frame_data:
                euler_frame = frame[:3]     # copy root
                joint_idx = 0
                for node_name in node_names:  # go through the node names to
                                              # to append specific data
                    if "children" in node_names[node_name].keys(): #ignore end sites completely
                        if not node_name.startswith("Bip") or not skip_joints:
                            if node_name in ["Bip01_R_Toe0","Bip01_L_Toe0"]:
                                #special fix for unused toe parameters
                                euler_frame = np.concatenate((euler_frame,([90.0,-1.00000000713e-06,75.0	])), axis = 0)
                            else:
                                #print node_name
                                i =joint_idx*4 +3 # get start index in the frame vector
                                if node_names[node_name]["level"]==0:
                                    channels = node_names[node_name]["channels"][3:]
                                else:
                                    channels = node_names[node_name]["channels"]
                                    
                                euler_frame = np.concatenate((euler_frame,self._quaternion_to_euler(frame[i:i+4],channels)), axis =0  )
                            joint_idx+=1
                        else:
                            if node_name in ["Bip01_R_Toe0","Bip01_L_Toe0"]:
                                #special fix for unused toe parameters
                                euler_frame = np.concatenate((euler_frame,([90.0,-1.00000000713e-06,75.0	])), axis = 0)
                            else:
                                euler_frame = np.concatenate((euler_frame,([0,0,0])), axis = 0)  # set rotation to 0

                euler_frames.append(euler_frame)        
                    
        # create frame string
        frame_parameter_string = "MOTION\n"
        frame_parameter_string+= "Frames: " + str(len(euler_frames)) +"\n"
        frame_parameter_string+= "Frame Time: "+ str(frame_time)+"\n"
        for frame in euler_frames:
            frame_parameter_string+= ' '.join([str(f) for f in frame])
            frame_parameter_string+= '\n'

        return frame_parameter_string
        

    def _quaternion_to_euler(self,q,rotation_channel_order = \
                                        ['Xrotation','Yrotation','Zrotation']):
        q = quat(q)
        q = q.normalize()
        return self._matrix_to_euler(q.toMat3(),rotation_channel_order)
        
    def _matrix_to_euler(self,matrix,rotation_channel_order):
        """ Wrapper around the matrix to euler angles conversion implemented in 
            cgkit. The channel order gives the rotation order around
            the X,Y and Z axis. For each rotation order a different method is 
            provided by cgkit.
            TODO: Use faster code by Ken Shoemake in Graphic Gems 4, p.222
            http://thehuwaldtfamily.org/jtrl/math/Shoemake,%20Euler%20Angle%20Conversion,%20Graphic%27s%20Gems%20IV.pdf
            https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2012/07/euler-angles1.pdf
        """
        if rotation_channel_order[0] =='Xrotation':
              if rotation_channel_order[1] =='Yrotation':
                  euler = matrix.toEulerXYZ()
              elif rotation_channel_order[1] =='Zrotation':
                  euler = matrix.toEulerXZY()
        elif rotation_channel_order[0] =='Yrotation':
            if rotation_channel_order[1] =='Xrotation':
                 euler = matrix.toEulerYXZ()
            elif rotation_channel_order[1] =='Zrotation':
                 euler = matrix.toEulerYZX()   
        elif rotation_channel_order[0] =='Zrotation': 
            if rotation_channel_order[1] =='Xrotation':
                euler = matrix.toEulerZXY()    
            elif rotation_channel_order[1] =='Yrotation': 
                euler = matrix.toEulerZYX()  
        return [degrees(e) for e in euler] 
    




def main():
    infilename = "skeleton.bvh"
    bvh = BVHReader(infilename)

    print bvh.node_names
    nn_map = create_filtered_node_name_map(bvh)
    print nn_map.keys()
    print bvh.max_level
    # print bvh.parent_dict
    print bvh.node_names.keys()
    #print [item for item in bvh.gen_all_parents("Head_EndSite")]
    print bvh.get_angles( ("Hips","Xposition") )
    out_file = "test.bvh"
    print bvh.node_names["Hips"]
    motion_sample = [[-3.48819382695, 90.0, -0.459214461254, 0.06425137629868186, 0.6697008710556662, 0.7321745783647411, 0.106268013754948, 0.9995145636981352, -0.014852844355131557, -0.007587299106641207, 0.0263146890947825, 1.0, 0.0, 0.0, 0.0, 0.9847379711013986, 0.04973077169840712, -0.15847143898521374, 0.052007515258259494, 0.9970557976693092, 0.03998152426672592, 0.016777467893057084, -0.06324342355078998, -0.15396156184161824, -0.5371658728323265, 0.8292603270116773, -0.008716225333311026, 0.8653228617078748, 0.4008109010973167, 0.21721821390497387, 0.20828637525096155, 0.9809514515642369, -2.576577675113159e-08, 0.19425305576493565, 3.602137076971754e-08, 0.723548329120542, -0.6742273278587321, -0.07509920225016631, 0.12749680630013477, -0.15508698214919572, 0.5399910840429485, 0.8220751458944435, -0.09246681363930477, 0.9246393769605211, -0.2925493858554578, 0.04995713519893805, -0.23866538092561565, 0.9799914115381161, 1.897373680313075e-08, 0.19903977821412494, 5.304918005395855e-08, 0.6698195602925033, 0.72478904142108, -0.0921621922630837, -0.1323961192887924, -0.08090559448381732, -0.005809439190131918, -0.13817241487826862, 0.9870810093220433, 0.9072787630150794, -6.559177073179788e-09, 0.42052972092567376, 5.290843683454061e-08, 0.9937604385939052, 0.031105657125272565, -0.07651742989583284, 0.07495139560135469, -0.021473027471794016, 0.05391698741069397, 0.11030654242633643, 0.9922017608620223, 0.9989918897376777, 9.109596944707124e-09, 0.04489102625629779, 9.109596944707124e-09, 0.9981276592923625, 0.018932127754130505, -0.012400207309823373, 0.05682415994033111], [-3.01607573024, 89.989022, -1.21928835199, 0.05980025458935302, 0.6690397350608593, 0.7337019029308177, 0.10242695004949032, 0.9994651574711655, -0.014384282359591598, -0.008565346246385839, 0.02809139131153647, 1.0, 0.0, 8.726646259971647e-09, 0.0, 0.984853306964563, 0.048088614574673574, -0.15821565310654842, 0.05214648618383991, 0.9972053798947869, 0.03805048214433433, 0.017392031198978385, -0.06189594791463493, -0.15156487502486382, -0.5377436769276789, 0.8293397462653426, -0.007443913455658202, 0.8652293141451448, 0.4023278181682907, 0.21478460342550937, 0.20827418179700458, 0.9810689271295503, -1.543289046220026e-08, 0.19365887591554637, 5.181456473272057e-09, 0.7239806954189358, -0.6728407308607974, -0.07848864095860933, 0.13029519099296244, -0.15596152365331706, 0.5395570620583672, 0.8224484016156243, -0.09018207472987601, 0.9252999076531249, -0.29230390388632455, 0.05042898810960447, -0.2362952090673325, 0.9797754508141561, -5.635867013952493e-08, 0.20010013988479392, 4.876885585500743e-09, 0.6690902507894037, 0.7256434600548237, -0.09135762719137672, -0.13196055900229672, -0.08390333786587717, 0.006078141079318436, -0.14097699299963434, 0.9864323461549535, 0.8985914393817626, -9.171190846625866e-10, 0.43878630911846955, 5.927123172025691e-08, 0.9946391178270254, 0.022630665946227813, -0.06776951660438478, 0.07475406922914121, -0.023019459168434727, 0.046738784011675256, 0.11222734487837994, 0.9923157832213756, 0.9989787006179313, 1.7829768584157662e-08, 0.04518357789849597, 9.506335943698903e-09, 0.9981176204591331, 0.022303968276645894, -0.015967434426379996, 0.05485243628091715], [-2.57422945936, 89.989006, -1.94837781352, 0.055316722586052344, 0.6686319864748833, 0.7349163875942845, 0.09883941577304424, 0.9994161457083771, -0.013769530883255696, -0.009773086115776876, 0.02970276930592276, 1.0, 0.0, 0.0, 0.0, 0.9849591544922657, 0.04651356952265195, -0.15805269926192084, 0.052070107433764404, 0.9973495214620444, 0.03610083408363249, 0.018126492279325557, -0.06051522201516479, -0.14912770231315073, -0.5383017948234897, 0.8294293302998237, -0.006252370030314994, 0.8651955960778546, 0.4037722119593215, 0.21244197739878934, 0.20802160372388045, 0.9811379870184311, -2.0625322710800183e-08, 0.1933086920689806, 2.062532271080019e-08, 0.723722428187896, -0.6724258454203111, -0.0806354858563858, 0.13254149451888825, -0.1568926264033345, 0.5390300607766283, 0.8228764068684122, -0.08778220992171441, 0.9260238779389792, -0.2919483866087157, 0.05100942878004009, -0.23376046547471835, 0.9795643161470632, 1.5569113824775242e-08, 0.20113117742243586, 3.5948445740843026e-08, 0.6685261388235837, 0.7263615186773641, -0.09039073926041463, -0.13153425466508598, -0.08652074014144114, 0.018100786824955636, -0.14367956694795567, 0.985668658872414, 0.8906369209614811, 1.2560428251952397e-08, 0.4547151581157679, 6.962245357261958e-08, 0.9952875603624685, 0.014512515076351314, -0.06035698937729492, 0.07449223400592872, -0.02441918647323469, 0.03935958247269339, 0.11394820303847415, 0.9924063349373904, 0.998968375125182, 9.11393192174323e-09, 0.04541129264570333, 9.11393192174323e-09, 0.9980683136283465, 0.025483082748248696, -0.02017702298725066, 0.05294470292754988], [-2.1559783432, 90.021996, -2.6562592372, 0.050824399618936165, 0.6687256288203509, 0.7355947258100947, 0.09551603594461586, 0.9993736090650138, -0.013009545212888399, -0.011194205903832655, 0.030948844758814726, 1.0, 0.0, 8.726646259971647e-09, 0.0, 0.9850559476745429, 0.045047755121248305, -0.15795163591248623, 0.051833969769969575, 0.9974860503901929, 0.03416139159830398, 0.01896692361037059, -0.05911712450550687, -0.14672296602181956, -0.5388479974522357, 0.8295115612158047, -0.005077075220442165, 0.8652889011038827, 0.4049667578647572, 0.2101063143360431, 0.20768336314463468, 0.9811646363472017, -2.7506083652102302e-08, 0.19317338424757716, 2.7506083652102302e-08, 0.7231361792755944, -0.6726180117187919, -0.08151733278488085, 0.13421624711559252, -0.15792274244298568, 0.5385244013779336, 0.8232734175209507, -0.08528045815542756, 0.9267670577445986, -0.29149690910080917, 0.0517433384006285, -0.23120337280250278, 0.9793838364219759, 2.0303867811337455e-09, 0.20200817051636824, 4.951756391341079e-08, 0.6680005407237063, 0.7270700584462834, -0.0893871141115121, -0.13097462171951305, -0.08829424175856818, 0.03016504125240914, -0.14573099039926432, 0.9849145524337477, 0.8833335917686508, 2.9579199043603387e-08, 0.46874488333536835, 7.661279606798873e-08, 0.9957798496560126, 0.007320160914455921, -0.054487469632693505, 0.07348484140462086, -0.025505088801461767, 0.03195659484195292, 0.11563093736521793, 0.9924503780116052, 0.9989575555927961, -1.7036738536952348e-08, 0.04564868151508347, 7.920829424751815e-09, 0.9980183063179169, 0.028121778121083893, -0.024957914416484858, 0.05045521140184646], [-1.72031104007, 90.082032, -3.38881566434, 0.04637019024553042, 0.6689703385758362, 0.7360617837996083, 0.09242046305912727, 0.9993436569639274, -0.012098852695389177, -0.012760944370388265, 0.03167067015702971, 1.0, 0.0, 0.0, 0.0, 0.9851474283489995, 0.0436832568817692, -0.1578845269507138, 0.051466432119427044, 0.9976176942803617, 0.03221824827455425, 0.019905722009837933, -0.05766006215849257, -0.14439588718271362, -0.5394236727176598, 0.8295527243376186, -0.003769167088812253, 0.8654122702673219, 0.40601193368689403, 0.20782685365979922, 0.20742206023578513, 0.9811558710930212, 2.3868977459874148e-08, 0.19321789932532915, 3.7621089368706495e-08, 0.722100988402669, -0.6736010474792874, -0.081295937484028, 0.13499171060212767, -0.15907473351436338, 0.5380268508657023, 0.8236317092700539, -0.08278372056001812, 0.9274825965791273, -0.29107823344993616, 0.052696844043865446, -0.2286318824711927, 0.9792526381405933, -1.856319166229282e-08, 0.20264321033453694, 2.8877178648064334e-08, 0.6675198726537874, 0.7277823558285351, -0.08806326146777224, -0.1303645816032788, -0.089065137380985, 0.042191870836127285, -0.14709144842079822, 0.984200870320752, 0.8765445206544168, 1.2157325494641078e-08, 0.48132079044096354, 7.07899461756692e-08, 0.9961093467307864, 0.0019605401249315865, -0.05096499678500833, 0.07186720212091596, -0.026164001949943227, 0.024639676600319814, 0.11727814402919534, 0.9924485721044001, 0.9989473966995137, 9.11775579649233e-09, 0.045870454840391914, 9.11775579649233e-09, 0.9979739837101289, 0.030379707661672087, -0.02991973707040631, 0.047220869684542074], [-1.31762108798, 90.15004, -4.09821829868, 0.04177796573632585, 0.6693653305450239, 0.7363418022117936, 0.08947293534233206, 0.9993254208373918, -0.011164948584391507, -0.014309735946897901, 0.03192614365074425, 1.0, 0.0, 8.726646259971647e-09, 0.0, 0.9852463059462596, 0.04231556912537897, -0.15783546538290838, 0.05086329812552474, 0.9977498770672196, 0.030329993946099452, 0.020734455752996324, -0.05608347906652618, -0.14215821885598257, -0.5400690576575414, 0.8295246836167186, -0.0022919518262850963, 0.8656321439805502, 0.40679605245185946, 0.20550824860709324, 0.2072783702455835, 0.9811086297642909, -3.497078909539826e-09, 0.19345763516086764, 2.3997127515335835e-08, 0.7208895562255355, -0.675052820263714, -0.08015325284671201, 0.13489771546267149, -0.1603337030764723, 0.5375228233506685, 0.8239642185346229, -0.08028252991492242, 0.9280842251213967, -0.2908803272256483, 0.05394802701685374, -0.2261369423488909, 0.9791441060188025, 3.866119994642599e-09, 0.20316697479816098, 5.803954343456382e-08, 0.6670716147939245, 0.7285129990151635, -0.08661454778002303, -0.12954609648700544, -0.08884168186223115, 0.054210679448530155, -0.14773831763947193, 0.9835353309864763, 0.8702276054665722, 1.7918514832998432e-08, 0.4926498905753559, 7.722828359607224e-08, 0.9964073224159713, -0.001370589134268812, -0.04886608398796971, 0.06915688799346627, -0.026416600668075947, 0.017479084508069696, 0.11908519648289456, 0.9923786378154936, 0.998943292885704, 4.010743834775035e-10, 0.04595973888814494, 8.717424750784791e-09, 0.9979320948546747, 0.03246975981488368, -0.03479609360095877, 0.0432027849407703], [-0.937370431536, 90.228012, -4.82415991617, 0.037296249261353345, 0.6699369544668607, 0.7364297360702555, 0.0863985570608479, 0.9993189580609417, -0.010199918855062749, -0.015847291168780798, 0.03172451856145222, 1.0, 0.0, 0.0, 0.0, 0.9853752132740964, 0.040963758008050626, -0.1576195559451332, 0.05013716364680288, 0.9978756100950688, 0.028432122833755666, 0.021429175126942486, -0.054558882154418134, -0.13999533710878553, -0.5407626246639382, 0.8294435677961469, -0.0006761626765379031, 0.8658560179084298, 0.407441513550748, 0.20325038406214374, 0.20730183468996105, 0.9810264864750116, 1.0252938797133056e-08, 0.1938737548883146, 1.0252938797133056e-08, 0.7195066716229676, -0.6769261694679064, -0.07823638596290147, 0.13401559047741754, -0.1616183544537246, 0.5370903798094465, 0.8242322484247556, -0.0778115163487231, 0.9285829589970614, -0.2908505893222537, 0.05541711250644457, -0.22376006478504507, 0.979045575556808, 7.450162403800001e-09, 0.2036412556007078, 7.511695449090778e-08, 0.6668274819984652, 0.7290916734460072, -0.08521338056692451, -0.12847225667685647, -0.08776176677162038, 0.06637241195115869, -0.14798305024186978, 0.982849730155135, 0.8643879415696369, 1.1784497052676077e-08, 0.5028255030017877, 7.173381959588926e-08, 0.9967044570632646, -0.003073206056390903, -0.04748359198954124, 0.06569694944613955, -0.02630654405282217, 0.010708054409170102, 0.12116818160143779, 0.9922255666319875, 0.9989405361110624, 1.78363983210281e-08, 0.04601961879615072, 9.520594561893177e-09, 0.9978893709248847, 0.03437913941193939, -0.03954222651819058, 0.03835740463609001], [-0.569527038436, 90.308022, -5.54882475895, 0.03293588097906855, 0.6707228523887885, 0.7362867242675372, 0.08323426394796767, 0.9993229814390465, -0.009186436435601806, -0.01726867735229836, 0.031160567002097354, 1.0, 0.0, 0.0, 0.0, 0.9854980144297546, 0.03963493842984676, -0.15745615650771577, 0.04929801201358108, 0.9979878849155488, 0.026478775842778354, 0.022060936629208595, -0.053220025053539184, -0.1379696660349697, -0.5414080878908116, 0.8293615613119507, 0.0010267613244806656, 0.8661995573799967, 0.4078432835558766, 0.20090740773218094, 0.20736054680530375, 0.9808925958082403, -2.228416946556722e-08, 0.19455003338106017, 1.2026497441741168e-08, 0.7179594827252911, -0.6790916493411087, -0.07590506344186088, 0.1326918773093988, -0.16304335782222976, 0.5367160557846663, 0.8244151778211917, -0.0754476872478983, 0.9289976772302149, -0.29094996663320677, 0.057008712456367704, -0.22149817001742664, 0.9789596490675084, -3.701255578620033e-08, 0.2040539279151863, 5.7660037800343767e-08, 0.6666572526782629, 0.7296013759272649, -0.08397688305907268, -0.12727066750903798, -0.08597616155764536, 0.07844735548004189, -0.14798423514484835, 0.98211749715123, 0.8598196211992094, 1.2905444608113366e-08, 0.5105979034434462, 7.080859815404379e-08, 0.9969932425608985, -0.0037846319212927063, -0.046494123073586455, 0.06187444843183504, -0.02591256728020493, 0.004230276702300058, 0.12356717848948272, 0.9919888084127362, 0.9989409665792647, 4.015153832586262e-10, 0.046010273740593986, 8.717404449931402e-09, 0.997820803081747, 0.03609293768751351, -0.04411244488670108, 0.03324209669230294], [-0.211279554568, 90.391007, -6.28086558269, 0.02864154144378839, 0.6715964231068247, 0.7360301736531266, 0.07998431124908878, 0.9993306986012546, -0.008156171994484256, -0.018616564738178028, 0.03041472026625687, 1.0, 0.0, 0.0, 0.0, 0.9856351659105304, 0.038284250471150985, -0.15728096521437343, 0.04816984396505998, 0.9980965958550693, 0.024565429789629584, 0.022679847908133915, -0.0518203579748149, -0.13600406152204558, -0.5421348068502782, 0.8292075111068454, 0.0027658588015524455, 0.8666793712980592, 0.40794659819063056, 0.19850106213780347, 0.2074699224496341, 0.9807178324578322, 3.4108812479768586e-09, 0.19542909992938914, 1.7116755209411286e-08, 0.7161103742033659, -0.6816503614710545, -0.07316821887300318, 0.13109206082615482, -0.16455570554714066, 0.5362388382016979, 0.8246333502160613, -0.07313799209310706, 0.9293981036764745, -0.2909837195044038, 0.058544772367485405, -0.2193630540779264, 0.9788562427014709, -3.0215449188454975e-08, 0.2045493977609221, 5.086977400119596e-08, 0.6667220655573642, 0.7298688340734101, -0.08299795815634103, -0.12603377043126632, -0.08371486640931368, 0.09037736042847971, -0.1478861964198784, 0.9813019039890387, 0.8565665429305582, -1.2571321457200393e-09, 0.5160365854588106, 6.114824192964242e-08, 0.9972886643497383, -0.004526209207748834, -0.04566187217972848, 0.05753109436439879, -0.02534394028248977, -0.002014889159730119, 0.1261405143111553, 0.9916865409805404, 0.9989421868345083, 4.01284119054346e-10, 0.045983772814878573, 8.717415098667261e-09, 0.9976896355136413, 0.037964712934279624, -0.04865338763828217, 0.028406330840056993], [0.123439237379, 90.46901, -7.01010321923, 0.024402645417183937, 0.672488152140678, 0.7357080731028659, 0.07679731309315298, 0.9993365861278689, -0.00709513988189403, -0.01997607305061907, 0.029614238498232814, 1.0, -8.726646259971647e-09, 0.0, 0.0, 0.9857835721937436, 0.03696583623173544, -0.15709036737621912, 0.046764219465692536, 0.9981941565434386, 0.022693630401140932, 0.023312887983234873, -0.05049687352332777, -0.13406160708495796, -0.5428950746482328, 0.8290190117050743, 0.004461127663479572, 0.867290719277103, 0.4077650624617429, 0.1960633296977552, 0.20759005958669158, 0.9805074777641516, -3.4126611966569893e-09, 0.196481770270377, 2.3954998835431714e-08, 0.7139619874287986, -0.6845434693406869, -0.07018870522140136, 0.1293524825779519, -0.16605960946908116, 0.5356528341590087, 0.8249092506051671, -0.07088706247559963, 0.9298606047415269, -0.29077776677614825, 0.06001784692274727, -0.21726804677438366, 0.9787339704707144, 1.2530901698856984e-08, 0.20513365166794784, 5.978745600040831e-08, 0.6670392977675274, 0.7298364067274, -0.08247989695920808, -0.12487778524076805, -0.08116364956796707, 0.10192980926066213, -0.14791276833219968, 0.9803798187117063, 0.854842375767721, 3.062271449049283e-08, 0.5188877649278212, 8.925785545461827e-08, 0.9975701769783364, -0.0060959371305749676, -0.04527510432302051, 0.05259987150602908, -0.02468815637674012, -0.008145261566295074, 0.12876354391333353, 0.991334504295908, 0.998944160231637, -8.316522484206848e-09, 0.04594088309022261, 8.316522484206848e-09, 0.9974954532833492, 0.039898837096365054, -0.053164082753454085, 0.024176099403909624]]

    BVHWriter(out_file,bvh,motion_sample,frame_time= 0.013889,\
                                        is_quaternion = True)
    #node_names = get_node_names(bvh)

    return


if __name__ == "__main__":
    main()
