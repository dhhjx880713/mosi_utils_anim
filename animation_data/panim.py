import numpy as np
from ..utilities import load_json_file, write_to_json_file


class Panim(object):
    """ A customized animation data format for point cloud data. It is also used in Unity.
    """

    def __init__(self):
        self.motion_data = None
        self.has_skeleton = False
        self.skeleton = None
    
    def load(self, filename):
        """
        
        Arguments:
            filename {str} -- path to .panim file
        """
        json_data = load_json_file(filename)
        self.motion_data = json_data['motion_data']
        self.has_skeleton = json_data['has_skeleton']
        self.skeleton = json_data['skeleton']
    
    def save(self, saveFilename):
        """
        
        Arguments:
            saveFilename {str} -- save path
        """
        output_data = {'motion_data': self.motion_data,
                       'has_skeleton': self.has_skeleton,
                       'skeleton': self.skeleton
        }
        write_to_json_file(saveFilename, output_data)
    
    def setSkeleton(self, ske_description):
        """list of joints
        
        Arguments:
            ske_description {list} -- a list of dictionary for joint information
        """
        self.skeleton = ske_description
        self.has_skeleton = True

    def setMotionData(self, motion_data):
        """
        
        Arguments:
            motion_data {numpy.array3d} -- n_frames * n_joints * 3
        """
        self.motion_data = np.asarray(motion_data).tolist()

    def convert_to_unity_format(self, scale=1.0):
        """ 
        Convert motion_data from numpy array to dictionary for Unity loading. The coordinate system is flipped because Unity use left-hand coordinate system
        """
        output_frames = []
        for frame in self.motion_data:
            world_pos = []
            for point in frame:
                world_pos.append({'x': -point[0] * scale,
                                  'y': point[1] * scale,
                                  'z': point[2] * scale})
                output_frame = {'WorldPos': world_pos}
        output_frames.append(output_frame)
        ## update motion_data
        self.motion_data = output_frames     

    def get_joint_index(self, joint_name):
        for i in range(len(self.skeleton)):
            if self.skeleton[i]['name'] == joint_name:
                return self.skeleton[i]['index']
        return None

    def mirror(self, joint_mapping=None):
        """[summary]
        
        Arguments:
            joint_mapping {[type]} -- [description]
        """
        assert self.skeleton is not None
        motion_data = np.asarray(self.motion_data)
        mirrored_data = np.zeros(motion_data.shape)
        if joint_mapping is None: 
            for i in range(len(self.skeleton)):
                if 'Left' in self.skeleton[i]['name']:
                    mirrored_joint_name = self.skeleton[i]['name'].replace("Left", "Right")
                    mirrored_joint_index = self.get_joint_index(mirrored_joint_name)

                    mirrored_data[:, self.skeleton[i]['index'], :] = motion_data[:, mirrored_joint_index, :]
                    mirrored_data[:, self.skeleton[i]['index'], 0] = - mirrored_data[:, self.skeleton[i]['index'], 0]
                elif 'Right' in self.skeleton[i]['name']:
                    mirrored_joint_name = self.skeleton[i]['name'].replace("Right", "Left")
                    mirrored_joint_index = self.get_joint_index(mirrored_joint_name)
                    mirrored_data[:, self.skeleton[i]['index'], :] = motion_data[:, mirrored_joint_index, :]
                    mirrored_data[:, self.skeleton[i]['index'], 0] = - mirrored_data[:, self.skeleton[i]['index'], 0]
                ### handle special 
                elif self.skeleton[i]['name'] == "RThumb":
                    mirrored_joint_name = "LThumb"
                    mirrored_joint_index = self.get_joint_index(mirrored_joint_name)
                    mirrored_data[:, self.skeleton[i]['index'], :] = motion_data[:, mirrored_joint_index, :]
                    mirrored_data[:, self.skeleton[i]['index'], 0] = - mirrored_data[:, self.skeleton[i]['index'], 0]  
                elif self.skeleton[i]['name'] == "LThumb":
                    mirrored_joint_name = "RThumb"
                    mirrored_joint_index = self.get_joint_index(mirrored_joint_name)
                    mirrored_data[:, self.skeleton[i]['index'], :] = motion_data[:, mirrored_joint_index, :]
                    mirrored_data[:, self.skeleton[i]['index'], 0] = - mirrored_data[:, self.skeleton[i]['index'], 0]                                        
                else:
                    mirrored_data[:, self.skeleton[i]['index'], :] = motion_data[:, self.skeleton[i]['index'], :]
                    mirrored_data[:, self.skeleton[i]['index'], 0] = - mirrored_data[:, self.skeleton[i]['index'], 0]
        else:
            pass
        return mirrored_data