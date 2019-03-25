import re
import numpy as np
import math

from .Animation import Animation
from .Quaternions import Quaternions

channelmap = {
    'Xrotation' : 'x',
    'Yrotation' : 'y',
    'Zrotation' : 'z'   
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x' : 0,
    'y' : 1,
    'z' : 2,
}

def load(filename, start=None, end=None, order=None, world=False):
    """
    Reads a BVH file and constructs an animation
    
    Parameters
    ----------
    filename: str
        File to be opened
        
    start : int
        Optional Starting Frame
        
    end : int
        Optional Ending Frame
    
    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'
        
    world : bool
        If set to true euler angles are applied
        together in world space rather than local
        space

    Returns
    -------
    
    (animation, joint_names, frametime)
        Tuple of loaded animation and joint names
    """
    
    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False
    
    names = []
    orients = Quaternions.id(0)
    offsets = np.array([]).reshape((0,3))
    parents = np.array([], dtype=int)
    # directions = np.array([]).reshape((0,3))
    directions = np.array([
        [0.000000, 0.100000, 0.000000],  # Hips
        [1.363060, -1.794630, 0.839290],  # LHipJoint
        [2.448110, -6.726130, -0.000000],  # LeftUpLeg
        [2.562200, -7.039591, 0.000000],  # LeftLeg
        [0.157640, -0.433110, 2.322551],  # LeftFoot
        [0.000000, 0.100000, 0.000000],  # LeftToeBase
        [-1.305520, -1.794630, 0.839290],  # RHipJoint
        [-2.542531, -6.985552, 0.000001],  # RightUpLeg
        [-2.568260, -7.056230, -0.000000],  # RightLeg
        [-0.164730, -0.452590, 2.363150],  # RightFoot
        [0.000000, 0.100000, 0.000000],  # RightToeBase
        [0.028270, 2.035590, -0.193380],  # LowerBack
        [0.056720, 2.048850, -0.042750],  # Spine
        [0.056720, 2.048850, -0.042750],  # Spine1
        [-0.054170, 1.746240, 0.172020],  # Neck
        [0.104070, 1.761360, -0.123970],  # Neck1
        [0.000000, 0.100000, 0.000000],  # Head
        [3.362410, 1.200890, -0.311210],  # LeftShoulder
        [4.983000, 0.000000, -0.000000],  # LeftArm
        [3.483560, 0.000000, 0.000000],  # LeftForeArm
        [3.483561, 0.000000, 0.000000],  # LeftHand
        [0.715260, 0.000000, 0.000000],  # LeftFingerBase
        [0.000000, 0.100000, 0.000000],  # LeftHandIndex1
        [0.000000, 0.100000, 0.000000],  # LThumb
        [-3.136600, 1.374050, -0.404650],  # RightShoulder
        [-5.241900, 0.000000, -0.000000],  # RightArm
        [-3.444170, 0.000000, 0.000000],  # RightForeArm
        [-3.444170, 0.000000, 0.000000],  # RightHand
        [-0.622530, 0.000000, 0.000000],  # RightFingerBase
        [0.000000, 0.100000, 0.000000],  # RightHandIndex1
        [0.000000, 0.100000, 0.000000]  # RThumb
    ])

    
    for line in f:
        
        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets    = np.append(offsets,    np.array([[0,0,0]]),   axis=0)
            orients.qs = np.append(orients.qs, np.array([[1,0,0,0]]), axis=0)
            parents    = np.append(parents, active)
            # directions = np.append(directions, np.array([[0,0.1,0]]), axis = 0)
            active = (len(parents)-1)
            continue

        if "{" in line: continue

        if "}" in line:
            if end_site: end_site = False
            else: active = parents[active]
            continue
        
        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            par = -1
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            # if parents[active] >= 0:
                # par = parents[active]
                # if par == 0 or par == 5 or par == 10 or par == 16 or par == 22 or par == 23 or par == 29 or par == 30:
                #     directions[par] = np.array([[0, 0.1, 0]])
                #     print("joint direction 1: ", par, names[par], directions[par])
                # else:
                # # if active == 13 or active == 36:
                # #     # Spine3 is connected to Neck, not to Right Shoulder or Left Shoulder
                # #     continue
                #     if end_site:
                #         par += 1
                #     directions[par] = np.array([list(map(float, offmatch.groups()))])
                #     print("joint direction: ", par, names[par], directions[par])

                #directions[par] /= math.sqrt(np.sum(directions[par]**2))
            continue
           
        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2+channelis:2+channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets    = np.append(offsets,    np.array([[0,0,0]]),   axis=0)
            orients.qs = np.append(orients.qs, np.array([[1,0,0,0]]), axis=0)
            parents    = np.append(parents, active)
            # directions = np.append(directions, np.array([[0, 0, 0]]), axis = 0)
            active = (len(parents)-1)
            continue
        
        if "End Site" in line:
            end_site = True
            continue
              
        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start)-1
            else:
                fnum = int(fmatch.group(1))
            jnum = len(parents)
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue
        
        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue
        
        if (start and end) and (i < start or i >= end-1):
            i += 1
            continue
        
        dmatch = line.strip().split(' ')
        if dmatch and not dmatch == ['']:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i - start if start else i
            if   channels == 3:
                positions[fi,0:1] = data_block[0:3]
                rotations[fi, : ] = data_block[3: ].reshape(N,3)
            elif channels == 6:
                print(fi)
                data_block = data_block.reshape(N,6)
                positions[fi,:] = data_block[:,0:3]
                rotations[fi,:] = data_block[:,3:6]
            elif channels == 9:
                positions[fi,0] = data_block[0:3]
                data_block = data_block[3:].reshape(N-1,9)
                rotations[fi,1:] = data_block[:,3:6]
                positions[fi,1:] += data_block[:,0:3] * data_block[:,6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()
    
    rotations = Quaternions.from_euler(np.radians(rotations), order=order, world=world)
    
    return (Animation(rotations, positions, orients, offsets, parents, directions), names, frametime)
    

    
def save(filename, anim, names=None, frametime=1.0/24.0, order='zyx', positions=False, orients=True):
    """
    Saves an Animation to file as BVH
    
    Parameters
    ----------
    filename: str
        File to be saved to
        
    anim : Animation
        Animation to save
        
    names : [str]
        List of joint names
    
    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'
    
    frametime : float
        Optional Animation Frame time
        
    positions : bool
        Optional specfier to save bone
        positions for each frame
        
    orients : bool
        Multiply joint orients to the rotations
        before saving.
        
    """
    
    if names is None:
        names = ["joint_" + str(i) for i in range(len(anim.parents))]
    
    with open(filename, 'w') as f:

        t = ""
        f.write("%sHIERARCHY\n" % t)
        f.write("%sROOT %s\n" % (t, names[0]))
        f.write("%s{\n" % t)
        t += '\t'

        f.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[0,0], anim.offsets[0,1], anim.offsets[0,2]) )
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % 
            (t, channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))

        save_order = [0]
            
        for i in range(anim.shape[1]):
            if anim.parents[i] == 0:
                t = save_joint(f, anim, names, t, i, save_order, order=order, positions=positions)
      
        t = t[:-1]
        f.write("%s}\n" % t)

        f.write("MOTION\n")
        f.write("Frames: %i\n" % anim.shape[0]);
        f.write("Frame Time: %f\n" % frametime);
            
        #if orients:        
        #    rots = np.degrees((-anim.orients[np.newaxis] * anim.rotations).euler(order=order[::-1]))
        #else:
        #    rots = np.degrees(anim.rotations.euler(order=order[::-1]))
        rots = np.degrees(anim.rotations.euler(order=order[::-1]))
        poss = anim.positions
        
        for i in range(anim.shape[0]):
            for j in save_order:
                
                if positions or j == 0:
                
                    f.write("%f %f %f %f %f %f " % (
                        poss[i,j,0],                  poss[i,j,1],                  poss[i,j,2], 
                        rots[i,j,ordermap[order[0]]], rots[i,j,ordermap[order[1]]], rots[i,j,ordermap[order[2]]]))
                
                else:
                    
                    f.write("%f %f %f " % (
                        rots[i,j,ordermap[order[0]]], rots[i,j,ordermap[order[1]]], rots[i,j,ordermap[order[2]]]))

            f.write("\n")
    
    
def save_joint(f, anim, names, t, i, save_order, order='zyx', positions=False):
    
    save_order.append(i)
    
    f.write("%sJOINT %s\n" % (t, names[i]))
    f.write("%s{\n" % t)
    t += '\t'
  
    f.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[i,0], anim.offsets[i,1], anim.offsets[i,2]))
    
    if positions:
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % (t, 
            channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
    else:
        f.write("%sCHANNELS 3 %s %s %s\n" % (t, 
            channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
    
    end_site = True
    
    for j in range(anim.shape[1]):
        if anim.parents[j] == i:
            t = save_joint(f, anim, names, t, j, save_order, order=order, positions=positions)
            end_site = False
    
    if end_site:
        f.write("%sEnd Site\n" % t)
        f.write("%s{\n" % t)
        t += '\t'
        f.write("%sOFFSET %f %f %f\n" % (t, 0.0, 0.0, 0.0))
        t = t[:-1]
        f.write("%s}\n" % t)
  
    t = t[:-1]
    f.write("%s}\n" % t)
    
    return t