import sys
import numpy as np

from joblib import Parallel, delayed

from generate_input_1rotation import Participant

sys.path.append('./motion')

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

average_hip_height = 94.963277

""" Options """

rng = np.random.RandomState(1234)

""" Data """
 
#path = "../../../data/data_praktikum_normalized/"
path = "../../../data/data_praktikum_aligned/"

participants = {
    'female' : [
        "NXDS9232",
        "JUIDM3242",
        "JKYRF234F",
        "MNCXJUD2",
        "ASD23S2D",
        "NXJD23356Z",
        "2EDRCG34",
        "98211XYDS",
        "985CCLPFE",
        "IJMX0867DSX",
        "IXJE534",
        "898MXJDFI", # 12
        #"898XDXYDS",
        #"JXDSD93223",
    ],
    'male' : [
        "JDXD98783",
        "ZXD2132",
        "89XFDD21FD",
        "NXHDS231",
        "XDSD123",
        "565MNMXDEFC",
        "JNGXI873",
        "23KJINCSE3",
        "214CSDF",
        "673DCDSD",
        "121MDXDEW",
        "787DXDECX", # 12
    ]
}


""" Sampling Patch Heightmap """

def patchfunc(P, Xp, hscale=3.937007874, vscale=3.0):

    Xp = Xp / hscale + np.array([P.shape[1]//2, P.shape[2]//2])

    A = np.fmod(Xp, 1.0)
    X0 = np.clip(np.floor(Xp).astype(np.int), 0, np.array([P.shape[1]-1, P.shape[2]-1]))
    X1 = np.clip(np.ceil (Xp).astype(np.int), 0, np.array([P.shape[1]-1, P.shape[2]-1]))

    H0 = P[:,X0[:,0],X0[:,1]]
    H1 = P[:,X0[:,0],X1[:,1]]
    H2 = P[:,X1[:,0],X0[:,1]]
    H3 = P[:,X1[:,0],X1[:,1]]

    HL = (1-A[:,0]) * H0 + (A[:,0]) * H2
    HR = (1-A[:,0]) * H1 + (A[:,0]) * H3

    return (vscale * ((1-A[:,1]) * HL + (A[:,1]) * HR))[...,np.newaxis]


def format_print(x, y, slc_start):
    j = 21
    w = 12
    g = 3

    print("Path Location")
    print("    f" , end="")
    for i in range(len(x)):
        print("%15d               "%((slc_start + i)*2), end="")
    print("", end="\n")
    for j in range(w):
        print("%5d"%(j - w//2), end = "")
        for i in range(len(x)):
            # if j >= 6:
            #     offset = 4
            #     pred = (y[i,offset + j-6], y[i,offset + j])
            # else:
            #     pred = (0,0)
            print("{:{w}.{p}f},{:{w}.{p}f} |{:{w}.{p}f},{:{w}.{p}f}  ".format(x[i,j],x[i,w + j],x[i,2*w + j], x[i,3 * w + j], w = 6, p = 2), end="")
        print("", end="\n")

    print("\nLocation Pred")
    print("     ", end="")
    for f in range(len(x)):
        print("   (%5.2f,%5.2f), %5.2f       "%(y[f,0], y[f,1], y[f,2]), end="")

    print("\nGait Input")
    base = 4 * w
    for j in range(w):
        print("%5d"%(j - w//2), end = "")
        for f in range(len(x)):
            print("     %6.4f,%6.4f,%6.4f     "%(x[f,base+j], x[f,base+w+j],x[f,base+2*w+j]), end="")
        print("",end="\n")

#def output_test_scenario(x, filepath):
#    np.savez_compressed(filepath, x)

""" Phases, Inputs, Outputs """

P, X, Y = [], [], []

participant_data = {}

def process(path, part, gender):
    global participant_data
    participant_data[part] = Participant(path, part, -1 if gender == "male" else 1)
    participant_data[part].generate_train_test(rng, randomized_indices_file = True)
    #participant_data[part].save_models("../../../experimental/participants_normalized/" + gender + "/" + part + "_%s.npz")
    participant_data[part].save_models("../../../experimental/participants_aligned/" + gender + "/" + part + "_%s.npz")

#for data in data_terrain:
d = []
for gender in participants:
    for p in participants[gender]:
        d.append((p, gender))

Parallel(n_jobs=24, backend="threading")(((delayed(process)(path, part, gender)) for (part, gender) in d))



for gender in participants:
    for part in participants[gender]:
        P.append(participant_data[part].Pun.astype(np.float32))
        X.append(participant_data[part].Xun.astype(np.float32))
        Y.append(participant_data[part].Yun.astype(np.float32))



""" Clip Statistics """

print('Total Clips: %i' % len(X))
print('Shortest Clip: %i' % min(map(len,X)))
print('Longest Clip: %i' % max(map(len,X)))
print('Average Clip: %i' % np.mean(list(map(len,X))))



""" Merge Clips """

print('Merging Clips...')

Xun = np.concatenate(X, axis=0)
Yun = np.concatenate(Y, axis=0)
Pun = np.concatenate(P, axis=0)


print(Xun.shape, Yun.shape, Pun.shape)

print('Saving Database...')

#np.savez_compressed('./masterThesis/database.npz', Xun=Xun, Yun=Yun, Pun=Pun)


# def print_positions(pos):
#     names = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "LeftUpLeg", "LeftLeg", "LeftFoot", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"]
#     pos = np.concatenate([pos[0:17], pos[36:40]], axis=0)
#     for i in range(len(pos)):
#         print(names[i], pos[i])
