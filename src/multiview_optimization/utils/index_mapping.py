import numpy as np

def get_shifted_op_joint_ids():
#     return [12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
#                                      8, 1, 4, 7]
#     return [0, 2, 1]
#     return [0, 2, 1, 12]
    # return [0, 2, 1, 12, 16, 17]
    return [0, 2, 1, 12, 16, 17]
#     return []

def get_smplx_ids_op_face():
    inds_76_144 = np.concatenate([
        np.arange(51, 68),
        np.arange(0,51)
    ]) + 76
    inds_irises = np.asarray([56, 57])
    return np.concatenate([inds_76_144, inds_irises])

def get_smplx_ids_op_mouth():    
    return get_smplx_ids_op_face()[48:68]

def get_smplx_ids_op_face_no_mouth():    
    all_face = get_smplx_ids_op_face()
    return np.concatenate([all_face[0:48], all_face[68:]])

def get_smplx_ids_op_lh():
    ind_root = [20]
    inds_big = [37, 38, 39, 66]
    inds_1 = [25, 26, 27, 67]
    inds_2 = [28, 29, 30, 68]
    inds_3 = [34, 35, 36, 69]
    inds_little = [31, 32, 33, 70]
    return np.asarray(
        ind_root + inds_big + inds_1 + inds_2 + inds_3 + inds_little, dtype=int)

def get_smplx_ids_op_rh():
    ind_root = [21]
    inds_big = [52, 53, 54, 71]
    inds_1 = [40, 41, 42, 72]
    inds_2 = [43, 44, 45, 73]
    inds_3 = [49, 50, 51, 74]
    inds_little = [46, 47, 48, 75]
    return np.asarray(
        ind_root + inds_big + inds_1 + inds_2 + inds_3 + inds_little, dtype=int)

def get_smplx_ids_op_body_25():
    body_mapping = np.array([
        55, 12,
        17, 19, 21,  # arm
        16, 18, 20,  # arm
        0,  # pelvis
        2, 5, 8,  # leg
        1, 4, 7,  # leg
        56, 57, 58, 59,  # head
        60, 61, 62, 63, 64, 65   # feet
    ], dtype=np.int32)
    return body_mapping

def get_kinect_smpl_joints():
    return np.asarray([[0, 0, 0],
                       [1, 3, 3],
                       [2, 6, 6],
                       # [3, 12, 12],
                       # [4, 13, 13],
                       [5, 16, 16],
                       [6, 18, 18],
                       [7, 20, 20],      #wrist left
                       # [11, 14, 14],
                       [12, 17, 17],
                       [13, 19, 19],
                       [14, 21, 21],              #wrist right
                       [18, 1, 1],
                       [19, 4, 4],
                       # [26, 15, 15], #head
                       # [20, 7],
                       # [21, 10],
                       [22, 2, 2],
                       [23, 5, 5],
                       # [27, 55, 15], #nose
                       [30, 56, 15], #right eye
                       [28, 57, 15], #left eye
                       [31, 58, 15], # right ear
                       [29, 59, 15] #left ear
                       # [24, 8],
                       # [25, 11]
                       ], dtype=int)

def get_kinect_smpl_joints_full():
    return np.asarray([[0, 0, 0],
                       [1, 3, 3],
                       [2, 6, 6],
                       [3, 12, 12],
                       [4, 13, 13],
                       [5, 16, 16],
                       [6, 18, 18],
                       [7, 20, 20],      #wrist left
                       [11, 14, 14],
                       [12, 17, 17],
                       [13, 19, 19],
                       [14, 21, 21],              #wrist right
                       [18, 1, 1],
                       [19, 4, 4],
                       [26, 15, 15], #head
                       [20, 7, 7],
                       [21, 10, 10],
                       [22, 2, 2],
                       [23, 5, 5],
                       [27, 55, 15], #nose
                       [30, 56, 15], #right eye
                       [28, 57, 15], #left eye
                       [31, 58, 15], # right ear
                       [29, 59, 15], #left ear
                       [24, 8, 8],
                       [25, 11, 11]
                       ], dtype=int)

def get_kinect_smplx_vert():
    return np.asarray(
        [[15, 7353, 21],
        [16, 7795, 21],
        [17, 8099, 21],
        [8, 4826, 20],
        [9, 5082, 20],
        [10, 5365, 20],
        [20, 5877, 7],
        [21, 5895, 10],
        [24, 8571, 8],
        [25, 8589, 11]], dtype=int)

def get_face_contour_vino():
    fc = np.stack([np.arange(18, 35), np.arange(127, 144)], axis=1)
    return fc

def get_vino_static_verts():
    # vino_vs = np.load('/home/alexander/data/vino_vs.npy')[0:18]
    # print(vino_vs)
    vino_vs = np.asarray([
        2485,
        2441,
        1243,
        1176,
        8970,
        8996,
        2747,
        1604,
        2821,
        9372,
        8985,
        8947,
        9215,
        335,
        9240,
        9364,
        16,
        694], dtype=int)
    return vino_vs

def get_face_smplx_landmarks():
    fc = get_face_contour_vino()

    fl = np.asarray([
        [12, 77],
        [14, 80],
        [15, 81],
        [17, 84],
        [5, 92],
        [6, 90],
        [7, 94],
        [4, 89],
        [10, 110],
        [11, 116],
        [8, 107],
        [9, 113]
    ], dtype=int)

    return np.concatenate([fc, fl], axis=0)