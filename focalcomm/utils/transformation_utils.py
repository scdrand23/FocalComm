# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib
import numpy as np

"""
Transformation utils
"""
def get_pairwise_transformation(base_data_dict, max_cav, proj_first):
    """
    Get pair-wise transformation matrix accross different agents.

    Parameters
    ----------
    base_data_dict : dict
        Key : cav id, item: transformation matrix to ego, lidar points.

    max_cav : int
        The maximum number of cav, default 5

    Return
    ------
    pairwise_t_matrix : np.array
        The pairwise transformation matrix across each cav.
        shape: (L, L, 4, 4), L is the max cav number in a scene
        pairwise_t_matrix[i, j] is Tji, i_to_j
    """
    pairwise_t_matrix = np.tile(np.eye(4), (max_cav, max_cav, 1, 1)) # (L, L, 4, 4)

    if proj_first:
        # if lidar projected to ego first, then the pairwise matrix
        # becomes identity
        # no need to warp again in fusion time.

        # pairwise_t_matrix[:, :] = np.identity(4)
        return pairwise_t_matrix
    else:
        t_list = []

        # save all transformation matrix in a list in order first.
        for cav_id, cav_content in base_data_dict.items():
            lidar_pose = cav_content['params']['lidar_pose']
            t_list.append(x_to_world(lidar_pose))  # Twx

        for i in range(len(t_list)):
            for j in range(len(t_list)):
                # identity matrix to self
                if i != j:
                    # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                    # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                    t_matrix = np.linalg.solve(t_list[j], t_list[i])  # Tjw*Twi = Tji
                    pairwise_t_matrix[i, j] = t_matrix

    return pairwise_t_matrix



def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def x1_to_x2(x1, x2):
    """
    Transformation matrix from x1 to x2.

    Parameters
    ----------
    x1 : list
        The pose of x1 under world coordinates.
    x2 : list
        The pose of x2 under world coordinates.

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.

    """
    x1_to_world = x_to_world(x1)
    x2_to_world = x_to_world(x2)
    world_to_x2 = np.linalg.inv(x2_to_world)

    transformation_matrix = np.dot(world_to_x2, x1_to_world)
    return transformation_matrix


def dist_to_continuous(p_dist, displacement_dist, res, downsample_rate):
    """
    Convert points discretized format to continuous space for BEV representation.
    Parameters
    ----------
    p_dist : numpy.array
        Points in discretized coorindates.

    displacement_dist : numpy.array
        Discretized coordinates of bottom left origin.

    res : float
        Discretization resolution.

    downsample_rate : int
        Dowmsamping rate.

    Returns
    -------
    p_continuous : numpy.array
        Points in continuous coorindates.

    """
    p_dist = np.copy(p_dist)
    p_dist = p_dist + displacement_dist
    p_continuous = p_dist * res * downsample_rate
    return p_continuous


def normalize_pairwise_tfm(pairwise_t_matrix, H, W, discrete_ratio, downsample_rate=1):
    """
    normalize the pairwise transformation matrix to affine matrix need by torch.nn.functional.affine_grid()

    pairwise_t_matrix: torch.tensor
        [B, L, L, 4, 4], B batchsize, L max_cav
    H: num.
        Feature map height
    W: num.
        Feature map width
    discrete_ratio * downsample_rate: num.
        One pixel on the feature map corresponds to the actual physical distance
    """

    pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
    pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
    pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
    pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (downsample_rate * discrete_ratio * W) * 2
    pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (downsample_rate * discrete_ratio * H) * 2

    normalized_affine_matrix = pairwise_t_matrix

    return normalized_affine_matrix


def tfm_to_pose(tfm):
    """
    Turn transformation matrix to [x, y, z, roll, yaw, pitch]
    We use degree format.
    tfm is pose in transformation format, and XYZ order, i.e. roll-pitch-yaw
    """
    yaw = np.degrees(np.arctan2(tfm[1,0], tfm[0,0]))
    roll = np.degrees(np.arctan2(-tfm[2,1], tfm[2,2]))
    pitch = np.degrees(np.arctan2(tfm[2,0], ((tfm[2,1]**2 + tfm[2,2]**2) ** 0.5)))
    
    x, y, z = tfm[:3,3]
    return [x, y, z, roll, yaw, pitch]


def rot_and_trans_to_trasnformation_matrix(json_file):
    matrix = np.empty([4,4])
    matrix[0:3, 0:3] = json_file["rotation"]
    matrix[:, 3][0:3] = np.array(json_file["translation"])[:, 0]
    matrix[3, 0:3] = 0
    matrix[3, 3] = 1
    return matrix


def muilt_coord(rotationA2B, translationA2B, rotationB2C, translationB2C):
    rotationA2B = np.array(rotationA2B).reshape(3, 3)
    rotationB2C = np.array(rotationB2C).reshape(3, 3)
    rotation = np.dot(rotationB2C, rotationA2B)
    translationA2B = np.array(translationA2B).reshape(3, 1)
    translationB2C = np.array(translationB2C).reshape(3, 1)
    translation = np.dot(rotationB2C, translationA2B) + translationB2C
    return rotation, translation


def veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel_json_file, novatel_to_world_json_file):
    matrix = np.empty([4,4])
    rotationA2B = lidar_to_novatel_json_file["transform"]["rotation"]
    translationA2B = lidar_to_novatel_json_file["transform"]["translation"]
    rotationB2C = novatel_to_world_json_file["rotation"]
    translationB2C = novatel_to_world_json_file["translation"]
    rotation, translation = muilt_coord(rotationA2B, translationA2B, rotationB2C, translationB2C)
    matrix[0:3, 0:3] = rotation
    matrix[:, 3][0:3] = np.array(translation)[:, 0]
    matrix[3, 0:3] = 0
    matrix[3, 3] = 1
    return matrix


def inf_side_rot_and_trans_to_trasnformation_matrix(json_file, system_error_offset):
    matrix = np.empty([4,4])
    matrix[0:3, 0:3] = json_file["rotation"]
    translation = np.array(json_file["translation"])
    translation[0][0] = translation[0][0] + system_error_offset["delta_x"]
    translation[1][0] = translation[1][0] + system_error_offset["delta_y"]
    matrix[:, 3][0:3] = translation[:, 0]
    matrix[3, 0:3] = 0
    matrix[3, 3] = 1
    return matrix