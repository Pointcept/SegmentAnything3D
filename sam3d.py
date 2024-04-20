"""
Main Script

Author: Yunhan Yang (yhyang.myron@gmail.com)
"""

import os
import cv2
import numpy as np
import open3d as o3d
import torch
import copy
import multiprocessing as mp
import pointops
import random
import argparse

from segment_anything import build_sam, SamAutomaticMaskGenerator
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from PIL import Image
from os.path import join
from util import *


def pcd_ensemble(org_path, new_path, data_path, vis_path):
    new_pcd = torch.load(new_path)
    new_pcd = num_to_natural(remove_small_group(new_pcd, 20))
    with open(org_path) as f:
        segments = json.load(f)
        org_pcd = np.array(segments['segIndices'])
    match_inds = [(i, i) for i in range(len(new_pcd))]
    new_group = cal_group(dict(group=new_pcd), dict(group=org_pcd), match_inds)
    print(new_group.shape)
    data = torch.load(data_path)
    visualize_partition(data["coord"], new_group, vis_path)


def get_sam(image, mask_generator):
    masks = mask_generator.generate(image)
    group_ids = np.full((image.shape[0], image.shape[1]), -1, dtype=int)
    num_masks = len(masks)
    group_counter = 0
    for i in reversed(range(num_masks)):
        # print(masks[i]["predicted_iou"])
        group_ids[masks[i]["segmentation"]] = group_counter
        group_counter += 1
    return group_ids


def get_pcd(scene_name, color_name, rgb_path, mask_generator, save_2dmask_path):
    intrinsic_path = join(rgb_path, scene_name, 'intrinsics', 'intrinsic_depth.txt')
    depth_intrinsic = np.loadtxt(intrinsic_path)

    pose = join(rgb_path, scene_name, 'pose', color_name[0:-4] + '.txt')
    depth = join(rgb_path, scene_name, 'depth', color_name[0:-4] + '.png')
    color = join(rgb_path, scene_name, 'color', color_name)

    depth_img = cv2.imread(depth, -1) # read 16bit grayscale image
    mask = (depth_img != 0)
    color_image = cv2.imread(color)
    color_image = cv2.resize(color_image, (640, 480))

    save_2dmask_path = join(save_2dmask_path, scene_name)
    if mask_generator is not None:
        group_ids = get_sam(color_image, mask_generator)
        if not os.path.exists(save_2dmask_path):
            os.makedirs(save_2dmask_path)
        img = Image.fromarray(num_to_natural(group_ids).astype(np.int16), mode='I;16')
        img.save(join(save_2dmask_path, color_name[0:-4] + '.png'))
    else:
        group_path = join(save_2dmask_path, color_name[0:-4] + '.png')
        img = Image.open(group_path)
        group_ids = np.array(img, dtype=np.int16)

    color_image = np.reshape(color_image[mask], [-1,3])
    group_ids = group_ids[mask]
    colors = np.zeros_like(color_image)
    colors[:,0] = color_image[:,2]
    colors[:,1] = color_image[:,1]
    colors[:,2] = color_image[:,0]

    pose = np.loadtxt(pose)
    
    depth_shift = 1000.0
    x,y = np.meshgrid(np.linspace(0,depth_img.shape[1]-1,depth_img.shape[1]), np.linspace(0,depth_img.shape[0]-1,depth_img.shape[0]))
    uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    uv_depth[:,:,0] = x
    uv_depth[:,:,1] = y
    uv_depth[:,:,2] = depth_img/depth_shift
    uv_depth = np.reshape(uv_depth, [-1,3])
    uv_depth = uv_depth[np.where(uv_depth[:,2]!=0),:].squeeze()
    
    intrinsic_inv = np.linalg.inv(depth_intrinsic)
    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    cx = depth_intrinsic[0,2]
    cy = depth_intrinsic[1,2]
    bx = depth_intrinsic[0,3]
    by = depth_intrinsic[1,3]
    n = uv_depth.shape[0]
    points = np.ones((n,4))
    X = (uv_depth[:,0]-cx)*uv_depth[:,2]/fx + bx
    Y = (uv_depth[:,1]-cy)*uv_depth[:,2]/fy + by
    points[:,0] = X
    points[:,1] = Y
    points[:,2] = uv_depth[:,2]
    points_world = np.dot(points, np.transpose(pose))
    group_ids = num_to_natural(group_ids)
    save_dict = dict(coord=points_world[:,:3], color=colors, group=group_ids)
    return save_dict


def make_open3d_point_cloud(input_dict, voxelize, th):
    input_dict["group"] = remove_small_group(input_dict["group"], th)
    # input_dict = voxelize(input_dict)

    xyz = input_dict["coord"]
    if np.isnan(xyz).any():
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def cal_group(input_dict, new_input_dict, match_inds, ratio=0.5):
    group_0 = input_dict["group"]
    group_1 = new_input_dict["group"]
    group_1[group_1 != -1] += group_0.max() + 1
    
    unique_groups, group_0_counts = np.unique(group_0, return_counts=True)
    group_0_counts = dict(zip(unique_groups, group_0_counts))
    unique_groups, group_1_counts = np.unique(group_1, return_counts=True)
    group_1_counts = dict(zip(unique_groups, group_1_counts))

    # Calculate the group number correspondence of overlapping points
    group_overlap = {}
    for i, j in match_inds:
        group_i = group_1[i]
        group_j = group_0[j]
        if group_i == -1:
            group_1[i] = group_0[j]
            continue
        if group_j == -1:
            continue
        if group_i not in group_overlap:
            group_overlap[group_i] = {}
        if group_j not in group_overlap[group_i]:
            group_overlap[group_i][group_j] = 0
        group_overlap[group_i][group_j] += 1

    # Update group information for point cloud 1
    for group_i, overlap_count in group_overlap.items():
        # for group_j, count in overlap_count.items():
        max_index = np.argmax(np.array(list(overlap_count.values())))
        group_j = list(overlap_count.keys())[max_index]
        count = list(overlap_count.values())[max_index]
        total_count = min(group_0_counts[group_j], group_1_counts[group_i]).astype(np.float32)
        # print(count / total_count)
        if count / total_count >= ratio:
            group_1[group_1 == group_i] = group_j
    return group_1


def cal_2_scenes(pcd_list, index, voxel_size, voxelize, th=50):
    if len(index) == 1:
        return(pcd_list[index[0]])
    # print(index, flush=True)
    input_dict_0 = pcd_list[index[0]]
    input_dict_1 = pcd_list[index[1]]
    pcd0 = make_open3d_point_cloud(input_dict_0, voxelize, th)
    pcd1 = make_open3d_point_cloud(input_dict_1, voxelize, th)
    if pcd0 == None:
        if pcd1 == None:
            return None
        else:
            return input_dict_1
    elif pcd1 == None:
        return input_dict_0

    # Cal Dul-overlap
    match_inds = get_matching_indices(pcd1, pcd0, 1.5 * voxel_size, 1)
    pcd1_new_group = cal_group(input_dict_0, input_dict_1, match_inds)
    # print(pcd1_new_group)

    match_inds = get_matching_indices(pcd1, pcd0, 1.5 * voxel_size, 1)
    input_dict_1["group"] = pcd1_new_group
    pcd0_new_group = cal_group(input_dict_1, input_dict_0, match_inds)
    # print(pcd0_new_group)

    pcd_new_group = np.concatenate((pcd0_new_group, pcd1_new_group), axis=0)
    pcd_new_group = num_to_natural(pcd_new_group)
    pcd_new_coord = np.concatenate((input_dict_0["coord"], input_dict_1["coord"]), axis=0)
    pcd_new_color = np.concatenate((input_dict_0["color"], input_dict_1["color"]), axis=0)
    pcd_dict = dict(coord=pcd_new_coord, color=pcd_new_color, group=pcd_new_group)

    pcd_dict = voxelize(pcd_dict)
    return pcd_dict


def seg_pcd(scene_name, rgb_path, data_path, save_path, mask_generator, voxel_size, voxelize, th, train_scenes, val_scenes, save_2dmask_path):
    print(scene_name, flush=True)
    if os.path.exists(join(save_path, scene_name + ".pth")):
        return
    color_names = sorted(os.listdir(join(rgb_path, scene_name, 'color')), key=lambda a: int(os.path.basename(a).split('.')[0]))
    pcd_list = []
    for color_name in color_names:
        print(color_name, flush=True)
        pcd_dict = get_pcd(scene_name, color_name, rgb_path, mask_generator, save_2dmask_path)
        if len(pcd_dict["coord"]) == 0:
            continue
        pcd_dict = voxelize(pcd_dict)
        pcd_list.append(pcd_dict)
    
    while len(pcd_list) != 1:
        print(len(pcd_list), flush=True)
        new_pcd_list = []
        for indice in pairwise_indices(len(pcd_list)):
            # print(indice)
            pcd_frame = cal_2_scenes(pcd_list, indice, voxel_size=voxel_size, voxelize=voxelize)
            if pcd_frame is not None:
                new_pcd_list.append(pcd_frame)
        pcd_list = new_pcd_list
    seg_dict = pcd_list[0]
    seg_dict["group"] = num_to_natural(remove_small_group(seg_dict["group"], th))

    if scene_name in train_scenes:
        scene_path = join(data_path, "train", scene_name + ".pth")
    elif scene_name in val_scenes:
        scene_path = join(data_path, "val", scene_name + ".pth")
    data_dict = torch.load(scene_path)
    scene_coord = torch.tensor(data_dict["coord"]).cuda().contiguous()
    new_offset = torch.tensor(scene_coord.shape[0]).cuda()
    gen_coord = torch.tensor(seg_dict["coord"]).cuda().contiguous().float()
    offset = torch.tensor(gen_coord.shape[0]).cuda()
    gen_group = seg_dict["group"]
    indices, dis = pointops.knn_query(1, gen_coord, offset, scene_coord, new_offset)
    indices = indices.cpu().numpy()
    group = gen_group[indices.reshape(-1)].astype(np.int16)
    mask_dis = dis.reshape(-1).cpu().numpy() > 0.6
    group[mask_dis] = -1
    group = group.astype(np.int16)
    torch.save(num_to_natural(group), join(save_path, scene_name + ".pth"))


def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Segment Anything on ScanNet.')
    parser.add_argument('--rgb_path', type=str, help='the path of rgb data')
    parser.add_argument('--data_path', type=str, default='', help='the path of pointcload data')
    parser.add_argument('--save_path', type=str, help='Where to save the pcd results')
    parser.add_argument('--save_2dmask_path', type=str, default='', help='Where to save 2D segmentation result from SAM')
    parser.add_argument('--sam_checkpoint_path', type=str, default='', help='the path of checkpoint for SAM')
    parser.add_argument('--scannetv2_train_path', type=str, default='scannet-preprocess/meta_data/scannetv2_train.txt', help='the path of scannetv2_train.txt')
    parser.add_argument('--scannetv2_val_path', type=str, default='scannet-preprocess/meta_data/scannetv2_val.txt', help='the path of scannetv2_val.txt')
    parser.add_argument('--img_size', default=[640,480])
    parser.add_argument('--voxel_size', default=0.05)
    parser.add_argument('--th', default=50, help='threshold of ignoring small groups to avoid noise pixel')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    print(args)
    with open(args.scannetv2_train_path) as train_file:
        train_scenes = train_file.read().splitlines()
    with open(args.scannetv2_val_path) as val_file:
        val_scenes = val_file.read().splitlines()
    mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=args.sam_checkpoint_path).to(device="cuda"))
    voxelize = Voxelize(voxel_size=args.voxel_size, mode="train", keys=("coord", "color", "group"))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    scene_names = sorted(os.listdir(args.rgb_path))
    for scene_name in scene_names:
        seg_pcd(scene_name, args.rgb_path, args.data_path, args.save_path, mask_generator, args.voxel_size, 
            voxelize, args.th, train_scenes, val_scenes, args.save_2dmask_path)
