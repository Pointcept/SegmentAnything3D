import numpy as np
import torch
import open3d as o3d
import os
import copy
from PIL import Image
import json
# import clip
import pointops


SCANNET_COLOR_MAP_20 = {-1: (0., 0., 0.), 0: (174., 199., 232.), 1: (152., 223., 138.), 2: (31., 119., 180.), 3: (255., 187., 120.), 4: (188., 189., 34.), 5: (140., 86., 75.),
                        6: (255., 152., 150.), 7: (214., 39., 40.), 8: (197., 176., 213.), 9: (148., 103., 189.), 10: (196., 156., 148.), 11: (23., 190., 207.), 12: (247., 182., 210.), 
                        13: (219., 219., 141.), 14: (255., 127., 14.), 15: (158., 218., 229.), 16: (44., 160., 44.), 17: (112., 128., 144.), 18: (227., 119., 194.), 19: (82., 84., 163.)}

class Voxelize(object):
    def __init__(self,
                 voxel_size=0.05,
                 hash_type="fnv",
                 mode='train',
                 keys=("coord", "normal", "color", "label"),
                 return_discrete_coord=False,
                 return_min_coord=False):
        self.voxel_size = voxel_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_discrete_coord = return_discrete_coord
        self.return_min_coord = return_min_coord

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        discrete_coord = np.floor(data_dict["coord"] / np.array(self.voxel_size)).astype(int)
        min_coord = discrete_coord.min(0) * np.array(self.voxel_size)
        discrete_coord -= discrete_coord.min(0)
        key = self.hash(discrete_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == 'train':  # train mode
            # idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1])
            idx_unique = idx_sort[idx_select]
            if self.return_discrete_coord:
                data_dict["discrete_coord"] = discrete_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == 'test':  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                if self.return_discrete_coord:
                    data_part["discrete_coord"] = discrete_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


def overlap_percentage(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    area_intersection = np.sum(intersection)

    area_mask1 = np.sum(mask1)
    area_mask2 = np.sum(mask2)

    smaller_area = min(area_mask1, area_mask2)

    return area_intersection / smaller_area


def remove_samll_masks(masks, ratio=0.8):
    filtered_masks = []
    skip_masks = set()

    for i, mask1_dict in enumerate(masks):
        if i in skip_masks:
            continue

        should_keep = True
        for j, mask2_dict in enumerate(masks):
            if i == j or j in skip_masks:
                continue
            mask1 = mask1_dict["segmentation"]
            mask2 = mask2_dict["segmentation"]
            overlap = overlap_percentage(mask1, mask2)
            if overlap > ratio:
                if np.sum(mask1) < np.sum(mask2):
                    should_keep = False
                    break
                else:
                    skip_masks.add(j)

        if should_keep:
            filtered_masks.append(mask1)

    return filtered_masks


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


def save_point_cloud(coord, color=None, file_path="pc.ply", logger=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    coord = to_numpy(coord)
    if color is not None:
        color = to_numpy(color)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(np.ones_like(coord) if color is None else color)
    o3d.io.write_point_cloud(file_path, pcd)
    if logger is not None:
        logger.info(f"Save Point Cloud to: {file_path}")


def remove_small_group(group_ids, th):
    unique_elements, counts = np.unique(group_ids, return_counts=True)
    result = group_ids.copy()
    for i, count in enumerate(counts):
        if count < th:
            result[group_ids == unique_elements[i]] = -1
    
    return result


def pairwise_indices(length):
    return [[i, i + 1] if i + 1 < length else [i] for i in range(0, length, 2)]


def num_to_natural(group_ids):
    '''
    Change the group number to natural number arrangement
    '''
    if np.all(group_ids == -1):
        return group_ids
    array = copy.deepcopy(group_ids)
    unique_values = np.unique(array[array != -1])
    mapping = np.full(np.max(unique_values) + 2, -1)
    mapping[unique_values + 1] = np.arange(len(unique_values))
    array = mapping[array + 1]
    return array


def get_matching_indices(pcd0, pcd1, search_voxel_size, K=None):
    match_inds = []
    scene_coord = torch.tensor(np.asarray(pcd0.points)).cuda().contiguous().float()
    new_offset = torch.tensor(scene_coord.shape[0]).cuda().float()
    gen_coord = torch.tensor(np.asarray(pcd1.points)).cuda().contiguous().float()
    offset = torch.tensor(gen_coord.shape[0]).cuda().float()

    indices, dis = pointops.knn_query(1, gen_coord, offset, scene_coord, new_offset)
    indices = indices.cpu().numpy()[:,0]
    dis = dis.cpu().numpy()[:,0]

    for i in range(scene_coord.shape[0]):
        if dis[i] < search_voxel_size:
            match_inds.append([i,indices[i]])
    
    return match_inds

    


def visualize_3d(data_dict, text_feat_path, save_path):
    text_feat = torch.load(text_feat_path)
    group_logits = np.einsum('nc,mc->nm', data_dict["group_feat"], text_feat)
    group_labels = np.argmax(group_logits, axis=-1)
    labels = group_labels[data_dict["group"]]
    labels[data_dict["group"] == -1] = -1
    visualize_pcd(data_dict["coord"], data_dict["color"], labels, save_path)


def visualize_pcd(coord, pcd_color, labels, save_path):
    # alpha = 0.5
    label_color = np.array([SCANNET_COLOR_MAP_20[label] for label in labels])
    # overlay = (pcd_color * (1-alpha) + label_color * alpha).astype(np.uint8) / 255
    label_color = label_color / 255
    save_point_cloud(coord, label_color, save_path)


def visualize_2d(img_color, labels, img_size, save_path):
    import matplotlib.pyplot as plt
    # from skimage.segmentation import mark_boundaries
    # from skimage.color import label2rgb
    label_names = ["wall", "floor", "cabinet", "bed", "chair",
           "sofa", "table", "door", "window", "bookshelf",
           "picture", "counter", "desk", "curtain", "refridgerator",
           "shower curtain", "toilet", "sink", "bathtub", "other"]
    colors = np.array(list(SCANNET_COLOR_MAP_20.values()))[1:]
    segmentation_color = np.zeros((img_size[0], img_size[1], 3))
    for i, color in enumerate(colors):
        segmentation_color[labels == i] = color
    alpha = 1
    overlay = (img_color * (1-alpha) + segmentation_color * alpha).astype(np.uint8)
    fig, ax = plt.subplots()
    ax.imshow(overlay)
    patches = [plt.plot([], [], 's', color=np.array(color)/255, label=label)[0] for label, color in zip(label_names, colors)]
    plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=4, fontsize='small')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def visualize_partition(coord, group_id, save_path):
    group_id = group_id.reshape(-1)
    num_groups = group_id.max() + 1
    group_colors = np.random.rand(num_groups, 3)
    group_colors = np.vstack((group_colors, np.array([0,0,0])))
    color = group_colors[group_id]
    save_point_cloud(coord, color, save_path)


def delete_invalid_group(group, group_feat):
    indices = np.unique(group[group != -1])
    group = num_to_natural(group)
    group_feat = group_feat[indices]
    return group, group_feat