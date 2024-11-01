import os
import cv2
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap

from model.dino_vit_features.extractor import (
    ViTExtractor,
)
from model.dino_vit_features.correspondences import (
    chunk_cosine_sim,
    draw_correspondences,
)

# params
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, nargs=1, type=str)
## extractor params
parser.add_argument('--extractor_type', required=False, default='dino_vits8', type=str)
parser.add_argument('--extractor_stride', required=False, default=4, type=int)
## find_correspondences params
parser.add_argument('--num_pairs', required=False, default=10, type=int)
parser.add_argument('--load_size', required=False, default=224, type=int)
parser.add_argument('--layer', required=False, default=9, type=int)
parser.add_argument('--facet', required=False, default='key', choices=['key', 'query', 'value', 'token'], type=str)
parser.add_argument('--bin', action='store_false') # default=True
parser.add_argument('--thresh', required=False, default=0.05, type=float)

parser.add_argument('--gpu_name', required=False, default='cpu', type=str)

args = parser.parse_args()

param_dataset_idx = args.dataset
## extractor params
param_extractor_type = args.extractor_type
param_extractor_stride = args.extractor_stride
## find_correspondences params
param_num_pairs = args.num_pairs
param_load_size = args.load_size
param_layer = args.layer
param_facet = args.facet
param_bin = args.bin
param_thresh = args.thresh

if args.gpu_name == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:' + args.gpu_name)

# datapath
LM_dataset_path = 'data/LM_dataset/'
LM_dataset_json_path = 'data/pairs/LINEMOD-test.json'
LM_dataset_points_path = 'data/LM_dataset-points/'
onepose_path = 'data/onepose/'
onepose_json_path = 'data/pairs/Onepose-HighTexture-test.json'
onepose_points_path = 'data/onepose-points/'
onepose_plusplus_path = 'data/onepose_plusplus/'
onepose_plusplus_json_path = 'data/pairs/OneposePlusPlus-test.json'
onepose_plusplus_points_path = 'data/onepose_plusplus-points/'
ycbv_path = 'data/ycbv/'
ycbv_json_path = 'data/pairs/YCB-VIDEO-test.json'
ycbv_points_path = 'data/ycbv-points'
paths = [
    ('linemod', LM_dataset_path, LM_dataset_json_path, LM_dataset_points_path),
    ('onepose', onepose_path, onepose_json_path, onepose_points_path),
    ('onepose_plusplus', onepose_plusplus_path, onepose_plusplus_json_path, onepose_plusplus_points_path),
    ('ycbv', ycbv_path, ycbv_json_path, ycbv_points_path),
]

# dataset
path = []
for idx in param_dataset_idx:
    # print(idx, type(idx))
    path.append(paths[int(idx)])

# model
extractor = ViTExtractor(model_type=param_extractor_type,
                         stride=param_extractor_stride,
                         device=device)

# pair_file path
if param_dataset_idx[0] == '0':
    id2name_dict = {
        1: "ape",
        2: "benchvise",
        4: "camera",
        5: "can",
        6: "cat",
        8: "driller",
        9: "duck",
        10: "eggbox",
        11: "glue",
        12: "holepuncher",
        13: "iron",
        14: "lamp",
        15: "phone",
    }
    ROOT_DIR = "data/LM_dataset/"
    with open("data/pairs/LINEMOD-test.json") as f:
        dir_list = json.load(f)
elif param_dataset_idx[0] == '1':
    ROOT_DIR = "data/onepose/"
    with open("data/pairs/Onepose-HighTexture-test.json") as f:
        dir_list = json.load(f)
elif param_dataset_idx[0] == '2':
    ROOT_DIR = "data/onepose_plusplus/"
    with open("data/pairs/OneposePlusPlus-test.json") as f:
        dir_list = json.load(f)
elif param_dataset_idx[0] == '3':
    ROOT_DIR = "data/ycbv/"
    with open("data/pairs/YCB-VIDEO-test.json") as f:
        dir_list = json.load(f)
else:
    pass

invalid_img = 0
all_img = 0

invalid_img_path_list = []

for label_idx, test_dict in enumerate(dir_list):
    if param_dataset_idx[0] == '0':
        logger.info(f"LINEMOD: {label_idx}/{len(dir_list) - 1}")
    elif param_dataset_idx[0] == '1':
        logger.info(f"ONEPOSE: {label_idx}/{len(dir_list) - 1}")
    elif param_dataset_idx[0] == '2':
        logger.info(f"ONEPOSE++: {label_idx}/{len(dir_list) - 1}")
    elif param_dataset_idx[0] == '3':
        logger.info(f"YCB-V: {label_idx}/{len(dir_list) - 1}")
    else:
        pass
    sample_data = dir_list[label_idx]["0"][0]
    label = sample_data.split("/")[0]
    name = label.split("-")[1]
    dir_name = os.path.dirname(sample_data)
    FULL_ROOT_DIR = os.path.join(ROOT_DIR, dir_name)
    for rotation_list in test_dict.values():
        # print(rotation_list[:10])
        if param_dataset_idx[0] == '3':
            rotation_list = rotation_list[::2]
        for pair_name in tqdm(rotation_list):
            all_img += 1

            if param_dataset_idx[0] == '0':
                points_file_path = os.path.join("data/LM_dataset-points/", pair_name.split("/")[0])
            elif param_dataset_idx[0] == '1':
                points_file_path = os.path.join("data/onepose-points/", pair_name.split("/")[0])
            elif param_dataset_idx[0] == '2':
                points_file_path = os.path.join("data/onepose_plusplus-points/", pair_name.split("/")[0])
            elif param_dataset_idx[0] == '3':
                points_file_path = os.path.join("data/ycbv-points/", pair_name.split("/")[0])
            else:
                pass

            crop_img0_loftr_path = os.path.join(points_file_path, "img0_loftr")
            crop_img1_loftr_path = os.path.join(points_file_path, "img1_loftr")
            # crop_img1_loftr_masked_path = os.path.join(points_file_path, "img1_masked_loftr")

            points_name = pair_name.split("/")[-1]

            crop_img0_loftr_path = os.path.join(crop_img0_loftr_path, f'{points_name}.png')
            crop_img1_loftr_path = os.path.join(crop_img1_loftr_path, f'{points_name}.png')
            # crop_img1_loftr_masked_path = os.path.join(crop_img1_loftr_masked_path, f'{points_name}.png')

            with torch.no_grad():

                # extracting descriptors for each image
                try:
                    image1_batch, image1_pil = extractor.preprocess(crop_img0_loftr_path, param_load_size)
                    image2_batch, image2_pil = extractor.preprocess(crop_img1_loftr_path, param_load_size)
                except:
                    invalid_img += 1
                    invalid_img_path_list.append((crop_img0_loftr_path, crop_img1_loftr_path))
                    continue

                descriptors1 = extractor.extract_descriptors(image1_batch.to(device), param_layer, param_facet, param_bin)
                descriptors2 = extractor.extract_descriptors(image2_batch.to(device), param_layer, param_facet, param_bin)
                num_patches1, load_size1 = extractor.num_patches, extractor.load_size
                num_patches2, load_size2 = extractor.num_patches, extractor.load_size

                # extracting saliency maps for each image
                saliency_map1 = extractor.extract_saliency_maps(image1_batch.to(device))[0]
                saliency_map2 = extractor.extract_saliency_maps(image2_batch.to(device))[0]
                # threshold saliency maps to get fg / bg masks
                fg_mask1 = saliency_map1 > param_thresh
                fg_mask2 = saliency_map2 > param_thresh

                # calculate similarity between image1 and image2 descriptors
                similarities = chunk_cosine_sim(descriptors1, descriptors2)

                # calculate best buddies
                image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=device)
                sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
                sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
                sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
                sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]
                bbs_mask = nn_2[nn_1] == image_idxs

                # remove best buddies where at least one descriptor is marked bg by saliency mask.
                fg_mask2_new_coors = nn_2[fg_mask2]
                fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=device)
                fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
                bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
                bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)

                # applying k-means to extract k high quality well distributed correspondence pairs
                bb_descs1 = descriptors1[0, 0, bbs_mask, :].cpu().numpy()
                bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].cpu().numpy()
                # apply k-means on a concatenation of a pairs descriptors.
                all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)
                n_clusters = min(param_num_pairs, len(all_keys_together))  # if not enough pairs, show all found pairs.
                length = np.sqrt((all_keys_together ** 2).sum(axis=1))[:, None]
                normalized = all_keys_together / length
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized)
                except:
                    print("Error: ", all_keys_together.shape, n_clusters)
                    print(crop_img1_loftr_path)
                bb_topk_sims = np.full((n_clusters), -np.inf)
                bb_indices_to_show = np.full((n_clusters), -np.inf)

                # rank pairs by their mean saliency value
                bb_cls_attn1 = saliency_map1[bbs_mask]
                bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
                bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
                ranks = bb_cls_attn

                for k in range(n_clusters):
                    for i, (label, rank) in enumerate(zip(kmeans.labels_, ranks)):
                        if rank > bb_topk_sims[label]:
                            bb_topk_sims[label] = rank
                            bb_indices_to_show[label] = i

                # get coordinates to show
                indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[bb_indices_to_show] # close bbs
                img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=device)[indices_to_show]
                img2_indices_to_show = nn_1[indices_to_show]
                # get features to save
                img1_features_to_show = descriptors1[0, 0, img1_indices_to_show, :] # 需要保存
                img2_features_to_show = descriptors2[0, 0, img2_indices_to_show, :] # 需要保存
                # coordinates in descriptor map's dimensions
                img1_y_to_show = (img1_indices_to_show / num_patches1[1])
                img1_x_to_show = (img1_indices_to_show % num_patches1[1])
                img2_y_to_show = (img2_indices_to_show / num_patches2[1])
                img2_x_to_show = (img2_indices_to_show % num_patches2[1])
                points1, points2 = [], []
                for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
                    x1_show = (int(x1) - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
                    y1_show = (int(y1) - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
                    x2_show = (int(x2) - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
                    y2_show = (int(y2) - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
                    points1.append((y1_show, x1_show))
                    points2.append((y2_show, x2_show))

                points1_to_show = torch.tensor(points1) # 需要保存
                points2_to_show = torch.tensor(points2) # 需要保存

            mkpts0_dino_path = os.path.join(points_file_path, "mkpts0_dino")                            # 需要保存
            mkpts1_dino_path = os.path.join(points_file_path, "mkpts1_dino")                            # 需要保存
            features0_dino_path = os.path.join(points_file_path, "features0_dino")                      # 需要保存
            features1_dino_path = os.path.join(points_file_path, "features1_dino")                      # 需要保存
            # img0_drawed_points_path = os.path.join(points_file_path, "img0_drawed_points")
            # img1_drawed_points_path = os.path.join(points_file_path, "img1_drawed_points")
            img_pair_drawed_points_path = os.path.join(points_file_path, "img_pair_dino_drawed_points") # 需要保存
            Path(mkpts0_dino_path).mkdir(parents=True, exist_ok=True)
            Path(mkpts1_dino_path).mkdir(parents=True, exist_ok=True)
            Path(features0_dino_path).mkdir(parents=True, exist_ok=True)
            Path(features1_dino_path).mkdir(parents=True, exist_ok=True)
            # Path(img0_drawed_points_path).mkdir(parents=True, exist_ok=True)
            # Path(img1_drawed_points_path).mkdir(parents=True, exist_ok=True)
            Path(img_pair_drawed_points_path).mkdir(parents=True, exist_ok=True)

            # save features and points
            torch.save(img1_features_to_show, os.path.join(features0_dino_path, f'{points_name}.pt'))
            torch.save(img2_features_to_show, os.path.join(features1_dino_path, f'{points_name}.pt'))
            torch.save(points1_to_show, os.path.join(mkpts0_dino_path, f'{points_name}.pt'))
            torch.save(points2_to_show, os.path.join(mkpts1_dino_path, f'{points_name}.pt'))

            img0_to_show = np.asarray(image1_pil)
            img1_to_show = np.asarray(image2_pil)
            res_img = np.hstack((img0_to_show, img1_to_show))

            if param_num_pairs > 15:
                cmap = plt.get_cmap('tab10')
            else:
                cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta",
                                       "indigo", "orange", "cyan", "darkgreen", "maroon",
                                       "black", "white", "chocolate", "gray", "blueviolet"])
            colors = np.array([cmap(x) for x in range(param_num_pairs)])
            radius1 = 100
            radius2 = 10

            plt.figure() # 需要保存
            plt.imshow(res_img)
            for point1, point2, color in zip(points1, points2, colors):
                y1, x1 = point1
                y2, x2 = point2
                plt.scatter(x1, y1, edgecolors='w', facecolors=color, s=radius1, alpha=0.5)
                plt.scatter(x1, y1, edgecolors='w', facecolors=color, s=radius2)
                plt.scatter(x2 + img0_to_show.shape[1], y2, edgecolors='w', facecolors=color, s=radius1, alpha=0.5)
                plt.scatter(x2 + img0_to_show.shape[1], y2, edgecolors='w', facecolors=color, s=radius2)
            plt.savefig(os.path.join(img_pair_drawed_points_path, f'{points_name}.png'))
            plt.close('all')

logger.info(f'invalid_img: {invalid_img}')
logger.info(f'all_img: {all_img}')
for invalid_img_path in invalid_img_path_list:
    print(invalid_img_path)
