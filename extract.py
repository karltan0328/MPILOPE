import os
import cv2
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from statistics import median
import torch.nn.functional as F

from utils.utils import (
    get_K_crop_resize,
    get_image_crop_resize,
)

# sam module
from utils.sam_utils import (
    get_sam_info,
)
from model.segment_anything import (
    SamAutomaticMaskGenerator,
    sam_model_registry,
)
# dinov2 module
from utils.dinov2_utils import (
    get_dinov2_info,
    set_torch_image,
    get_cls_token_torch,
)
from model.dinov2.models import (
    build_model_from_cfg,
)
from model.dinov2.utils.utils import (
    load_pretrained_weights,
)
# loftr module
from model.loftr.loftr import (
    LoFTR,
    default_cfg,
)

# params
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, nargs=1, type=str)
parser.add_argument('--sam_model', required=False, default='h', choices=['h', 'l', 'b'], type=str)
parser.add_argument('--dinov2_model', required=False, default='s', choices=['g', 'l', 'b', 's'], type=str)
# parser.add_argument('--train_type', required=False, default='pts+img', choices=['pts+img', 'pts', 'img'], type=str)
# parser.add_argument('--rot_mode', required=False, default='6d', choices=['matrix', 'quat', '6d'], type=str)

# parser.add_argument('--num_epochs', required=False, default=500, type=int)
parser.add_argument('--gpu_name', required=False, default='cpu', type=str)

args = parser.parse_args()

param_dataset_idx = args.dataset
param_sam_model_type = args.sam_model
param_dinov2_model_type = args.dinov2_model
if args.gpu_name == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:' + args.gpu_name)

# datapath
LM_dataset_path = 'data/LM_dataset/'
LM_dataset_json_path = 'data/pairs/LINEMOD-test.json'
LM_dataset_points_path = 'data/LM_dataset-points/'
onepose_path = 'data/onepose/'
onepose_json_path = 'data/pairs/Onepose-test.json'
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

# print(param_dataset_idx)

# dataset
path = []
for idx in param_dataset_idx:
    # print(idx, type(idx))
    path.append(paths[int(idx)])

# model
## sam
sam_model_type, sam_checkpoint = get_sam_info(param_sam_model_type)
sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
sam = sam.to(device=device)
MASK_GEN = SamAutomaticMaskGenerator(sam)
# sam自带eval()
## dinov2
dinov2_model_cfg, dinov2_checkpoint = get_dinov2_info(param_dinov2_model_type)
dinov2, _, _ = build_model_from_cfg(dinov2_model_cfg, only_teacher=False)
load_pretrained_weights(dinov2, dinov2_checkpoint, checkpoint_key='student')
dinov2 = dinov2.eval().to(device=device)
## loftr
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load('weights/indoor_ot.ckpt')['state_dict'], strict=False)
matcher = matcher.eval().to(device=device)

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
    for rotation_key, rotation_list in zip(test_dict.keys(), test_dict.values()):
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

            pre_bbox_path = os.path.join(points_file_path, "pre_bbox")
            mkpts0_path = os.path.join(points_file_path, "mkpts0")
            mkpts1_path = os.path.join(points_file_path, "mkpts1")
            pre_K_path = os.path.join(points_file_path, "pre_K")
            crop_img0_path = os.path.join(points_file_path, "img0")
            crop_img1_path = os.path.join(points_file_path, "img1")
            Path(pre_bbox_path).mkdir(parents=True, exist_ok=True)
            Path(mkpts0_path).mkdir(parents=True, exist_ok=True)
            Path(mkpts1_path).mkdir(parents=True, exist_ok=True)
            Path(pre_K_path).mkdir(parents=True, exist_ok=True)
            Path(crop_img0_path).mkdir(parents=True, exist_ok=True)
            Path(crop_img1_path).mkdir(parents=True, exist_ok=True)
            # print("points_file_path =", points_file_path)
            # print("mkpts0_path =", mkpts0_path)
            # print("mkpts1_path =", mkpts1_path)
            # print("pre_K_path =", pre_K_path)
            points_name = pair_name.split("/")[-1]

            if os.path.exists(os.path.join(pre_bbox_path, f"{points_name}.txt")):
                continue


            base_name = os.path.basename(pair_name)
            if param_dataset_idx[0] == '3':
                idx0_name = base_name.split("png-")[0] + "png"
                idx1_name = base_name.split("png-")[1]
            else:
                idx0_name = base_name.split("-")[0]
                idx1_name = base_name.split("-")[1]

            if param_dataset_idx[0] == '0' or param_dataset_idx[0] == '3':
                image0_name = os.path.join(FULL_ROOT_DIR, idx0_name)
                image1_name = os.path.join(FULL_ROOT_DIR.replace("color", "color_full"), idx1_name)
                K0_path = image0_name.replace("color", "intrin_ba").replace("png", "txt")
                K1_path = image1_name.replace("color_full", "intrin").replace("png", "txt")
                # pose0_path = image0_name.replace("color", "poses_ba").replace("png", "txt")
                # pose1_path = image1_name.replace("color_full", "poses_ba").replace("png", "txt")
            elif param_dataset_idx[0] == '1' or param_dataset_idx[0] == '2':
                image0_name = os.path.join(FULL_ROOT_DIR, idx0_name)
                image1_name = os.path.join(FULL_ROOT_DIR.replace("color", "color"), idx1_name)
                K0_path = image0_name.replace("color", "intrin_ba").replace("png", "txt")
                K1_path = image1_name.replace("color", "intrin_ba").replace("png", "txt")
                # pose0_path = image0_name.replace("color", "poses_ba").replace("png", "txt")
                # pose1_path = image1_name.replace("color", "poses_ba").replace("png", "txt")

            K0 = np.loadtxt(K0_path, delimiter=' ')
            K1 = np.loadtxt(K1_path, delimiter=' ')
            image0 = cv2.imread(image0_name)
            # prompt image input to dinov2
            ref_torch_image = set_torch_image(image0, center_crop=True).to(device=device)
            ref_fea = get_cls_token_torch(dinov2, ref_torch_image)

            image1 = cv2.imread(image1_name)
            image_h, image_w, _ = image1.shape
            # target image input to sam
            masks = MASK_GEN.generate(image1)
            similarity_score, top_images = np.array([0, 0, 0], np.float32), [{}, {}, {}]
            compact_percent = 0.3

            # matching
            """
            mask['segmentation']:       the mask
            mask['area']:               the area of the mask in pixels
            mask['bbox']:               the boundary box of the mask in XYWH format
            mask['predicted_iou']:      the model's own prediction for the quality of the mask
            mask['point_coords']:       the sampled input point that generated this mask
            mask['stability_score']:    an additional measure of mask quality
            mask['crop_box']:           the crop of the image used to generate this mask in XYWH format
            """

            # 计算中位数 筛选一半的mask
            masks_scores = []
            for mask in masks:
                masks_scores.append(mask['predicted_iou'] + mask['stability_score'])
            median_masks_scores = median(masks_scores)

            for mask in masks:
                if mask['predicted_iou'] + mask['stability_score'] < median_masks_scores:
                    continue
                object_mask = np.expand_dims(mask["segmentation"], -1)
                x0, y0, w, h = mask["bbox"]
                x1, y1 = x0 + w, y0 + h
                x0 = int(x0 - w * compact_percent)
                y0 = int(y0 - h * compact_percent)
                x1 = int(x1 + w * compact_percent)
                y1 = int(y1 + h * compact_percent)
                box = np.array([x0, y0, x1, y1])
                resize_shape = np.array([y1 - y0, x1 - x0])
                K_crop, K_crop_homo = get_K_crop_resize(box, K1, resize_shape)
                image_crop, _ = get_image_crop_resize(image1, box, resize_shape)
                mask_crop, _ = get_image_crop_resize(np.float32(mask["segmentation"]) * 255, box, resize_shape)
                # object_mask, _ = get_image_crop_resize(object_mask, box, resize_shape)
                box_new = np.array([0, 0, x1 - x0, y1 - y0])
                if param_dataset_idx[0] == '0' or param_dataset_idx[0] == '3':
                    resize_shape = np.array([256, 256])
                elif param_dataset_idx[0] == '1' or param_dataset_idx[0] == '2':
                    resize_shape = np.array([512, 512])
                else:
                    pass
                K_crop, K_crop_homo = get_K_crop_resize(box_new, K_crop, resize_shape)
                image_crop, _ = get_image_crop_resize(image_crop, box_new, resize_shape)
                mask_crop, _ = get_image_crop_resize(mask_crop, box_new, resize_shape)
                crop_tensor = set_torch_image(image_crop, center_crop=True).to(device=device)
                with torch.no_grad():
                    fea = get_cls_token_torch(dinov2, crop_tensor)
                score = F.cosine_similarity(ref_fea, fea, dim=1, eps=1e-8)
                if (score.item() > similarity_score).any():
                    mask['crop_mask'] = mask_crop
                    mask["crop_image"] = image_crop
                    mask["K"] = K_crop
                    mask["bbox"] = box
                    min_idx = np.argmin(similarity_score)
                    similarity_score[min_idx] = score.item()
                    top_images[min_idx] = mask.copy()

            crop_img0 = image0
            img0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            img0 = torch.from_numpy(img0).float()[None] / 255.
            img0 = img0.unsqueeze(0).to(device=device)

            matching_score = [0 for _ in range(len(top_images))]
            for top_idx in range(len(top_images)):
                if 'crop_image' not in top_images[top_idx]:
                    continue
                crop_mask1 = top_images[top_idx]['crop_mask']
                crop_img1 = top_images[top_idx]["crop_image"]
                img1 = cv2.cvtColor(crop_img1, cv2.COLOR_BGR2GRAY)
                img1 = torch.from_numpy(img1).float()[None] / 255.
                img1 = img1.unsqueeze(0).to(device=device)
                batch = {'image0':img0, 'image1':img1}

                with torch.no_grad():
                    matcher(batch)
                    mkpts0 = batch['mkpts0_f'].cpu().numpy()
                    mkpts1 = batch['mkpts1_f'].cpu().numpy()
                    confidences = batch["mconf"].cpu().numpy()

                if mkpts0.shape[0] == 0 or mkpts1.shape[0] == 0:
                    continue

                mkpts0_masked = []
                mkpts1_masked = []

                # print(confidences)
                # print(mkpts0.shape)
                median_confidences = median(confidences)

                for idx, mkpt1 in enumerate(mkpts1):
                    ptx, pty = mkpt1
                    # print(ptx, pty)
                    if crop_mask1[int(ptx)][int(pty)] == 255.0 and confidences[idx] > median_confidences:
                        mkpts0_masked.append(mkpts0[idx])
                        mkpts1_masked.append(mkpts1[idx])
                    else:
                        pass

                mkpts0_masked = np.array(mkpts0_masked)
                mkpts1_masked = np.array(mkpts1_masked)

                conf_mask = np.where(confidences > 0.9)
                matching_score[top_idx] = conf_mask[0].shape[0]
                top_images[top_idx]["mkpts0"] = mkpts0
                top_images[top_idx]["mkpts1"] = mkpts1
                top_images[top_idx]["mkpts0_masked"] = mkpts0_masked
                top_images[top_idx]["mkpts1_masked"] = mkpts1_masked
                top_images[top_idx]["mconf"] = confidences
                top_images[top_idx]["crop_img0"] = crop_img0
                top_images[top_idx]["crop_img1"] = crop_img1


            max_match_idx = np.argmax(matching_score)
            if "crop_image" not in top_images[max_match_idx] or 'mkpts0' not in top_images[max_match_idx]:
                invalid_img += 1
                continue
            # print(matching_score, max_match_idx)
            pre_bbox = top_images[max_match_idx]["bbox"]                # 需要保存
            mkpts0 = top_images[max_match_idx]["mkpts0"]                # 需要保存
            mkpts1 = top_images[max_match_idx]["mkpts1"]                # 需要保存
            mkpts0_masked = top_images[max_match_idx]["mkpts0_masked"]  # 需要保存
            mkpts1_masked = top_images[max_match_idx]["mkpts1_masked"]  # 需要保存
            pre_K = top_images[max_match_idx]["K"]                      # 需要保存
            crop_img0 = top_images[max_match_idx]["crop_img0"]          # 需要保存
            crop_img1 = top_images[max_match_idx]["crop_img1"]          # 需要保存
            if (mkpts0_masked.shape[0] < 5 or mkpts1_masked.shape[0] < 5 or pre_K.shape[0] != 3):
                invalid_img += 1
                continue

            np.savetxt(os.path.join(pre_bbox_path, f"{points_name}.txt"), pre_bbox)
            np.savetxt(os.path.join(mkpts0_path, f"{points_name}.txt"), mkpts0_masked)
            np.savetxt(os.path.join(mkpts1_path, f"{points_name}.txt"), mkpts1_masked)
            np.savetxt(os.path.join(pre_K_path, f"{points_name}.txt"), pre_K)
            cv2.imwrite(os.path.join(crop_img0_path, f"{points_name}.png"), crop_img0)
            cv2.imwrite(os.path.join(crop_img1_path, f"{points_name}.png"), crop_img1)


logger.info(f'invalid_img: {invalid_img}')
logger.info(f'all_img: {all_img}')
