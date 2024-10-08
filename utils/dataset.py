import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from utils.utils import (
    project_points,
    recall_object,
)

class build_dataset(Dataset):
    def __init__(self, json_paths:list, input_size:int=224):
        self.data = []
        self.mkpts_max_len = 0
        self.mkpts_sum_len = 0
        self.all_img = 0
        self.invalid_img = 0
        for dataset_name, dataset_path, json_path, points_path in json_paths:
            assert os.path.exists(dataset_path), f'{dataset_path} does not exist'
            assert os.path.exists(json_path), f'{json_path} does not exist'
            with open(json_path) as f:
                dir_list = json.load(f)
            for label_idx, test_dict in enumerate(tqdm(dir_list)):
                sample_data = dir_list[label_idx]["0"][0]
                label = sample_data.split("/")[0]
                name = label.split("-")[1]
                dir_name = os.path.dirname(sample_data)
                FULL_ROOT_DIR = os.path.join(dataset_path, dir_name)
                for rotation_list in test_dict.values():
                    if dataset_name == 'ycbv':
                        rotation_list = rotation_list[::2]
                    for pair_name in rotation_list:
                        self.all_img += 1

                        base_name = os.path.basename(pair_name)
                        if dataset_name == 'ycbv':
                            idx0_name = base_name.split("png-")[0] + "png"
                            idx1_name = base_name.split("png-")[1]
                        else:
                            idx0_name = base_name.split("-")[0]
                            idx1_name = base_name.split("-")[1]

                        if dataset_name == 'linemod' or dataset_name == 'ycbv':
                            image0_name = os.path.join(FULL_ROOT_DIR, idx0_name)
                            image1_name = os.path.join(FULL_ROOT_DIR.replace("color", "color_full"), idx1_name)
                            K0_path = image0_name.replace("color", "intrin_ba").replace("png", "txt")
                            K1_path = image1_name.replace("color_full", "intrin").replace("png", "txt")
                            pose0_path = image0_name.replace("color", "poses_ba").replace("png", "txt")
                            pose1_path = image1_name.replace("color_full", "poses_ba").replace("png", "txt")
                        elif dataset_name == 'onepose' or dataset_name == 'onepose_plusplus':
                            image0_name = os.path.join(FULL_ROOT_DIR, idx0_name)
                            image1_name = os.path.join(FULL_ROOT_DIR.replace("color", "color"), idx1_name)
                            K0_path = image0_name.replace("color", "intrin_ba").replace("png", "txt")
                            K1_path = image1_name.replace("color", "intrin_ba").replace("png", "txt")
                            pose0_path = image0_name.replace("color", "poses_ba").replace("png", "txt")
                            pose1_path = image1_name.replace("color", "poses_ba").replace("png", "txt")

                        points_file_path = os.path.join(points_path, pair_name.split('/')[0])
                        pre_bbox_path = os.path.join(points_file_path, "pre_bbox")
                        mkpts0_path = os.path.join(points_file_path, "mkpts0")
                        mkpts1_path = os.path.join(points_file_path, "mkpts1")
                        pre_K_path = os.path.join(points_file_path, "pre_K")
                        mconf_path = os.path.join(points_file_path, 'mconf')
                        img0_path = os.path.join(points_file_path, "img0")
                        img1_path = os.path.join(points_file_path, "img1")

                        points_name = pair_name.split("/")[-1]
                        pre_bbox_path = os.path.join(pre_bbox_path, f'{points_name}.txt')
                        mkpts0_path = os.path.join(mkpts0_path, f'{points_name}.txt')
                        mkpts1_path = os.path.join(mkpts1_path, f'{points_name}.txt')
                        pre_K_path = os.path.join(pre_K_path, f'{points_name}.txt')
                        mconf_path = os.path.join(mconf_path, f'{points_name}.txt')
                        img0_path = os.path.join(img0_path, f'{points_name}.png')
                        img1_path = os.path.join(img1_path, f'{points_name}.png')

                        K0 = np.loadtxt(K0_path, delimiter=' ')
                        K1 = np.loadtxt(K1_path, delimiter=' ')
                        pose0 = np.loadtxt(pose0_path, delimiter=' ')
                        pose1 = np.loadtxt(pose1_path, delimiter=' ')
                        if pose0.shape[0] == 3:
                            pose0 = np.concatenate((pose0, np.array([[0, 0, 0, 1]])), axis=0)
                        if pose1.shape[0] == 3:
                            pose1 = np.concatenate((pose1, np.array([[0, 0, 0, 1]])), axis=0)

                        try:
                            mkpts0 = np.loadtxt(mkpts0_path, delimiter=' ')
                        except:
                            print(f'{mkpts0_path} does not exist')
                            self.invalid_img += 1
                            continue
                        mkpts1 = np.loadtxt(mkpts1_path, delimiter=' ')
                        pre_bbox = np.loadtxt(pre_bbox_path, delimiter=' ')
                        if dataset_name == 'ycbv':
                            gt_bbox_name = image0_name.replace("color", "bbox_2d").replace("png", "txt")
                            gt_bbox = np.loadtxt(gt_bbox_name, delimiter=' ')
                        else:
                            _3d_bbox = np.loadtxt(f'{os.path.join(dataset_path, label)}/box3d_corners.txt', delimiter=' ')
                            bbox_pts_3d, _ = project_points(_3d_bbox, pose1[:3, :4], K1)
                            bbox_pts_3d = bbox_pts_3d.astype(np.int32)
                            x0, y0, w, h = cv2.boundingRect(bbox_pts_3d)
                            x1, y1 = x0 + w, y0 + h
                            gt_bbox = np.array([x0, y0, x1, y1])
                        iou = recall_object(pre_bbox, gt_bbox)
                        pre_K = np.loadtxt(pre_K_path, delimiter=' ')
                        mconf = np.loadtxt(mconf_path, delimiter=' ')
                        img0 = cv2.imread(img0_path)
                        img1 = cv2.imread(img1_path)

                        img0 = cv2.resize(img0, (input_size, input_size))
                        img1 = cv2.resize(img1, (input_size, input_size))

                        if mkpts0.shape[0] == 0:
                            print(f'file {mkpts0_path} is empty')
                            continue
                        if mkpts0.shape[0] != mkpts1.shape[0]:
                            print("mkpts0.shape[0] != mkpts1.shape[0]")
                            continue
                        self.mkpts_max_len = max(self.mkpts_max_len, mkpts0.shape[0])
                        self.mkpts_sum_len += mkpts0.shape[0]
                        item = {}
                        item['K0'] = K0
                        item['K1'] = K1
                        item['pose0'] = pose0
                        item['pose1'] = pose1
                        item['pre_bbox'] = pre_bbox
                        item['gt_bbox'] = gt_bbox
                        item['mkpts0'] = mkpts0
                        item['mkpts1'] = mkpts1
                        item['pre_K'] = pre_K
                        item['name'] = name
                        item['pair_name'] = pair_name
                        item['img0'] = img0
                        item['img1'] = img1
                        item['iou'] = iou
                        item['mconf'] = mconf
                        self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_mkpts_info(self):
        return self.mkpts_max_len, self.mkpts_sum_len, self.invalid_img, self.all_img


if __name__ == '__main__':
    LM_dataset_path = 'data/LM_dataset/'
    LM_dataset_json_path = 'data/pairs/LINEMOD-test.json'
    LM_dataset_points_path = 'data/LM_dataset-points/'

    onepose_path = 'data/onepose/'
    onepose_json_path = 'data/pairs/Onepose-HighTexture-test.json'
    onepose_points_path = 'data/onepose-points/'

    oneposeplusplus_path = 'data/onepose_plusplus/'
    oneposeplusplus_json_path = 'data/pairs/OneposePlusPlus-test.json'
    oneposeplusplus_points_path = 'data/onepose_plusplus-points/'

    ycbv_path = 'data/ycbv/'
    ycbv_json_path = 'data/pairs/YCB-VIDEO-test.json'
    ycbv_points_path = 'data/ycbv-points'
    paths = [
        # ('linemod', LM_dataset_path, LM_dataset_json_path, LM_dataset_points_path),
        # ('onepose', onepose_path, onepose_json_path, onepose_points_path),
        ('onepose_plusplus', oneposeplusplus_path, oneposeplusplus_json_path, oneposeplusplus_points_path),
        # ('ycbv', ycbv_path, ycbv_json_path, ycbv_points_path),
    ]

    data = build_dataset(paths)
    # for key in data[0].keys():
    #     print(key)
    print(type(data[0]['img0']))
    print(f'data.all_img = {data.all_img}, data.invalid_img = {data.invalid_img}')
    print(f'data.mkpts_max_len = {data.mkpts_max_len}')
