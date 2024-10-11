import torch
import random
import argparse

import numpy as np
import pandas as pd
from torch import nn
from loguru import logger
from datetime import datetime
from tabulate import tabulate
from torch.utils.data import DataLoader

from utils.dataset import build_dataset
from utils.utils import(
    collate_fn,
    geodesic_distance,
    recall_object,
    relative_pose_error_np,
    aggregate_metrics
)
from model.mpilope.mpilope import mpilope

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, nargs='*', type=int)
parser.add_argument('--num_sample', required=False, default=200, type=int)
parser.add_argument('--cnn_model', required=False, default='b', choices=['n', 't', 'b', 'l', 'h'], type=str)
parser.add_argument('--train_type', required=False, default=0, choices=[0, 1, 2], type=int)
parser.add_argument('--rot_mode', required=False, default='6d', choices=['6d', 'quat', 'matrix'], type=str)

parser.add_argument('--num_epochs', required=False, default=500, type=int)
parser.add_argument('--gpu_name', required=False, default='cpu', type=str)

args = parser.parse_args()

param_dataset_idx = args.dataset
param_num_sample = args.num_sample
param_cnn_model = args.cnn_model
param_train_type = args.train_type
param_rot_mode = args.rot_mode
param_num_epochs = args.num_epochs

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

# print(param_dataset_idx)

path = []
for idx in param_dataset_idx:
    path.append(paths[idx])
# print(path)

dataset = build_dataset(path)
# print(dataset.get_mkpts_info())

random.seed(20241003)
torch.manual_seed(20241003)
torch.cuda.manual_seed(20241003)

train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset=dataset,
    lengths=[train_size, val_size, test_size]
)

train_dataloader = DataLoader(train_dataset,
                              batch_size=128,
                              shuffle=False,
                              drop_last=True,
                              collate_fn=collate_fn(num_sample=param_num_sample))
val_dataloader = DataLoader(val_dataset,
                            batch_size=128,
                            shuffle=False,
                            drop_last=True,
                            collate_fn=collate_fn(num_sample=param_num_sample))
test_dataloader = DataLoader(test_dataset,
                             batch_size=128,
                             shuffle=False,
                             drop_last=True,
                             collate_fn=collate_fn(num_sample=param_num_sample))

model = mpilope(rotation_mode=param_rot_mode,
                pfe_num_sample=param_num_sample,
                ife_type=param_cnn_model)
model = model.train().to(device=device)

L2 = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4)

train_start_time = datetime.now()

train_mkpts0s = []
train_mkpts1s = []
train_img0s = []
train_img1s = []
train_rots = []
train_ts = []
for batch in train_dataloader:
    train_pose0 = []
    train_pose1 = []
    train_mkpts0 = []
    train_mkpts1 = []
    train_img0 = []
    train_img1 = []
    for data in batch:
        train_pose0.append(data['pose0'])
        train_pose1.append(data['pose1'])
        train_mkpts0.append(data['mkpts0'])
        train_mkpts1.append(data['mkpts1'])
        train_img0.append(data['img0'])
        train_img1.append(data['img1'])

    train_pose0 = np.stack(train_pose0, axis=0)
    train_pose1 = np.stack(train_pose1, axis=0)
    train_relative_pose = np.matmul(train_pose1, np.linalg.inv(train_pose0))
    train_relative_pose = torch.from_numpy(train_relative_pose).float()
    train_pose1 = torch.from_numpy(train_pose1).float()

    train_rot = train_relative_pose[:, :3, :3]
    train_t = train_pose1[:, :3, 3]
    train_rots.append(train_rot)
    train_ts.append(train_t)

    train_mkpts0 = torch.from_numpy(np.stack(train_mkpts0, axis=0)).float()
    train_mkpts1 = torch.from_numpy(np.stack(train_mkpts1, axis=0)).float()
    train_img0 = torch.from_numpy(np.stack(train_img0, axis=0)).float()
    train_img1 = torch.from_numpy(np.stack(train_img1, axis=0)).float()
    train_img0 = train_img0.permute(0, 3, 1, 2)
    train_img1 = train_img1.permute(0, 3, 1, 2)
    train_mkpts0s.append(train_mkpts0)
    train_mkpts1s.append(train_mkpts1)
    train_img0s.append(train_img0)
    train_img1s.append(train_img1)

val_mkpts0s = []
val_mkpts1s = []
val_img0s = []
val_img1s = []
val_rots = []
val_ts = []
for batch in val_dataloader:
    val_pose0 = []
    val_pose1 = []
    val_mkpts0 = []
    val_mkpts1 = []
    val_img0 = []
    val_img1 = []
    for data in batch:
        val_pose0.append(data['pose0'])
        val_pose1.append(data['pose1'])
        val_mkpts0.append(data['mkpts0'])
        val_mkpts1.append(data['mkpts1'])
        val_img0.append(data['img0'])
        val_img1.append(data['img1'])

    val_pose0 = np.stack(val_pose0, axis=0)
    val_pose1 = np.stack(val_pose1, axis=0)
    val_relative_pose = np.matmul(val_pose1, np.linalg.inv(val_pose0))
    val_relative_pose = torch.from_numpy(val_relative_pose).float()
    val_pose1 = torch.from_numpy(val_pose1).float()

    val_rot = val_relative_pose[:, :3, :3]
    val_t = val_pose1[:, :3, 3]
    val_rots.append(val_rot)
    val_ts.append(val_t)

    val_mkpts0 = torch.from_numpy(np.stack(val_mkpts0, axis=0)).float()
    val_mkpts1 = torch.from_numpy(np.stack(val_mkpts1, axis=0)).float()
    val_img0 = torch.from_numpy(np.stack(val_img0, axis=0)).float()
    val_img1 = torch.from_numpy(np.stack(val_img1, axis=0)).float()
    val_img0 = val_img0.permute(0, 3, 2, 1)
    val_img1 = val_img1.permute(0, 3, 2, 1)
    val_mkpts0s.append(val_mkpts0)
    val_mkpts1s.append(val_mkpts1)
    val_img0s.append(val_img0)
    val_img1s.append(val_img1)

patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

for epoch in range(1, param_num_epochs + 1):
    model.train()
    for i in range(len(train_dataloader)):
        pre_t, pre_rot = model(train_mkpts0s[i].to(device=device),
                               train_mkpts1s[i].to(device=device),
                               train_img0s[i].to(device=device),
                               train_img1s[i].to(device=device))

        t_loss = L2(train_ts[i].to(device=device), pre_t)
        rot_loss = geodesic_distance(train_rots[i].to(device=device), pre_rot)
        loss = t_loss + rot_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i in range(len(val_dataloader)):
            pre_t, pre_rot = model(val_mkpts0s[i].to(device=device),
                                   val_mkpts1s[i].to(device=device),
                                   val_img0s[i].to(device=device),
                                   val_img1s[i].to(device=device))
            t_loss = L2(val_ts[i].to(device=device), pre_t)
            rot_loss = geodesic_distance(val_rots[i].to(device=device), pre_rot)
            val_loss += t_loss.item() + rot_loss.item()

    val_loss /= len(val_dataloader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f'No improvement for {epochs_no_improve} epoch(s)')

    if epochs_no_improve >= patience:
        print(f'Early stopping after {epoch} epochs')
        early_stop = True
        break

    print(f'{param_dataset_idx}, epoch: {epoch}/{param_num_epochs}, val_loss: {val_loss:.4f}, r_loss: {rot_loss.item():.4f}, t_loss: {t_loss.item():.4f}, loss: {loss.item():.4f}')


train_end_time = datetime.now()

model.eval()

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

ycbv_dict = {
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine',
    10: 'ten',
}

# linemod
ape_data = []
benchvise_data = []
camera_data = []
can_data = []
cat_data = []
driller_data = []
duck_data = []
eggbox_data = []
glue_data = []
holepuncher_data = []
iron_data = []
lamp_data = []
phone_data = []
# onepose
aptamil_data = []
jzhg_data = []
minipuff_data = []
hlyormosiapie_data = []
brownhouse_data = []
oreo_data = []
mfmilkcake_data = []
diycookies_data = []
taipingcookies_data = []
tee_data = []
# onepose++
toyrobot_data = []
yellowduck_data = []
sheep_data = []
fakebanana_data = []
teabox_data = []
orange_data = []
greenteapot_data = []
lecreusetcup_data = []
insta_data = []
# ycbv
one_data = []
two_data = []
three_data = []
four_data = []
five_data = []
six_data = []
seven_data = []
eight_data = []
nine_data = []
ten_data = []

all_data = {
    # linemod
    'ape_data': ape_data,
    'benchvise_data': benchvise_data,
    'camera_data': camera_data,
    'can_data': can_data,
    'cat_data': cat_data,
    'driller_data': driller_data,
    'duck_data': duck_data,
    'eggbox_data': eggbox_data,
    'glue_data': glue_data,
    'holepuncher_data': holepuncher_data,
    'iron_data': iron_data,
    'lamp_data': lamp_data,
    'phone_data': phone_data,
    # onepose
    'aptamil_data': aptamil_data,
    'jzhg_data': jzhg_data,
    'minipuff_data': minipuff_data,
    'hlyormosiapie_data': hlyormosiapie_data,
    'brownhouse_data': brownhouse_data,
    'oreo_data': oreo_data,
    'mfmilkcake_data': mfmilkcake_data,
    'diycookies_data': diycookies_data,
    'taipingcookies_data': taipingcookies_data,
    'tee_data': tee_data,
    # onepose++
    'toyrobot_data': toyrobot_data,
    'yellowduck_data': yellowduck_data,
    'sheep_data': sheep_data,
    'fakebanana_data': fakebanana_data,
    'teabox_data': teabox_data,
    'orange_data': orange_data,
    'greenteapot_data': greenteapot_data,
    'lecreusetcup_data': lecreusetcup_data,
    'insta_data': insta_data,
    # ycbv
    'one_data': one_data,
    'two_data': two_data,
    'three_data': three_data,
    'four_data': four_data,
    'five_data': five_data,
    'six_data': six_data,
    'seven_data': seven_data,
    'eight_data': eight_data,
    'nine_data': nine_data,
    'ten_data': ten_data,
}

linemod_type = ['ape_data', 'benchvise_data', 'camera_data', 'can_data', 'cat_data', 'driller_data', 'duck_data', 'eggbox_data', 'glue_data', 'holepuncher_data', 'iron_data', 'lamp_data', 'phone_data']
onepose_type = ['aptamil_data', 'jzhg_data', 'minipuff_data', 'hlyormosiapie_data', 'brownhouse_data', 'oreo_data', 'mfmilkcake_data', 'diycookies_data', 'taipingcookies_data', 'tee_data']
oneposeplusplus_type = ['toyrobot_data', 'yellowduck_data', 'sheep_data', 'fakebanana_data', 'teabox_data', 'orange_data', 'greenteapot_data', 'lecreusetcup_data', 'insta_data']
ycbv_type = ['one_data', 'two_data', 'three_data', 'four_data', 'five_data', 'six_data', 'seven_data', 'eight_data', 'nine_data', 'ten_data']

for batch in test_dataloader:
    for data in batch:
        if 'lm' in data['name']:
            all_data[f"{id2name_dict[int(data['name'][2:])]}_data"].append(data)
        else:
            if data['name'] in ('12345678910'):
                all_data[f"{ycbv_dict[int(data['name'])]}_data"].append(data)
            else:
                all_data[f"{data['name']}_data"].append(data)

empty_keys = []
for key in all_data.keys():
    if len(all_data[key]) == 0:
        empty_keys.append(key)

for key in empty_keys:
    all_data.pop(key)

for key in all_data.keys():
    print(key, len(all_data[key]))

print(f'len(all_data) = {len(all_data)}')

res_table = []

for key in all_data.keys():
    if key in linemod_type:
        logger.info(f"LINEMOD: {key}")
    elif key in onepose_type:
        logger.info(f"ONEPOSE: {key}")
    elif key in oneposeplusplus_type:
        logger.info(f"ONEPOSE++: {key}")
    elif key in ycbv_type:
        logger.info(f"YCBV: {key}")
    metrics = dict()
    metrics.update({'R_errs':[], 't_errs':[], 'inliers':[], "identifiers":[]})
    recall_image, all_image = 0, 0
    for item in all_data[key]:
        all_image += 1
        K0 = item['K0']
        K1 = item['K1']
        pose0 = item['pose0']
        pose1 = item['pose1']
        pre_bbox = item['pre_bbox']
        gt_bbox = item['gt_bbox']
        mkpts0 = item['mkpts0']
        mkpts1 = item['mkpts1']
        pre_K = item['pre_K']
        img0 = item['img0']
        img1 = item['img1']
        name = item['name']
        pair_name = item['pair_name']
        # linemod
        if 'lm' in name:
            name = id2name_dict[int(name[2:])]
        # ycbv
        if name in '12345678910':
            name = ycbv_dict[int(name)]

        if name not in key:
            print(f'name: {name}, key: {key}')
            continue

        is_recalled = recall_object(pre_bbox, gt_bbox)

        recall_image = recall_image + int(is_recalled > 0.5)

        batch_mkpts0 = torch.from_numpy(mkpts0).unsqueeze(0).float().to(device)
        batch_mkpts1 = torch.from_numpy(mkpts1).unsqueeze(0).float().to(device)
        img0 = torch.from_numpy(img0).unsqueeze(0).float().to(device)
        img1 = torch.from_numpy(img1).unsqueeze(0).float().to(device)
        img0 = img0.permute(0, 3, 2, 1)
        img1 = img1.permute(0, 3, 2, 1)
        pre_t, pre_rot = model(batch_mkpts0, batch_mkpts1, img0, img1)
        pre_t = pre_t.squeeze(0).detach().cpu().numpy()
        pre_rot = pre_rot.squeeze(0).detach().cpu().numpy()

        relative_pose = np.matmul(pose1, np.linalg.inv(pose0))
        gt_pose = np.zeros_like(pose1)
        gt_pose[:3, :3] = relative_pose[:3, :3]
        gt_pose[:3, 3] = pose1[:3, 3]
        t_err, R_err = relative_pose_error_np(gt_pose, pre_rot, pre_t, ignore_gt_t_thr=0.0)

        metrics['R_errs'].append(R_err)
        metrics['t_errs'].append(t_err)
        metrics['identifiers'].append(pair_name)

    print(f"Acc: {recall_image}/{all_image}")
    val_metrics_4tb = aggregate_metrics(metrics, 5e-4)
    val_metrics_4tb["AP50"] = recall_image / all_image
    # logger.info('\n' + pprint.pformat(val_metrics_4tb))
    res_table.append([f"{name}"] + list(val_metrics_4tb.values()))

for i, key in enumerate(all_data.keys()):
    if i == 1:
        break
    if key in linemod_type:
        logger.info('LINEMOD')
    elif key in onepose_type:
        logger.info('ONEPOSE')
    elif key in oneposeplusplus_type:
        logger.info('ONEPOSE++')
    elif key in ycbv_type:
        logger.info('YCBV')

headers = ["Category"] + list(val_metrics_4tb.keys())
all_data_res = np.array(res_table)[:, 1:].astype(np.float32)
res_table.append(["Avg"] + all_data_res.mean(0).tolist())
# print(tabulate(res_table, headers=headers, tablefmt='fancy_grid'))
logger.info(f'R:ACC15 = {all_data_res.mean(0).tolist()[10]}')

df = pd.DataFrame(res_table, columns=headers)
df_rounded = df.round(6)
data_name = ''
for idx in param_dataset_idx:
    data_name += str(idx)
csv_name = f"{train_start_time.strftime('%Y%m%d')}-{data_name}-{(all_data_res.mean(0).tolist()[10]):.2f}"
df_rounded.to_csv(f'./res/csv/{csv_name}.csv', sep=',', index=False)

ckpts_name = f"{train_start_time.strftime('%Y%m%d')}-{data_name}-{(all_data_res.mean(0).tolist()[10]):.2f}"
torch.save(model, f'./res/pth/{ckpts_name}.pth')

test_end_time = datetime.now()


logger.info(f'train cost time: {((train_end_time - train_start_time).total_seconds() / 3600):.4f}h')
logger.info(f'test cost time: {((test_end_time - train_end_time).total_seconds() / 3600):.4f}h')
