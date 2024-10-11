import cv2
import torch
import random
import numpy as np
from loguru import logger
from collections import OrderedDict

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_K_crop_resize(box, K_orig, resize_shape):
    """Update K (crop an image according to the box, and resize the cropped image to resize_shape)
    @param box: [x0, y0, x1, y1]
    @param K_orig: [3, 3] or [3, 4]
    @resize_shape: [h, w]
    """
    center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
    scale = np.array([box[2] - box[0], box[3] - box[1]])  # w, h

    resize_h, resize_w = resize_shape
    trans_crop = get_affine_transform(center, scale, 0, [resize_w, resize_h])
    trans_crop_homo = np.concatenate([trans_crop, np.array([[0, 0, 1]])], axis=0)

    if K_orig.shape == (3, 3):
        K_orig_homo = np.concatenate([K_orig, np.zeros((3, 1))], axis=-1)
    else:
        K_orig_homo = K_orig.copy()
    assert K_orig_homo.shape == (3, 4)

    K_crop_homo = trans_crop_homo @ K_orig_homo  # [3, 4]
    K_crop = K_crop_homo[:3, :3]

    return K_crop, K_crop_homo

def get_image_crop_resize(image, box, resize_shape):
    """Crop image according to the box, and resize the cropped image to resize_shape
    @param image: the image waiting to be cropped
    @param box: [x0, y0, x1, y1]
    @param resize_shape: [h, w]
    """
    center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
    scale = np.array([box[2] - box[0], box[3] - box[1]])

    resize_h, resize_w = resize_shape
    trans_crop = get_affine_transform(center, scale, 0, [resize_w, resize_h])
    image_crop = cv2.warpAffine(
        image, trans_crop, (resize_w, resize_h), flags=cv2.INTER_LINEAR
    )

    trans_crop_homo = np.concatenate([trans_crop, np.array([[0, 0, 1]])], axis=0)
    return image_crop, trans_crop_homo

def normalize_vector(v, return_mag=False):
    v_mag = torch.sqrt(v.pow(2).sum(-1))
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])).to(v.device))
    v_mag = v_mag.view(v.shape[0], 1).expand(v.shape[0], -1)
    v = v/v_mag
    if(return_mag==True):
        return v, v_mag[:, 0]
    else:
        return v

def qua2mat(quaternion):
    quat = normalize_vector(quaternion).contiguous()

    # (B, 1)
    qw = quat[..., 0].contiguous().view(quaternion.shape[0], 1)
    qx = quat[..., 1].contiguous().view(quaternion.shape[0], 1)
    qy = quat[..., 2].contiguous().view(quaternion.shape[0], 1)
    qz = quat[..., 3].contiguous().view(quaternion.shape[0], 1)

    # Unit quaternion rotation matrices computatation
    xx = qx * qx # (B, 1)
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    # (B, 3)
    row0 = torch.cat((1 - 2 * yy - 2 * zz,
                      2 * xy - 2 * zw,
                      2 * xz + 2 * yw), dim=-1)
    row1 = torch.cat((2 * xy + 2 * zw,
                      1 - 2 * xx - 2 * zz,
                      2 * yz - 2 * xw), dim=-1)
    row2 = torch.cat((2 * xz - 2 * yw,
                      2 * yz + 2 * xw,
                      1 - 2 * xx - 2 * yy), dim=-1)

    matrix = torch.cat((row0.view(quaternion.shape[0], 1, 3),
                        row1.view(quaternion.shape[0], 1, 3),
                        row2.view(quaternion.shape[0], 1, 3)), dim=-2)

    return matrix # (B, 3, 3)

def cross_product(u, v):
    i = u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1]
    j = u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2]
    k = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    out = torch.cat((i.view(u.shape[0], 1),
                     j.view(u.shape[0], 1),
                     k.view(u.shape[0], 1)), dim=-1)

    return out # (B, 3)

def o6d2mat(ortho6d):
    # ortho6d - (B, 6)
    x_raw = ortho6d[..., 0:3] # (B, 3)
    y_raw = ortho6d[..., 3:6] # (B, 3)

    x = normalize_vector(x_raw) # (B, 3)
    z = cross_product(x, y_raw) # (B, 3)
    z = normalize_vector(z) # (B, 3)
    y = cross_product(z, x) # (B, 3)

    x = x.view(x_raw.shape[0], 3, 1)
    y = y.view(x_raw.shape[0], 3, 1)
    z = z.view(x_raw.shape[0], 3, 1)
    matrix = torch.cat((x, y, z), dim=-1) # (B, 3, 3)
    return matrix

def project_points(pts, RT, K):
    pts = np.matmul(pts, RT[:, :3].transpose()) + RT[:, 3:].transpose()
    pts = np.matmul(pts, K.transpose())
    dpt = pts[:, 2]
    mask0 = (np.abs(dpt) < 1e-4) & (np.abs(dpt) > 0)
    if np.sum(mask0) > 0:
        dpt[mask0]=  1e-4
    mask1 = (np.abs(dpt) > -1e-4) & (np.abs(dpt) < 0)
    if np.sum(mask1) > 0:
        dpt[mask1] = -1e-4
    pts2d = pts[:, :2] / dpt[:, None]
    return pts2d, dpt

def recall_object(boxA, boxB, thresholded=0.5):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def collate_fn(num_sample):
    def collate(batch):
        after_process_batch = []
        for data in batch:
            item = {}
            for key in data.keys():
                if key == 'mkpts0' or key == 'mkpts1':
                    if data[key].shape[0] > num_sample:
                        sorted_idx = np.argsort(data['mconf'])[::-1]
                        clip_sorted_idx = sorted_idx[0:num_sample]
                        item[key] = data[key][clip_sorted_idx]
                    else:
                        item[key] = np.concatenate((
                            data[key],
                            np.zeros((num_sample - data[key].shape[0], 2), dtype=np.float32)), axis=0)
                else:
                    item[key] = data[key]
            after_process_batch.append(item)
        return after_process_batch
    return collate

def geodesic_distance(X, X1=None):
    assert X.dim() in [2, 3]

    if X.dim() == 2:
        X = X.expand(1, -1, -1)

    if X1 is None:
        X1 = torch.eye(3).expand(X.shape[0], 3, 3).to(X.device)

    m = X @ X1.permute(0, 2, 1)
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.clamp(cos, -1.0, 1.0) # handle numercial errors
    # cos = torch.min(cos, torch.ones(X.shape[0])).to(device)
    # cos = torch.max(cos, -torch.ones(X.shape[0])).to(device)
    return torch.acos(cos).mean()

def relative_pose_error_np(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)      # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))
    return t_err, R_err

def error_acc(type, errors, thresholds):
    accs = []
    for thr in thresholds:
        accs.append(np.sum(errors < thr) / errors.shape[0])
    res = {f'{type}:ACC{t:d}': auc for t, auc in zip(thresholds, accs)}
    res[f"{type}:medianErr"] = np.median(errors)
    res[f"{type}:meanErr"] = np.mean(errors)
    return res


def error_auc(type, errors, thresholds):
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))
    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)
    return {f'{type}:auc@{t:d}': auc for t, auc in zip(thresholds, aucs)}

def aggregate_metrics(metrics, epi_err_thr=5e-4):
    """
    Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """

    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    # pose auc
    angular_thresholds = [1, 5, 10, 15, 20, 25, 30]
    rotation_aucs = error_auc("R", np.array(metrics['R_errs']), angular_thresholds)
    translation_aucs = error_auc("t", np.array(metrics['t_errs']), angular_thresholds)

    rotation_accs = error_acc("R", np.array(metrics['R_errs']), angular_thresholds)
    translation_accs = error_acc("t", np.array(metrics['t_errs']), angular_thresholds)

    return {**rotation_aucs, **rotation_accs, **translation_aucs, **translation_accs}
