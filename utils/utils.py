import cv2
import torch
import numpy as np

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
