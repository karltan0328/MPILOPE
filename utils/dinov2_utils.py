import numpy as np
from torchvision import transforms
from model.dinov2.utils.config import get_cfg

def get_dinov2_info(type:str='s') -> tuple:
    if type == 'g':
        dinov2_model_cfg = get_cfg('./model/dinov2/configs/eval/vits14_pretrain.yaml')
        dinov2_checkpoint = './weights/dinov2_vitg14_pretrain.pth'
    elif type == 'l':
        dinov2_model_cfg = get_cfg('./model/dinov2/configs/eval/vitl14_pretrain.yaml')
        dinov2_checkpoint = './weights/dinov2_vitl14_pretrain.pth'
    elif type == 'b':
        dinov2_model_cfg = get_cfg('./model/dinov2/configs/eval/vitb14_pretrain.yaml')
        dinov2_checkpoint = './weights/dinov2_vitb14_pretrain.pth'
    elif type == 's':
        dinov2_model_cfg = get_cfg('./model/dinov2/configs/eval/vits14_pretrain.yaml')
        dinov2_checkpoint = './weights/dinov2_vits14_pretrain.pth'
    else:
        raise NotImplementedError
    return dinov2_model_cfg, dinov2_checkpoint

def set_torch_image(
        image: np.ndarray,
        image_format: str = "RGB",
        center_crop = False
    ):
    # Transform the image to the form expected by the model
    if center_crop:
        prep = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop((196, 196)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        prep = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    input_tensor = prep(image)[None, ...]
    # input_tensor = input_tensor.cuda()
    return input_tensor

def get_cls_token_torch(model, input_tensor):
    # input_tensor = input_tensor.cuda()
    out = model(input_tensor, is_training=True)
    cls_token = out['x_norm_clstoken']
    # norm_cls_token = torch.nn.functional.normalize(cls_token)
    return cls_token
