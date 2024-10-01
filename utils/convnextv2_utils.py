import os
import math
import torch
from collections import OrderedDict
from timm.models.layers import trunc_normal_
import model.convnextv2.convnextv2 as convnextv2

def build_convnextv2(type:str='l',
                     input_size:int=224,
                     nb_classes:int=1000,
                     drop_path:float=0.0,
                     layer_decay_type:str='single',
                     head_init_scale:float=0.001):
    checkpoint_path = './weights/convnextv2_'
    if type == 'n':
        checkpoint_path += 'nano_22k_'
        model_name = 'convnextv2_nano'
    elif type == 't':
        checkpoint_path += 'tiny_22k_'
        model_name = 'convnextv2_tiny'
    elif type == 'b':
        checkpoint_path += 'base_22k_'
        model_name = 'convnextv2_base'
    elif type == 'l':
        checkpoint_path += 'large_22k_'
        model_name = 'convnextv2_large'
    elif type == 'h':
        checkpoint_path += 'huge_22k_'
        model_name = 'convnextv2_huge'
    else:
        raise NotImplementedError

    model = convnextv2.__dict__[model_name](
        num_classes=nb_classes,
        drop_path_rate=drop_path,
        head_init_scale=head_init_scale
    )

    if input_size == 224:
        checkpoint_path += '224_ema.pt'
    elif input_size == 384:
        checkpoint_path += '384_ema.pt'
    elif input_size == 512:
        checkpoint_path += '512_ema.pt'
    else:
        raise NotImplementedError

    if not os.path.exists(checkpoint_path):
        print(f'{checkpoint_path} is not exist')
        raise FileNotFoundError

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Load pre-trained checkpoint from: {checkpoint_path}")
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            # print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # remove decoder weights
    checkpoint_model_keys = list(checkpoint_model.keys())
    for k in checkpoint_model_keys:
        if 'decoder' in k or 'mask_token'in k or \
            'proj' in k or 'pred' in k:
            # print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    checkpoint_model = remap_checkpoint_keys(checkpoint_model)
    load_state_dict(model, checkpoint_model)

    # manually initialize fc layer
    # trunc_normal_(model.head.weight, std=2e-5)
    # torch.nn.init.constant_(model.head.bias, 0.)

    # freeze params
    for param in model.parameters():
        param.requires_grad = False

    return model

def remap_checkpoint_keys(ckpt):
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith('encoder'):
            k = '.'.join(k.split('.')[1:]) # remove encoder in the name
        if k.endswith('kernel'):
            k = '.'.join(k.split('.')[:-1]) # remove kernel in the name
            new_k = k + '.weight'
            if len(v.shape) == 3: # resahpe standard convolution
                kv, in_dim, out_dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = v.permute(2, 1, 0).\
                    reshape(out_dim, in_dim, ks, ks).transpose(3, 2)
            elif len(v.shape) == 2: # reshape depthwise convolution
                kv, dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = v.permute(1, 0).\
                    reshape(dim, 1, ks, ks).transpose(3, 2)
            continue
        elif 'ln' in k or 'linear' in k:
            k = k.split('.')
            k.pop(-2) # remove ln and linear in the name
            new_k = '.'.join(k)
        else:
            new_k = k
        new_ckpt[new_k] = v

    # reshape grn affine parameters and biases
    for k, v in new_ckpt.items():
        if k.endswith('bias') and len(v.shape) != 1:
            new_ckpt[k] = v.reshape(-1)
        elif 'grn' in k:
            new_ckpt[k] = v.unsqueeze(0).unsqueeze(1)
    return new_ckpt

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    # if len(missing_keys) > 0:
    #     print("Weights of {} not initialized from pretrained model: {}".format(
    #         model.__class__.__name__, missing_keys))
    # if len(unexpected_keys) > 0:
    #     print("Weights from pretrained model not used in {}: {}".format(
    #         model.__class__.__name__, unexpected_keys))
    # if len(ignore_missing_keys) > 0:
    #     print("Ignored weights of {} not initialized from pretrained model: {}".format(
    #         model.__class__.__name__, ignore_missing_keys))
    # if len(error_msgs) > 0:
    #     print('\n'.join(error_msgs))
