import torch
from torch import nn

from model.mpilope.pfe import pfe
from model.mpilope.ife import ife
from model.mpilope.fa import fa

from utils.utils import(
    qua2mat,
    o6d2mat
)

class mpilope(nn.Module):
    def __init__(self,
                 rotation_mode:str='6d',
                 pfe_num_sample:int=300,
                 pfe_pts_size:int=2,
                 pfe_N_freqs:int=9,
                 pfe_attn_nhead:int=1,
                 ife_type:str='l',
                 ife_input_size:int=224,
                 ife_nb_classes:int=1000,
                 ife_drop_path:float=0.0,
                 ife_layer_decay_type:str='single',
                 ife_head_init_scale:float=0.001,
                 fa_attn_d_model:int=512,
                 fa_attn_nhead:int=1):
        super().__init__()
        self.pfe = pfe(num_sample=pfe_num_sample,
                       pts_size=pfe_pts_size,
                       N_freqs=pfe_N_freqs,
                       attn_nhead=pfe_attn_nhead)
        self.ife = ife(type=ife_type,
                       input_size=ife_input_size,
                       nb_classes=ife_nb_classes,
                       drop_path=ife_drop_path,
                       layer_decay_type=ife_layer_decay_type,
                       head_init_scale=ife_head_init_scale)
        self.fa = fa(attn_d_model=fa_attn_d_model,
                     attn_nhead=fa_attn_nhead)
        self.translation_head = nn.Linear(in_features=32,
                                          out_features=3)
        self.rotation_mode = rotation_mode
        if self.rotation_mode == '6d':
            self.rotation_head = nn.Linear(in_features=32,
                                           out_features=6)
        elif self.rotation_mode == 'quat':
            self.rotation_head = nn.Linear(in_features=32,
                                           out_features=4)
        elif self.rotation_mode == 'matrix':
            self.rotation_head = nn.Linear(in_features=32,
                                           out_features=9)
        else:
            print('Please select a valid rotation type!')
            raise NotImplementedError

    def convert2matrix(self, x: torch.tensor):
        if self.mode == 'matrix':
            matrix = x.view(x.shape[0], 3, 3)
        elif self.mode == 'quat':
            matrix = qua2mat(x)
        elif self.mode == '6d':
            matrix = o6d2mat(x)
        return matrix

    def forward(self,
                batch_mkpts0: torch.tensor,
                batch_mkpts1: torch.tensor,
                batch_imgs0: torch.tensor,
                batch_imgs1: torch.tensor):
        fmkpts = self.pfe(batch_mkpts0, batch_mkpts1)
        fimgs = self.ife(batch_imgs0, batch_imgs1)
        f = self.fa(fmkpts, fimgs)

        trans = self.translation_head(f)
        rot = self.rotation_head(f)

        return trans, rot
