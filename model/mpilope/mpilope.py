import torch
import roma
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
                 pfe_num_sample:int=100,
                 pfe_pts_size:int=2,
                 pfe_N_freqs:int=9,
                 pfe_attn_nhead:int=4,
                 ife_type:str='l',
                 ife_input_size:int=224,
                 ife_nb_classes:int=1000,
                 ife_drop_path:float=0.0,
                 ife_layer_decay_type:str='single',
                 ife_head_init_scale:float=0.001,
                 fa_attn_d_model:int=512,
                 fa_attn_nhead:int=4):
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

        self.mlp_r = nn.Sequential(
            nn.Linear(in_features=2048,
                    out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024,
                    out_features=512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512,
                    out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256,
                    out_features=128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=128,
                    out_features=64),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=64,
                    out_features=32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=32,
                    out_features=32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
        )
        self.mlr_t = nn.Sequential(
            nn.Linear(in_features=2048,
                    out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024,
                    out_features=512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512,
                    out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256,
                    out_features=128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=128,
                    out_features=64),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=64,
                    out_features=32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=32,
                    out_features=32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
        )

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
        if self.rotation_mode == 'matrix':
            x_device = x.device
            matrix = x.reshape(x.shape[0], 3, 3).to('cpu')
            matrix = roma.special_procrustes(matrix).to(x_device)
        elif self.rotation_mode == 'quat':
            matrix = qua2mat(x)
        elif self.rotation_mode == '6d':
            matrix = o6d2mat(x)
        return matrix

    def forward(self,
                batch_mkpts0: torch.tensor,
                batch_mkpts1: torch.tensor,
                batch_imgs0: torch.tensor,
                batch_imgs1: torch.tensor):
        fmkpts = self.pfe(batch_mkpts0, batch_mkpts1)
        fimgs = self.ife(batch_imgs0, batch_imgs1)
        f = self.fa(fmkpts, fimgs) # (B, 2048)

        trans = self.translation_head(self.mlr_t(f))
        try:
            rot = self.convert2matrix(self.rotation_head(self.mlp_r(f)))
        except Exception as e:
            print(e)
            rot = torch.eye(3).unsqueeze(0).expand(trans.shape[0], 3, 3)

        return trans, rot
