import torch
from torch import nn
from utils.convnextv2_utils import (
    build_convnextv2,
)

class ife(nn.Module):
    def __init__(self,
                 type:str='l',
                 input_size:int=224,
                 nb_classes:int=1000,
                 drop_path:float=0.0,
                 layer_decay_type:str='single',
                 head_init_scale:float=0.001):
        super().__init__()
        self.input_size = input_size
        self.convnextv2 = build_convnextv2(type=type,
                                           input_size=input_size,
                                           nb_classes=nb_classes,
                                           drop_path=drop_path,
                                           layer_decay_type=layer_decay_type,
                                           head_init_scale=head_init_scale)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=1000,
                      out_features=512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )

    def forward(self,
                x1:torch.tensor,
                x2:torch.tensor):
        if x1.shape[2] != self.input_size or x1.shape[3] != self.input_size:
            assert False, 'x1 shape error!'
        if x2.shape[2] != self.input_size or x2.shape[3] != self.input_size:
            assert False, 'x2 shape error!'
        x1 = self.convnextv2(x1) # (B, nb_classes)
        x2 = self.convnextv2(x2) # (B, nb_classes)
        x1 = self.mlp(x1).unsqueeze(1) # (B, 1, 512)
        x2 = self.mlp(x2).unsqueeze(1) # (B, 1, 512)
        x = torch.cat((x1, x2), dim=1)
        return x # (B, 2, 512)
