import torch
from torch import nn

class fa(nn.Module):
    def __init__(self,
                 attn_d_model:int,
                 attn_nhead:int):
        super().__init__()
        self.attn_d_model = attn_d_model
        self.attn_nhead = attn_nhead
        self.mkpts_as_q = nn.Transformer(d_model=self.attn_d_model,
                                         nhead=self.attn_nhead)
        self.img_as_q = nn.Transformer(d_model=self.attn_d_model,
                                       nhead=self.attn_nhead)

        self.fusion = nn.Transformer(d_model=self.attn_d_model * 2,
                                     nhead=self.attn_nhead * 2)

    def forward(self,
                fmkpts:torch.tensor,
                fimgs:torch.tensor):
        qmkpts = self.mkpts_as_q(fimgs, fmkpts)
        qimgs = self.img_as_q(fmkpts, fimgs)

        x = torch.cat((qmkpts, qimgs), dim=-1)
        x = self.fusion(x, x)

        x = x.reshape(x.shape[0], -1)
        return x
