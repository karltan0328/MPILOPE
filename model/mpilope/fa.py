import torch
from torch import nn

class fa(nn.Module):
    def __init__(self,
                 attn_d_model:int=512,
                 attn_nhead:int=1):
        super().__init__()
        self.attn_d_model = attn_d_model
        self.attn_nhead = attn_nhead
        self.mkpts_as_q = nn.Transformer(d_model=self.attn_d_model,
                                         nhead=self.attn_nhead)
        self.img_as_q = nn.Transformer(d_model=self.attn_d_model,
                                       nhead=self.attn_nhead)
        self.mlp = nn.Sequential(
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

    def forward(self,
                fmkpts:torch.tensor,
                fimgs:torch.tensor):
        qmkpts = self.mkpts_as_q(fimgs, fmkpts)
        qimgs = self.img_as_q(fmkpts, fimgs)

        x = torch.cat((qmkpts, qimgs), dim=-1)
        x = x.reshape(x.shape[0], -1)
        x = self.mlp(x)
        return x
