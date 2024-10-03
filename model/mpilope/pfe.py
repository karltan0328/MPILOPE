import torch
from torch import nn

class embedding(nn.Module):
    def __init__(self,
                 in_channels:int,
                 N_freqs:int,
                 logscale:bool=True):
        super().__init__()
        self.in_channels = in_channels
        self.N_freqs = N_freqs
        self.logscale = logscale
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = self.in_channels * (len(self.funcs) * self.N_freqs + 1)

        if self.logscale:
            self.freq_bands = 2 ** torch.linspace(0, self.N_freqs - 1, self.N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x:torch.tensor):
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]
        return torch.cat(tensors=out, dim=-1)

class pfe(nn.Module):
    def __init__(self,
                 num_sample:int=300,
                 pts_size:int=2,
                 N_freqs:int=9,
                 attn_nhead:int=1):
        super().__init__()
        self.pts_size = pts_size
        self.N_freqs = N_freqs
        self.attn_nhead = attn_nhead
        self.embedding = embedding(in_channels=self.pts_size,
                                   N_freqs=self.N_freqs,
                                   logscale=False)
        self.transformer_mkpts = nn.Transformer(d_model=2 * self.pts_size * (2 * self.N_freqs + 1),
                                                nhead=self.attn_nhead)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=2 * self.pts_size * (2 * self.N_freqs + 1) * num_sample,
                      out_features=2 * (2 * self.N_freqs + 1) * num_sample),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=2 * (2 * self.N_freqs + 1) * num_sample,
                      out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )

    def forward(self,
                batch_mkpts0:torch.tensor,
                batch_mkpts1:torch.tensor):
        x = self.embedding(torch.cat((batch_mkpts0, batch_mkpts1), dim=-1))
        x = self.transformer_mkpts(x, x)
        x = x.reshape(x.shape[0], -1)
        x = self.mlp(x)
        x = x.reshape(x.shape[0], 2, -1)
        return x # (B, 2, 512)
