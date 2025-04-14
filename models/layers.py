
from einops.layers.torch import Rearrange
import torch.nn as nn

class WalkingEmbedding(nn.Module):
    def __init__(self, c_in, emb_dim, walking_step=3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.walking_step = walking_step
        if c_in % walking_step == 0:
            pd = 0
        else:
            pd = walking_step - c_in % walking_step
        self.conv = nn.Conv1d(1, emb_dim, kernel_size=walking_step, stride=walking_step, padding=pd, bias=False)

    def forward(self, x, it=0):
        try:
            x = x.unsqueeze(1)
            x = self.conv(x)
            if it:print(x.shape)
            x = x.transpose(1, 2)
        except Exception as e:
            raise e
        return x