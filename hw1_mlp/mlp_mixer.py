import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange


# 用于mixing的mlp
class MlpBlock(nn.Module):
    '''
    dim: 输入输出维度, hidden_dim: 隐藏层维度
    '''
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # 两层全连接层+GELU，输入输出维度都是dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


# mixer layer，token mixing(cross-location) + channel mixing(per-location)
class MixerBlock(nn.Module):
    '''
    dim: patch的维度(channles), num_patch: patch的数量(patches)
    token_dim: token-mix的隐藏层维度, channel_dim: channel-mix的隐藏层维度
    '''
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()
        # token_mixing，在同一维度上各个patch/token间进行mixing
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),  
            Rearrange('b n d -> b d n'),  # 转置
            MlpBlock(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')  # 转置回来
        )
        # channel_mixing，在每个patch/token的各个维度上进行mixing
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            MlpBlock(dim, channel_dim, dropout),
        )

    def forward(self, x):
        # skip connection
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):
    '''
    in_channels: 输入图像的通道数, dim: patch的维度(channles), num_classes: 输出类别数
    patch_size: patch的大小, image_size: 图像大小, depth: mixer block的数量
    token_dim: token-mix的隐藏层维度, channel_dim: channel-mix的隐藏层维度
    '''
    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim, dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.num_patch =  (image_size // patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            # 将图像分割成patch，然后将每个patch展平
            nn.Conv2d(in_channels, dim, patch_size, patch_size),  # kernel_size=stride=patch_size
            Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim, dropout))

        self.layer_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)  # 输出类别数
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)  # global average pooling
        return self.mlp_head(x)



if __name__ == "__main__":
    img = torch.ones([32, 1, 28, 28])

    model = MLPMixer(in_channels=1, image_size=28, patch_size=7, num_classes=10,
                        dim=64, depth=3, token_dim=32, channel_dim=128, dropout=0.2)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(img)

    print("Shape of out :", out_img.shape)



