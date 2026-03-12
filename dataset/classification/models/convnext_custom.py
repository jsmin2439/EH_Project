import torch
import torch.nn as nn


# ---------------------------------------------------------
# DropPath (Stochastic Depth)
# ---------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor)
        return x / keep_prob * random_tensor


# ---------------------------------------------------------
# ConvNeXt Block
#   - NCHW 입력
#   - 블록 내부에서만 NHWC <-> NCHW 변환
# ---------------------------------------------------------
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path: float = 0.0):
        super().__init__()

        # Depthwise Conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # LayerNorm: channels_last(NHWC)용
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # MLP
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x  # (B, C, H, W)

        # DWConv: (B, C, H, W)
        x = self.dwconv(x)

        # (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)

        # LayerNorm over C
        x = self.norm(x)

        # MLP
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Skip + DropPath
        x = shortcut + self.drop_path(x)
        return x


# ---------------------------------------------------------
# ConvNeXt-Tiny
#  - dims=[96, 192, 384, 768]
#  - depths=[3, 3, 9, 3]
#  - downsample_layers 에는 LayerNorm 사용 X (Conv2d만)
#  - 마지막에만 LayerNorm + Linear head
# ---------------------------------------------------------
class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes: int = 10, drop_path_rate: float = 0.2):
        super().__init__()

        dims = [96, 192, 384, 768]
        depths = [3, 3, 9, 3]

        # --------- Downsample layers (no LayerNorm here) ---------
        self.downsample_layers = nn.ModuleList()
        # Stage 0: stem
        self.downsample_layers.append(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4)  # (B,3,H,W)->(B,96,H/4,W/4)
        )
        # Stage 1,2,3
        for i in range(3):
            self.downsample_layers.append(
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )

        # --------- Stages (ConvNeXt blocks) ---------
        self.stages = nn.ModuleList()
        # stochastic depth 비율 분배
        drop_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        idx = 0
        for i in range(4):
            blocks = []
            for _ in range(depths[i]):
                blocks.append(ConvNeXtBlock(dims[i], drop_path=drop_rates[idx]))
                idx += 1
            self.stages.append(nn.Sequential(*blocks))

        # 마지막 LayerNorm과 Head (채널 차원에 대해)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 3, H, W), 일반적으로 224x224 사용

        for i in range(4):
            # Downsample
            x = self.downsample_layers[i](x)   # (B, C_i, H_i, W_i)

            # Stage blocks
            x = self.stages[i](x)              # (B, C_i, H_i, W_i)

        # Global Average Pooling over H,W
        x = x.mean(dim=[2, 3])  # (B, C_last)

        # Final LayerNorm over channels
        x = self.norm(x)        # (B, C_last)

        # Classifier
        x = self.head(x)        # (B, num_classes)
        return x


# ---------------------------------------------------------
# Builder 함수 (experiment_v3.py 에서 사용하는 이름)
# ---------------------------------------------------------
def convnext_tiny_custom(num_classes: int = 10):
    return ConvNeXtTiny(num_classes=num_classes)