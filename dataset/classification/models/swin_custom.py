import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# DropPath (Stochastic Depth)
# ---------------------------------------------------------v
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
# MLP
# ---------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------
# Window partition & reverse (B, H, W, C)
# ---------------------------------------------------------
def window_partition(x, window_size):
    # x: (B, H, W, C)
    B, H, W, C = x.shape
    x = x.view(
        B,
        H // window_size, window_size,
        W // window_size, window_size,
        C
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows  # (num_windows*B, window_size*window_size, C)


def window_reverse(windows, window_size, H, W):
    # windows: (num_windows*B, window_size*window_size, C)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B,
        H // window_size, W // window_size,
        window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x  # (B, H, W, C)


# ---------------------------------------------------------
# Window Attention with Relative Position Bias
# ---------------------------------------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # int or (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        Wh = Ww = window_size
        num_rel = (2 * Wh - 1) * (2 * Ww - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_rel, num_heads)
        )  # (2*Wh-1 * 2*Ww-1, nH)

        # pair-wise relative position index for each token inside the window
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= 2 * Ww - 1
        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        # x: (num_windows*B, N, C)
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B_, nH, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]   # each: (B_, nH, N, head_dim)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)     # (B_, nH, N, N)

        # relative position bias
        rel_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)  # (N, N, nH)
        rel_bias = rel_bias.permute(2, 0, 1).contiguous()  # (nH, N, N)
        attn = attn + rel_bias.unsqueeze(0)                # (B_, nH, N, N)

        if mask is not None:
            # mask: (num_windows, N, N)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # (B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------
# Swin Transformer Block
#   x: (B, L, C)
#   input_resolution: (H, W)
# ---------------------------------------------------------
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads,
                 window_size=7, shift_size=0, mlp_ratio=4.0,
                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        H, W = input_resolution
        if min(H, W) <= window_size:
            self.window_size = min(H, W)
            self.shift_size = 0

        assert 0 <= self.shift_size < self.window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, self.window_size, num_heads,
            qkv_bias=True, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)

        # attention mask for SW-MSA
        if self.shift_size > 0:
            self.register_buffer("attn_mask", self._create_attn_mask(H, W))
        else:
            self.attn_mask = None

    def _create_attn_mask(self, H, W):
        # generate attention mask for shift-window
        img_mask = torch.zeros((1, H, W, 1))  # (1, H, W, 1)
        cnt = 0
        ws = self.window_size
        ss = self.shift_size

        for h in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
            for w in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # (num_windows*1, ws*ws, 1)
        mask_windows = window_partition(img_mask, ws)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask  # (num_windows, N, N)

    def forward(self, x, H, W):
        # x: (B, L, C), L = H*W
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, ws*ws, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------
# Patch Merging (downsample)
#   x: (B, H*W, C) -> (B, H/2*W/2, 2C)
# ---------------------------------------------------------
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)

        if (H % 2) == 1:
            x = x[:, :-1, :, :]
            H -= 1
        if (W % 2) == 1:
            x = x[:, :, :-1, :]
            W -= 1

        x0 = x[:, 0::2, 0::2, :]  # B, H/2, W/2, C
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)  # B, H/2, W/2, 4C
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)                   # B, H/2*W/2, 2C
        return x, H // 2, W // 2


# ---------------------------------------------------------
# Basic Swin Layer (multiple blocks + optional downsample)
# ---------------------------------------------------------
class BasicLayer(nn.Module):
    def __init__(self, dim, depth, input_resolution,
                 num_heads, window_size=7,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0,
                 drop_path=None, downsample=True):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.input_resolution = input_resolution
        self.window_size = window_size

        if drop_path is None:
            drop_path = [0.0] * depth
        assert len(drop_path) == depth

        self.blocks = nn.ModuleList()
        H, W = input_resolution
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            blk = SwinTransformerBlock(
                dim=dim,
                input_resolution=(H, W),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i],
            )
            self.blocks.append(blk)

        self.downsample = PatchMerging(input_resolution, dim) if downsample else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)

        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)

        return x, H, W


# ---------------------------------------------------------
# Swin-Tiny (for 224x224 input, 10 classes)
# ---------------------------------------------------------
class SwinTiny(nn.Module):
    def __init__(self, num_classes=10,
                 img_size=224, patch_size=4,
                 embed_dim=96, depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=7,
                 mlp_ratio=4.0,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.2):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.window_size = window_size

        # patch embedding: (B, 3, 224,224) -> (B, C, H/4, W/4)
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        total_blocks = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()

        # stages
        self.layers = nn.ModuleList()
        H = W = img_size // patch_size
        curr = 0
        dim = embed_dim
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=dim,
                depth=depths[i_layer],
                input_resolution=(H, W),
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[curr:curr + depths[i_layer]],
                downsample=(i_layer < self.num_layers - 1),
            )
            self.layers.append(layer)
            curr += depths[i_layer]
            if i_layer < self.num_layers - 1:
                dim *= 2
                H //= 2
                W //= 2

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(self, x):
        # x: (B,3,224,224)
        x = self.patch_embed(x)  # (B, C, H', W') = (B, embed_dim, 56, 56)
        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', C)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)       # keep (B, L, C), update H,W

        x = self.norm(x)        # (B, L, C)
        x = x.mean(dim=1)       # global average over tokens
        x = self.head(x)        # (B, num_classes)
        return x


def swin_tiny_custom(num_classes=10):
    return SwinTiny(num_classes=num_classes)