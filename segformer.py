
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import gc
import random
import shutil
import math

import numpy as np
import rasterio
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F



BASE_SEED = 6

NUM_CLASSES = 8
IN_CHANNELS = 12

BATCH_SIZE = 32
NUM_WORKERS = 6
NUM_EPOCHS = 200
LR = 1e-3
PATIENCE = 20

USE_STRICT_DETERMINISM = True
USE_AMP = False

DATA_ROOT = "256x256_Dfc2020"

S1_DIR = os.path.join(DATA_ROOT, "s1_0")
S2_DIR = os.path.join(DATA_ROOT, "CLOUDY_TRAINING_S2")
LABEL_DIR = os.path.join(DATA_ROOT, "dfc_0")

CLEAR_S2_DIR = os.path.join(DATA_ROOT, "s2_0")

NDI_CLOUDY_DIR = os.path.join(DATA_ROOT, "ndi_cloud_TRAINING_S2")
NDI_CLEAN_DIR = os.path.join(DATA_ROOT, "ndi")

CLOUD_MASK_DIR = os.path.join(DATA_ROOT, "CLOUDY_TRAINING_S2", "cloud_masks")
SHADOW_MASK_DIR = os.path.join(DATA_ROOT, "CLOUDY_TRAINING_S2", "shadow_masks")

# Save paths.
CLOUD_CKPT_PATH = "lightsegformer_seed_clouds6_b32.pth"

RGB_REC_CKPT_PATH = "best_lightsegformer_rgbrec040_freezebackbone_seed6_b32.pth"
RGB_REC_OVERALL_PATH = "best_lightsegformer_rgbrec040_overall_freezebackbone_seed6_b32.pth"

NDI_REC_CKPT_PATH = "best_lightsegformer_ndirec040_freezebackbone_seed6_b32.pth"
NDI_REC_OVERALL_PATH = "best_lightsegformer_ndirec040_overall_freezebackbone_seed6_b32.pth"

RUN_CLOUD_ONLY = True
RUN_RGB_RECON = True
RUN_NDI_RECON = True
OVERWRITE_EXISTING = True

W_REC = 0.4
W_SEG = 1.0 - W_REC

S1_BANDS = [1, 2]
S2_BANDS = [2, 3, 4, 5, 6, 7, 8, 9, 12, 13]

RGB_BANDS = [4, 3, 2]
RGB_SCALE = 10000.0

MEAN = torch.tensor([
    -12.64386368,
    -19.35255814,
    438.37207031,
    614.05566406,
    588.40960693,
    942.84332275,
    1769.93164062,
    2049.55151367,
    2193.29199219,
    2235.55664062,
    1568.22680664,
    997.7324829,
], dtype=torch.float32).view(12, 1, 1)

STD = torch.tensor([
    5.1334939,
    5.5905056,
    607.02685547,
    603.29681396,
    684.56884766,
    738.43267822,
    1100.45605469,
    1275.80541992,
    1369.3717041,
    1356.54406738,
    1070.16125488,
    813.52764893,
], dtype=torch.float32).view(12, 1, 1)

CLASS_MAP = {
    1: 0,
    2: 1,
    4: 2,
    5: 3,
    6: 4,
    7: 5,
    9: 6,
    10: 7,
}




def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def set_seed_everywhere(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    if USE_STRICT_DETERMINISM:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def seed_worker_factory(seed: int):
    def _seed_worker(worker_id: int):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)
    return _seed_worker


def check_dirs(required):
    missing = []
    print("\nChecking paths...")
    for name, path in required.items():
        ok = os.path.isdir(path)
        print(f"{name}: {path} | exists={ok}")
        if not ok:
            missing.append(f"{name}: {path}")
    if missing:
        raise FileNotFoundError("Missing folders:\n" + "\n".join(missing))


def check_cloud_paths():
    check_dirs({
        "DATA_ROOT": DATA_ROOT,
        "S1_DIR": S1_DIR,
        "S2_DIR": S2_DIR,
        "LABEL_DIR": LABEL_DIR,
    })


def check_rgb_paths():
    check_dirs({
        "DATA_ROOT": DATA_ROOT,
        "S1_DIR": S1_DIR,
        "S2_DIR": S2_DIR,
        "CLEAR_S2_DIR": CLEAR_S2_DIR,
        "LABEL_DIR": LABEL_DIR,
        "CLOUD_MASK_DIR": CLOUD_MASK_DIR,
        "SHADOW_MASK_DIR": SHADOW_MASK_DIR,
    })
    if not os.path.isfile(CLOUD_CKPT_PATH):
        raise FileNotFoundError(f"Missing cloud checkpoint: {CLOUD_CKPT_PATH}")


def check_ndi_paths():
    check_dirs({
        "DATA_ROOT": DATA_ROOT,
        "S1_DIR": S1_DIR,
        "S2_DIR": S2_DIR,
        "LABEL_DIR": LABEL_DIR,
        "NDI_CLOUDY_DIR": NDI_CLOUDY_DIR,
        "NDI_CLEAN_DIR": NDI_CLEAN_DIR,
        "CLOUD_MASK_DIR": CLOUD_MASK_DIR,
        "SHADOW_MASK_DIR": SHADOW_MASK_DIR,
    })
    if not os.path.isfile(CLOUD_CKPT_PATH):
        raise FileNotFoundError(f"Missing cloud checkpoint: {CLOUD_CKPT_PATH}")



class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def CBR(in_ch, out_ch, k=3, p=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            padding=p,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,
            bias=True,
        )

    def forward(self, x, h, w):
        b, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x



class OverlapPatchEmbed(nn.Module):
    def __init__(
        self,
        in_chans,
        embed_dim,
        kernel_size,
        stride,
        padding,
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        x_tokens = x.flatten(2).transpose(1, 2)
        x_tokens = self.norm(x_tokens)
        return x_tokens, h, w


class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim,
                dim,
                kernel_size=sr_ratio,
                stride=sr_ratio,
            )
            self.norm = nn.LayerNorm(dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, h, w):
        b, n, c = x.shape

        q = self.q(x)
        q = q.reshape(b, n, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(b, c, h, w)
            x_ = self.sr(x_)
            x_ = x_.reshape(b, c, -1).transpose(1, 2)
            x_ = self.norm(x_)
        else:
            x_ = x

        kv = self.kv(x_)
        kv = kv.reshape(b, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class MixFFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features=None,
        drop=0.0,
    ):
        super().__init__()

        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = self.dwconv(x, h, w)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SegFormerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        sr_ratio=1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = EfficientSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.drop_path = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = MixFFN(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop,
        )

    def forward(self, x, h, w):
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.mlp(self.norm2(x), h, w))
        return x


class LightSegFormerB0Encoder(nn.Module):
    """
    MiT-B0-style light SegFormer encoder adapted for 12-channel S1+S2 input.

    Feature stages:
      f0: H/4,  32
      f1: H/8,  64
      f2: H/16, 160
      f3: H/32, 256
    """
    def __init__(
        self,
        in_channels=12,
        embed_dims=(32, 64, 160, 256),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(4, 4, 4, 4),
        depths=(2, 2, 2, 2),
        sr_ratios=(8, 4, 2, 1),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.embed_dims = list(embed_dims)

        self.patch_embed1 = OverlapPatchEmbed(
            in_chans=in_channels,
            embed_dim=embed_dims[0],
            kernel_size=7,
            stride=4,
            padding=3,
        )

        self.patch_embed2 = OverlapPatchEmbed(
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.patch_embed3 = OverlapPatchEmbed(
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.patch_embed4 = OverlapPatchEmbed(
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
            kernel_size=3,
            stride=2,
            padding=1,
        )

        total_blocks = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        cur = 0

        self.block1 = nn.ModuleList([
            SegFormerBlock(
                dim=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratios[0],
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                sr_ratio=sr_ratios[0],
            )
            for i in range(depths[0])
        ])
        cur += depths[0]

        self.block2 = nn.ModuleList([
            SegFormerBlock(
                dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1],
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                sr_ratio=sr_ratios[1],
            )
            for i in range(depths[1])
        ])
        cur += depths[1]

        self.block3 = nn.ModuleList([
            SegFormerBlock(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                sr_ratio=sr_ratios[2],
            )
            for i in range(depths[2])
        ])
        cur += depths[2]

        self.block4 = nn.ModuleList([
            SegFormerBlock(
                dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3],
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                sr_ratio=sr_ratios[3],
            )
            for i in range(depths[3])
        ])

        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    @staticmethod
    def tokens_to_map(x, h, w):
        b, n, c = x.shape
        return x.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x):
        outs = []

        # Stage 1: H/4
        x, h, w = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, h, w)
        x = self.norm1(x)
        f0 = self.tokens_to_map(x, h, w)
        outs.append(f0)

        # Stage 2: H/8
        x, h, w = self.patch_embed2(f0)
        for blk in self.block2:
            x = blk(x, h, w)
        x = self.norm2(x)
        f1 = self.tokens_to_map(x, h, w)
        outs.append(f1)

        # Stage 3: H/16
        x, h, w = self.patch_embed3(f1)
        for blk in self.block3:
            x = blk(x, h, w)
        x = self.norm3(x)
        f2 = self.tokens_to_map(x, h, w)
        outs.append(f2)

        # Stage 4: H/32
        x, h, w = self.patch_embed4(f2)
        for blk in self.block4:
            x = blk(x, h, w)
        x = self.norm4(x)
        f3 = self.tokens_to_map(x, h, w)
        outs.append(f3)

        return outs


class MLPProjection(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2).reshape(b, -1, h, w)
        return x


class LightSegFormerHead(nn.Module):
    """
    SegFormer-style all-MLP decoder.
    Returns a shared decoder feature at H/4 with 128 channels.
    """
    def __init__(
        self,
        in_channels=(32, 64, 160, 256),
        channels=128,
        num_classes=8,
        dropout_ratio=0.1,
    ):
        super().__init__()

        self.in_channels = list(in_channels)
        self.channels = channels

        self.linear_c1 = MLPProjection(in_channels[0], channels)
        self.linear_c2 = MLPProjection(in_channels[1], channels)
        self.linear_c3 = MLPProjection(in_channels[2], channels)
        self.linear_c4 = MLPProjection(in_channels[3], channels)

        self.linear_fuse = CBR(channels * 4, channels, k=1, p=0)

        self.dropout = nn.Dropout2d(dropout_ratio)

        self.cls_seg = nn.Conv2d(
            channels,
            num_classes,
            kernel_size=1,
        )

    def forward_features(self, features):
        c1, c2, c3, c4 = features

        target_size = c1.shape[-2:]

        _c4 = self.linear_c4(c4)
        _c4 = F.interpolate(
            _c4,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        _c3 = self.linear_c3(c3)
        _c3 = F.interpolate(
            _c3,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        _c2 = self.linear_c2(c2)
        _c2 = F.interpolate(
            _c2,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        _c1 = self.linear_c1(c1)

        x = self.linear_fuse(torch.cat([_c1, _c2, _c3, _c4], dim=1))
        x = self.dropout(x)

        return x

    def classify(self, feat):
        return self.cls_seg(feat)

    def forward(self, features):
        feat = self.forward_features(features)
        return self.classify(feat)


# ============================================================
# MODELS
# ============================================================

class LightSegFormerCloud(nn.Module):
    """
    Light SegFormer-B0-style cloud-only model for 12-channel input.
    """
    def __init__(
        self,
        in_channels=12,
        num_classes=8,
    ):
        super().__init__()

        self.backbone = LightSegFormerB0Encoder(
            in_channels=in_channels,
        )

        self.head = LightSegFormerHead(
            in_channels=(32, 64, 160, 256),
            channels=128,
            num_classes=num_classes,
            dropout_ratio=0.1,
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        feats = self.backbone(x)

        logits = self.head(feats)

        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits


class LightSegFormerRecon(nn.Module):

    def __init__(
        self,
        in_channels=12,
        num_classes=8,
        recon_name="RGB",
    ):
        super().__init__()

        self.recon_name = recon_name

        self.backbone = LightSegFormerB0Encoder(
            in_channels=in_channels,
        )

        self.head = LightSegFormerHead(
            in_channels=(32, 64, 160, 256),
            channels=128,
            num_classes=num_classes,
            dropout_ratio=0.1,
        )

        # shared decoder feature: H/4, 128.
        # early encoder feature f0: H/4, 32.
        self.recon_pre = nn.Sequential(
            CBR(128, 128, k=3, p=1),
            CBR(128, 128, k=3, p=1),
        )

        # 128 + 32 + 3 = 163.
        self.recon_inject = CBR(128 + 32 + 3, 128, k=1, p=0)

        self.recon_final = nn.Sequential(
            CBR(128, 64, k=3, p=1),
            CBR(64, 32, k=3, p=1),
        )

        self.recon_out = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh(),
        )

        self.seg_fuse = CBR(128 + 32, 256, k=1, p=0)
        self.seg_dropout = nn.Dropout2d(0.1)
        self.seg_out = nn.Conv2d(256, num_classes, kernel_size=1)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.backbone.eval()
        return self

    def forward(self, x, recon_cloudy, return_shapes=False):
        input_size = x.shape[-2:]

        feats = self.backbone(x)

        early_feat = feats[0]  # H/4, 32

        # Shared SegFormer decoder feature.
        head_feat = self.head.forward_features(feats)  # H/4, 128

        # 1) Shared decoder feature -> upsample to early-feature resolution.
        shared_up = F.interpolate(
            head_feat,
            size=early_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # 2) Conv 3x3, 128 two times.
        r128 = self.recon_pre(shared_up)

        # 3) Concatenate with early encoding feature and cloudy RGB/NDI.
        recon_cloudy_early = F.interpolate(
            recon_cloudy,
            size=early_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        recon_concat = torch.cat(
            [r128, early_feat, recon_cloudy_early],
            dim=1,
        )

        # 4) Conv 1x1, 128.
        r = self.recon_inject(recon_concat)

        # 5) Upsample to full input resolution.
        r_full = F.interpolate(
            r,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        # 6) 64 -> 32 -> 3 reconstruction output.
        r_feat32 = self.recon_final(r_full)
        recon_pred = self.recon_out(r_feat32)

        # Segmentation fusion.
        shared_full = F.interpolate(
            head_feat,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        seg_feat = self.seg_fuse(
            torch.cat([shared_full, r_feat32.detach()], dim=1)
        )

        seg_feat = self.seg_dropout(seg_feat)
        seg_logits = self.seg_out(seg_feat)

        if return_shapes:
            return seg_logits, recon_pred, {
                "input": tuple(x.shape),
                "f0_early_H4_32": tuple(feats[0].shape),
                "f1_H8_64": tuple(feats[1].shape),
                "f2_H16_160": tuple(feats[2].shape),
                "f3_H32_256": tuple(feats[3].shape),
                "head_feat_H4_128": tuple(head_feat.shape),
                "shared_up_H4_128": tuple(shared_up.shape),
                "r128_after_two_3x3": tuple(r128.shape),
                "recon_cloudy_H4_3": tuple(recon_cloudy_early.shape),
                "concat_128_plus_32_plus_3": tuple(recon_concat.shape),
                "r_after_1x1_128": tuple(r.shape),
                "r_feat32": tuple(r_feat32.shape),
                "recon_pred": tuple(recon_pred.shape),
                "shared_full": tuple(shared_full.shape),
                "seg_feat_256": tuple(seg_feat.shape),
                "seg_logits": tuple(seg_logits.shape),
            }

        return seg_logits, recon_pred


# ============================================================
# DATASETS
# ============================================================

class CloudOnlyDataset(Dataset):
    def __init__(self, s1_dir, s2_dir, label_dir):
        self.s1_dir = s1_dir
        self.s2_dir = s2_dir
        self.label_dir = label_dir
        self.data_pairs = self._load_data_pairs()
        if not self.data_pairs:
            raise RuntimeError("No valid cloud-only samples found.")

    def _load_data_pairs(self):
        pairs = []
        for file in sorted(os.listdir(self.s1_dir)):
            if not file.endswith(".tif"):
                continue

            s1 = os.path.join(self.s1_dir, file)
            s2 = os.path.join(self.s2_dir, file.replace("s1", "s2"))
            y = os.path.join(self.label_dir, file.replace("s1", "dfc"))

            if os.path.exists(s2) and os.path.exists(y):
                pairs.append((s1, s2, y))
        return pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        s1_path, s2_path, label_path = self.data_pairs[idx]

        with rasterio.open(s1_path) as src:
            s1 = src.read(S1_BANDS).astype(np.float32)

        with rasterio.open(s2_path) as src:
            s2 = src.read(S2_BANDS).astype(np.float32)

        x = torch.from_numpy(
            np.concatenate([s1, s2], axis=0)
        ).float()

        x = (x - MEAN) / STD

        with rasterio.open(label_path) as src:
            label = src.read(1).astype("int64")

        y = torch.from_numpy(label).long()

        for original, new in CLASS_MAP.items():
            y[y == original] = new

        return x, y


class RGBReconDataset(Dataset):
    def __init__(
        self,
        s1_dir,
        cloudy_s2_dir,
        clear_s2_dir,
        label_dir,
        cloud_mask_dir,
        shadow_mask_dir,
    ):
        self.s1_dir = s1_dir
        self.cloudy_s2_dir = cloudy_s2_dir
        self.clear_s2_dir = clear_s2_dir
        self.label_dir = label_dir
        self.cloud_mask_dir = cloud_mask_dir
        self.shadow_mask_dir = shadow_mask_dir
        self.data_pairs = self._load_data_pairs()
        if not self.data_pairs:
            raise RuntimeError("No valid RGB reconstruction samples found.")

    def _load_data_pairs(self):
        pairs = []
        for file in sorted(os.listdir(self.s1_dir)):
            if not file.endswith(".tif"):
                continue

            s2_name = file.replace("s1", "s2")
            label_name = file.replace("s1", "dfc")

            paths = [
                os.path.join(self.s1_dir, file),
                os.path.join(self.cloudy_s2_dir, s2_name),
                os.path.join(self.clear_s2_dir, s2_name),
                os.path.join(self.label_dir, label_name),
                os.path.join(self.cloud_mask_dir, s2_name),
                os.path.join(self.shadow_mask_dir, s2_name),
            ]

            if all(os.path.exists(p) for p in paths):
                pairs.append(tuple(paths))

        return pairs

    def __len__(self):
        return len(self.data_pairs)

    @staticmethod
    def rgb_to_tanh_range(rgb_np):
        rgb_np = rgb_np.astype(np.float32)
        rgb_np = np.clip(rgb_np / RGB_SCALE, 0.0, 1.0)
        return rgb_np * 2.0 - 1.0

    def __getitem__(self, idx):
        (
            s1_path,
            cloudy_s2_path,
            clear_s2_path,
            label_path,
            cloud_mask_path,
            shadow_mask_path,
        ) = self.data_pairs[idx]

        with rasterio.open(s1_path) as src:
            s1 = src.read(S1_BANDS).astype(np.float32)

        with rasterio.open(cloudy_s2_path) as src:
            cloudy_s2 = src.read(S2_BANDS).astype(np.float32)
            rgb_cloudy_np = src.read(RGB_BANDS).astype(np.float32)

        x = torch.from_numpy(
            np.concatenate([s1, cloudy_s2], axis=0)
        ).float()

        x = (x - MEAN) / STD

        with rasterio.open(label_path) as src:
            label = src.read(1).astype("int64")

        y = torch.from_numpy(label).long()

        for original, new in CLASS_MAP.items():
            y[y == original] = new

        with rasterio.open(clear_s2_path) as src:
            rgb_clear_np = src.read(RGB_BANDS).astype(np.float32)

        rgb_cloudy = torch.from_numpy(
            self.rgb_to_tanh_range(rgb_cloudy_np)
        ).float()

        rgb_clear = torch.from_numpy(
            self.rgb_to_tanh_range(rgb_clear_np)
        ).float()

        with rasterio.open(cloud_mask_path) as src:
            cloud = src.read(1) > 0

        with rasterio.open(shadow_mask_path) as src:
            shadow = src.read(1) > 0

        mask = torch.from_numpy(
            (cloud | shadow).astype(np.float32)
        ).unsqueeze(0)

        return x, y, rgb_cloudy, rgb_clear, mask


class NDIReconDataset(Dataset):
    def __init__(
        self,
        s1_dir,
        s2_dir,
        label_dir,
        ndi_cloudy_dir,
        ndi_clean_dir,
        cloud_mask_dir,
        shadow_mask_dir,
    ):
        self.s1_dir = s1_dir
        self.s2_dir = s2_dir
        self.label_dir = label_dir
        self.ndi_cloudy_dir = ndi_cloudy_dir
        self.ndi_clean_dir = ndi_clean_dir
        self.cloud_mask_dir = cloud_mask_dir
        self.shadow_mask_dir = shadow_mask_dir
        self.data_pairs = self._load_data_pairs()
        if not self.data_pairs:
            raise RuntimeError("No valid NDI reconstruction samples found.")

    def _load_data_pairs(self):
        pairs = []
        for file in sorted(os.listdir(self.s1_dir)):
            if not file.endswith(".tif"):
                continue

            paths = [
                os.path.join(self.s1_dir, file),
                os.path.join(self.s2_dir, file.replace("s1", "s2")),
                os.path.join(self.label_dir, file.replace("s1", "dfc")),
                os.path.join(self.ndi_cloudy_dir, file.replace("s1", "ndi")),
                os.path.join(self.ndi_clean_dir, file.replace("s1", "ndi")),
                os.path.join(self.cloud_mask_dir, file.replace("s1", "s2")),
                os.path.join(self.shadow_mask_dir, file.replace("s1", "s2")),
            ]

            if all(os.path.exists(p) for p in paths):
                pairs.append(tuple(paths))
        return pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        (
            s1_path,
            s2_path,
            label_path,
            ndi_cloudy_path,
            ndi_clean_path,
            cloud_mask_path,
            shadow_mask_path,
        ) = self.data_pairs[idx]

        with rasterio.open(s1_path) as src:
            s1 = src.read(S1_BANDS).astype(np.float32)

        with rasterio.open(s2_path) as src:
            s2 = src.read(S2_BANDS).astype(np.float32)

        x = torch.from_numpy(
            np.concatenate([s1, s2], axis=0)
        ).float()

        x = (x - MEAN) / STD

        with rasterio.open(label_path) as src:
            label = src.read(1).astype("int64")

        y = torch.from_numpy(label).long()

        for original, new in CLASS_MAP.items():
            y[y == original] = new

        with rasterio.open(ndi_cloudy_path) as src:
            ndi_cloudy = src.read([1, 2, 3]).astype(np.float32)

        with rasterio.open(ndi_clean_path) as src:
            ndi_clean = src.read([1, 2, 3]).astype(np.float32)

        ndi_cloudy = torch.from_numpy(ndi_cloudy).float()
        ndi_clean = torch.from_numpy(ndi_clean).float()

        with rasterio.open(cloud_mask_path) as src:
            cloud = src.read(1) > 0

        with rasterio.open(shadow_mask_path) as src:
            shadow = src.read(1) > 0

        mask = torch.from_numpy(
            (cloud | shadow).astype(np.float32)
        ).unsqueeze(0)

        return x, y, ndi_cloudy, ndi_clean, mask


# ============================================================
# LOSSES, METRICS, LOADERS
# ============================================================

def calculate_iou(logits, labels, num_classes=8):
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        ious = []

        for cls in range(num_classes):
            pred_cls = preds == cls
            true_cls = labels == cls

            intersection = (pred_cls & true_cls).float().sum()
            union = (pred_cls | true_cls).float().sum()

            if union > 0:
                ious.append(intersection / (union + 1e-6))

        if len(ious) == 0:
            return torch.tensor(0.0, device=logits.device)

        return torch.mean(torch.stack(ious))


def masked_l1(pred, target, mask):
    m = mask.expand_as(pred)
    denom = m.sum()

    if denom.item() == 0:
        return pred.new_tensor(0.0)

    return (torch.abs(pred - target) * m).sum() / (denom + 1e-6)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_split_loaders(dataset, seed):
    split_gen = torch.Generator().manual_seed(seed)

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=split_gen,
    )

    worker_init_fn = seed_worker_factory(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    print(f"Total samples: {len(dataset)}")
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    return train_loader, val_loader


def make_cloud_loaders(seed=BASE_SEED):
    dataset = CloudOnlyDataset(S1_DIR, S2_DIR, LABEL_DIR)
    return make_split_loaders(dataset, seed)


def make_rgb_loaders(seed=BASE_SEED):
    dataset = RGBReconDataset(
        S1_DIR,
        S2_DIR,
        CLEAR_S2_DIR,
        LABEL_DIR,
        CLOUD_MASK_DIR,
        SHADOW_MASK_DIR,
    )
    return make_split_loaders(dataset, seed)


def make_ndi_loaders(seed=BASE_SEED):
    dataset = NDIReconDataset(
        S1_DIR,
        S2_DIR,
        LABEL_DIR,
        NDI_CLOUDY_DIR,
        NDI_CLEAN_DIR,
        CLOUD_MASK_DIR,
        SHADOW_MASK_DIR,
    )
    return make_split_loaders(dataset, seed)


# ============================================================
# CHECKPOINT INITIALIZATION
# ============================================================

def load_cloud_weights_as_initialization(model, ckpt_path):
    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
    )

    if isinstance(ckpt, dict):
        ckpt = ckpt.get(
            "model_state_dict",
            ckpt.get("state_dict", ckpt),
        )

    ckpt = {
        k.replace("module.", ""): v
        for k, v in ckpt.items()
    }

    current = model.state_dict()
    loadable = {}

    for k, v in ckpt.items():
        if k in current and current[k].shape == v.shape:
            loadable[k] = v

    missing, unexpected = model.load_state_dict(
        loadable,
        strict=False,
    )

    print("\nLoaded compatible cloud-only weights.")
    print("Checkpoint:", ckpt_path)
    print("Exact tensors loaded:", len(loadable))
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    return model


def build_rgb_model(device):
    model = LightSegFormerRecon(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        recon_name="RGB",
    )

    model = load_cloud_weights_as_initialization(
        model,
        CLOUD_CKPT_PATH,
    )

    model.freeze_backbone()
    model.to(device)

    total, trainable = count_parameters(model)

    print(
        f"\nLight SegFormer RGB model parameters: "
        f"total={total:,}, trainable={trainable:,}, frozen_backbone=True"
    )

    return model


def build_ndi_model(device):
    model = LightSegFormerRecon(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        recon_name="NDI",
    )

    model = load_cloud_weights_as_initialization(
        model,
        CLOUD_CKPT_PATH,
    )

    model.freeze_backbone()
    model.to(device)

    total, trainable = count_parameters(model)

    print(
        f"\nLight SegFormer NDI model parameters: "
        f"total={total:,}, trainable={trainable:,}, frozen_backbone=True"
    )

    return model


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_cloud_only(
    model,
    train_loader,
    val_loader,
    device,
    save_path,
):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    amp_enabled = bool(USE_AMP and device.type == "cuda")

    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled,
    )

    model.to(device)

    total, trainable = count_parameters(model)

    print(f"\nLight SegFormer cloud-only parameters: total={total:,}, trainable={trainable:,}")

    hist = {
        "train_loss": [],
        "val_loss": [],
        "train_iou": [],
        "val_iou": [],
    }

    best_val_iou = -1.0
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(NUM_EPOCHS):
        model.train()

        tr_loss = 0.0
        tr_iou = 0.0
        tr_batches = 0

        for x, y in tqdm(
            train_loader,
            desc=f"LightSegFormer cloud epoch {epoch + 1}/{NUM_EPOCHS} [train]",
        ):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                with torch.cuda.amp.autocast(enabled=True):
                    logits = model(x)
                    loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            tr_loss += float(loss.detach().cpu())
            tr_iou += calculate_iou(logits.detach(), y).item()
            tr_batches += 1

            del x, y, logits, loss

        hist["train_loss"].append(tr_loss / max(1, tr_batches))
        hist["train_iou"].append(tr_iou / max(1, tr_batches))

        free_gpu_memory()

        model.eval()

        val_loss = 0.0
        val_iou_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for x, y in tqdm(
                val_loader,
                desc=f"LightSegFormer cloud epoch {epoch + 1}/{NUM_EPOCHS} [val]",
            ):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if amp_enabled:
                    with torch.cuda.amp.autocast(enabled=True):
                        logits = model(x)
                        loss = criterion(logits, y)
                else:
                    logits = model(x)
                    loss = criterion(logits, y)

                val_loss += float(loss.detach().cpu())
                val_iou_sum += calculate_iou(logits, y).item()
                val_batches += 1

                del x, y, logits, loss

        val_iou = val_iou_sum / max(1, val_batches)

        hist["val_loss"].append(val_loss / max(1, val_batches))
        hist["val_iou"].append(val_iou)

        free_gpu_memory()

        print(
            f"\nLightSegFormer cloud epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss={hist['train_loss'][-1]:.4f} "
            f"Val Loss={hist['val_loss'][-1]:.4f} | "
            f"Train IoU={hist['train_iou'][-1]:.4f} "
            f"Val IoU={hist['val_iou'][-1]:.4f}\n"
        )

        if val_iou > best_val_iou + 1e-6:
            best_val_iou = val_iou
            best_epoch = epoch + 1
            bad_epochs = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "best_epoch": best_epoch,
                    "val_iou": best_val_iou,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "hist": hist,
                    "model": "LightSegFormerCloud",
                    "training_setting": "cloud_only_no_reconstruction",
                    "num_classes": NUM_CLASSES,
                    "in_channels": IN_CHANNELS,
                    "batch_size": BATCH_SIZE,
                    "lr": LR,
                    "patience": PATIENCE,
                    "seed": BASE_SEED,
                    "data_root": DATA_ROOT,
                    "use_amp": amp_enabled,
                    "strict_determinism": USE_STRICT_DETERMINISM,
                    "architecture": {
                        "variant": "LightSegFormer-B0-style",
                        "encoder": "MiT-B0-style",
                        "embed_dims": [32, 64, 160, 256],
                        "depths": [2, 2, 2, 2],
                        "num_heads": [1, 2, 5, 8],
                        "sr_ratios": [8, 4, 2, 1],
                        "decoder_channels": 128,
                    },
                },
                save_path,
            )

            print(
                f" Saved best LightSegFormer cloud-only model to {save_path} "
                f"(val IoU={best_val_iou:.4f})"
            )

        else:
            bad_epochs += 1

            print(f"No improvement for {bad_epochs}/{PATIENCE} epochs.")

            if bad_epochs >= PATIENCE:
                print(f"Early stopping after {PATIENCE} bad epochs.")
                break

        free_gpu_memory()

    print(
        f" Best LightSegFormer cloud-only: "
        f"epoch={best_epoch}, val IoU={best_val_iou:.4f}"
    )

    return hist, best_val_iou


def train_reconstruction(
    model,
    train_loader,
    val_loader,
    device,
    best_path,
    overall_path,
    recon_name,
    recon_input,
    recon_target,
):
    ce = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
    )

    amp_enabled = bool(USE_AMP and device.type == "cuda")

    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled,
    )

    hist = {
        "train_iou": [],
        "val_iou": [],
        "train_seg": [],
        "val_seg": [],
        "train_rec": [],
        "val_rec": [],
        "train_total": [],
        "val_total": [],
    }

    best_val_iou = -1.0
    best_epoch = 0
    bad_epochs = 0
    printed_shapes = False

    for epoch in range(NUM_EPOCHS):
        model.train()

        tr_iou = 0.0
        tr_seg = 0.0
        tr_rec = 0.0
        tr_total = 0.0

        tr_batches = 0
        tr_rec_batches = 0

        for x, y, rec_cloudy, rec_clean, mask in tqdm(
            train_loader,
            desc=f"LightSegFormer {recon_name} epoch {epoch + 1}/{NUM_EPOCHS} [train]",
        ):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            rec_cloudy = rec_cloudy.to(device, non_blocking=True)
            rec_clean = rec_clean.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                with torch.cuda.amp.autocast(enabled=True):
                    if not printed_shapes:
                        seg_logits, rec_pred, shape_info = model(
                            x,
                            rec_cloudy,
                            return_shapes=True,
                        )
                    else:
                        seg_logits, rec_pred = model(x, rec_cloudy)

                    seg_loss = ce(seg_logits, y)

                    use_rec = mask.sum().item() > 0.0

                    if use_rec:
                        rec_loss = masked_l1(rec_pred, rec_clean, mask)
                    else:
                        rec_loss = seg_loss.new_tensor(0.0)

                    loss = W_SEG * seg_loss + W_REC * rec_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                if not printed_shapes:
                    seg_logits, rec_pred, shape_info = model(
                        x,
                        rec_cloudy,
                        return_shapes=True,
                    )
                else:
                    seg_logits, rec_pred = model(x, rec_cloudy)

                seg_loss = ce(seg_logits, y)

                use_rec = mask.sum().item() > 0.0

                if use_rec:
                    rec_loss = masked_l1(rec_pred, rec_clean, mask)
                else:
                    rec_loss = seg_loss.new_tensor(0.0)

                loss = W_SEG * seg_loss + W_REC * rec_loss

                loss.backward()
                optimizer.step()

            if not printed_shapes:
                print("\nShape check:")
                for k, v in shape_info.items():
                    print(f"  {k:28s}: {v}")
                printed_shapes = True

            tr_iou += calculate_iou(seg_logits.detach(), y).item()
            tr_seg += float(seg_loss.detach().cpu())
            tr_total += float(loss.detach().cpu())
            tr_batches += 1

            if use_rec:
                tr_rec += float(rec_loss.detach().cpu())
                tr_rec_batches += 1

            del x, y, rec_cloudy, rec_clean, mask
            del seg_logits, rec_pred, seg_loss, rec_loss, loss

        hist["train_iou"].append(tr_iou / max(1, tr_batches))
        hist["train_seg"].append(tr_seg / max(1, tr_batches))
        hist["train_total"].append(tr_total / max(1, tr_batches))
        hist["train_rec"].append(tr_rec / max(1, tr_rec_batches))

        free_gpu_memory()

        model.eval()

        val_iou_sum = 0.0
        val_seg = 0.0
        val_rec = 0.0
        val_total = 0.0

        val_batches = 0
        val_rec_batches = 0

        with torch.no_grad():
            for x, y, rec_cloudy, rec_clean, mask in tqdm(
                val_loader,
                desc=f"LightSegFormer {recon_name} epoch {epoch + 1}/{NUM_EPOCHS} [val]",
            ):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                rec_cloudy = rec_cloudy.to(device, non_blocking=True)
                rec_clean = rec_clean.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                if amp_enabled:
                    with torch.cuda.amp.autocast(enabled=True):
                        seg_logits, rec_pred = model(x, rec_cloudy)
                        seg_loss = ce(seg_logits, y)
                        use_rec = mask.sum().item() > 0.0
                        if use_rec:
                            rec_loss = masked_l1(rec_pred, rec_clean, mask)
                        else:
                            rec_loss = seg_loss.new_tensor(0.0)
                        loss = W_SEG * seg_loss + W_REC * rec_loss
                else:
                    seg_logits, rec_pred = model(x, rec_cloudy)
                    seg_loss = ce(seg_logits, y)
                    use_rec = mask.sum().item() > 0.0
                    if use_rec:
                        rec_loss = masked_l1(rec_pred, rec_clean, mask)
                    else:
                        rec_loss = seg_loss.new_tensor(0.0)
                    loss = W_SEG * seg_loss + W_REC * rec_loss

                val_iou_sum += calculate_iou(seg_logits, y).item()
                val_seg += float(seg_loss.detach().cpu())
                val_total += float(loss.detach().cpu())
                val_batches += 1

                if use_rec:
                    val_rec += float(rec_loss.detach().cpu())
                    val_rec_batches += 1

                del x, y, rec_cloudy, rec_clean, mask
                del seg_logits, rec_pred, seg_loss, rec_loss, loss

        val_iou = val_iou_sum / max(1, val_batches)

        hist["val_iou"].append(val_iou)
        hist["val_seg"].append(val_seg / max(1, val_batches))
        hist["val_total"].append(val_total / max(1, val_batches))
        hist["val_rec"].append(val_rec / max(1, val_rec_batches))

        free_gpu_memory()

        print(
            f"\nLightSegFormer {recon_name} epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train IoU={hist['train_iou'][-1]:.4f} "
            f"Val IoU={hist['val_iou'][-1]:.4f} | "
            f"Train Total={hist['train_total'][-1]:.4f} "
            f"Val Total={hist['val_total'][-1]:.4f} | "
            f"Train Seg={hist['train_seg'][-1]:.4f} "
            f"Val Seg={hist['val_seg'][-1]:.4f} | "
            f"Train Rec={hist['train_rec'][-1]:.4f} "
            f"Val Rec={hist['val_rec'][-1]:.4f}\n"
        )

        if val_iou > best_val_iou + 1e-6:
            best_val_iou = val_iou
            best_epoch = epoch + 1
            bad_epochs = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "best_epoch": best_epoch,
                    "val_iou": best_val_iou,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "hist": hist,
                    "loss_weights": {
                        "w_seg": W_SEG,
                        "w_rec": W_REC,
                    },
                    "model": f"LightSegFormer{recon_name}Recon",
                    "reconstruction_input": recon_input,
                    "reconstruction_target": recon_target,
                    "pretrained_ckpt_path": CLOUD_CKPT_PATH,
                    "num_classes": NUM_CLASSES,
                    "in_channels": IN_CHANNELS,
                    "batch_size": BATCH_SIZE,
                    "lr": LR,
                    "patience": PATIENCE,
                    "seed": BASE_SEED,
                    "data_root": DATA_ROOT,
                    "use_amp": amp_enabled,
                    "strict_determinism": USE_STRICT_DETERMINISM,
                    "encoder_backbone_frozen": True,
                    "seg_dropout": 0.1,
                    "architecture": {
                        "variant": "LightSegFormer-B0-style",
                        "encoder": "MiT-B0-style",
                        "embed_dims": [32, 64, 160, 256],
                        "depths": [2, 2, 2, 2],
                        "num_heads": [1, 2, 5, 8],
                        "sr_ratios": [8, 4, 2, 1],
                        "decoder_channels": 128,
                        "has_auxiliary_reconstruction_branch": True,
                        "reconstruction_head_order": "head_feat -> upsample -> two 3x3 convs -> concat f0 early feature + cloudy RGB/NDI -> 1x1 -> upsample -> 64->32->3",
                        "segmentation_fusion": "full-resolution head_feat + detached r_feat32 -> 256 -> classes",
                    },
                },
                best_path,
            )

            print(
                f"✅ Saved best LightSegFormer {recon_name} model to {best_path} "
                f"(val IoU={best_val_iou:.4f})"
            )

            if overall_path is not None:
                shutil.copyfile(best_path, overall_path)
                print(f"✅ Copied to overall path: {overall_path}")

        else:
            bad_epochs += 1

            print(f"No improvement for {bad_epochs}/{PATIENCE} epochs.")

            if bad_epochs >= PATIENCE:
                print(f"⛔ Early stopping after {PATIENCE} bad epochs.")
                break

        free_gpu_memory()

    print(
        f"🏁 Best LightSegFormer {recon_name}: "
        f"epoch={best_epoch}, val IoU={best_val_iou:.4f}"
    )

    return hist, best_val_iou




if __name__ == "__main__":
    set_seed_everywhere(BASE_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 100)
    print("LIGHT SEGFORMER-B0-STYLE seed-6 pipeline")
    print("1) cloud-only/no reconstruction")
    print("2) RGB reconstruction at w_rec=0.4")
    print("3) NDI reconstruction at w_rec=0.4")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Seed: {BASE_SEED}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"LR: {LR}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Patience: {PATIENCE}")
    print(f"w_rec: {W_REC}")
    print(f"w_seg: {W_SEG}")
    print(f"OVERWRITE_EXISTING: {OVERWRITE_EXISTING}")
    print("Light SegFormer config: MiT-B0-style, embed_dims [32,64,160,256], decoder channels 128")
    print("=" * 100)

    summary = {}

    # ---------------- Stage 1: no reconstruction ----------------
    if RUN_CLOUD_ONLY:
        if os.path.isfile(CLOUD_CKPT_PATH) and not OVERWRITE_EXISTING:
            print(f"Skipping cloud-only because {CLOUD_CKPT_PATH} exists.")
            summary["cloud_only"] = {
                "checkpoint": CLOUD_CKPT_PATH,
                "skipped": True,
            }

        else:
            print("\n" + "#" * 100)
            print("STAGE 1/3: LIGHT SEGFORMER CLOUD-ONLY / NO RECONSTRUCTION")
            print("#" * 100)

            check_cloud_paths()

            set_seed_everywhere(BASE_SEED)

            train_loader, val_loader = make_cloud_loaders(BASE_SEED)

            cloud_model = LightSegFormerCloud(
                IN_CHANNELS,
                NUM_CLASSES,
            )

            hist, best = train_cloud_only(
                cloud_model,
                train_loader,
                val_loader,
                device,
                CLOUD_CKPT_PATH,
            )

            summary["cloud_only"] = {
                "checkpoint": CLOUD_CKPT_PATH,
                "best_val_iou": best,
                "skipped": False,
            }

            del cloud_model
            del hist
            del train_loader
            del val_loader

            free_gpu_memory()

    if not os.path.isfile(CLOUD_CKPT_PATH):
        raise FileNotFoundError(f"Cloud checkpoint not found: {CLOUD_CKPT_PATH}")

    # ---------------- Stage 2: RGB reconstruction ----------------
    if RUN_RGB_RECON:
        if os.path.isfile(RGB_REC_CKPT_PATH) and not OVERWRITE_EXISTING:
            print(f"Skipping RGB because {RGB_REC_CKPT_PATH} exists.")
            summary["rgb_reconstruction"] = {
                "checkpoint": RGB_REC_CKPT_PATH,
                "overall_checkpoint": RGB_REC_OVERALL_PATH,
                "skipped": True,
            }

        else:
            print("\n" + "#" * 100)
            print("STAGE 2/3: LIGHT SEGFORMER RGB RECONSTRUCTION")
            print("#" * 100)

            check_rgb_paths()

            set_seed_everywhere(BASE_SEED)

            train_loader, val_loader = make_rgb_loaders(BASE_SEED)

            rgb_model = build_rgb_model(device)

            hist, best = train_reconstruction(
                model=rgb_model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                best_path=RGB_REC_CKPT_PATH,
                overall_path=RGB_REC_OVERALL_PATH,
                recon_name="RGB",
                recon_input="RGB from cloudy CLOUDY_TRAINING_S2",
                recon_target="RGB from clear s2_0",
            )

            summary["rgb_reconstruction"] = {
                "checkpoint": RGB_REC_CKPT_PATH,
                "overall_checkpoint": RGB_REC_OVERALL_PATH,
                "best_val_iou": best,
                "skipped": False,
            }

            del rgb_model
            del hist
            del train_loader
            del val_loader

            free_gpu_memory()

    # ---------------- Stage 3: NDI reconstruction ----------------
    if RUN_NDI_RECON:
        if os.path.isfile(NDI_REC_CKPT_PATH) and not OVERWRITE_EXISTING:
            print(f"Skipping NDI because {NDI_REC_CKPT_PATH} exists.")
            summary["ndi_reconstruction"] = {
                "checkpoint": NDI_REC_CKPT_PATH,
                "overall_checkpoint": NDI_REC_OVERALL_PATH,
                "skipped": True,
            }

        else:
            print("\n" + "#" * 100)
            print("STAGE 3/3: LIGHT SEGFORMER NDI RECONSTRUCTION")
            print("#" * 100)

            check_ndi_paths()

            set_seed_everywhere(BASE_SEED)

            train_loader, val_loader = make_ndi_loaders(BASE_SEED)

            ndi_model = build_ndi_model(device)

            hist, best = train_reconstruction(
                model=ndi_model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                best_path=NDI_REC_CKPT_PATH,
                overall_path=NDI_REC_OVERALL_PATH,
                recon_name="NDI",
                recon_input="NDI from cloudy ndi_cloud_TRAINING_S2",
                recon_target="NDI from clean ndi",
            )

            summary["ndi_reconstruction"] = {
                "checkpoint": NDI_REC_CKPT_PATH,
                "overall_checkpoint": NDI_REC_OVERALL_PATH,
                "best_val_iou": best,
                "skipped": False,
            }

            del ndi_model
            del hist
            del train_loader
            del val_loader

            free_gpu_memory()

    print("\n" + "=" * 100)
    print("PIPELINE FINISHED")
    print("=" * 100)

    for name, info in summary.items():
        print(f"\n{name}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    print("\nSaved checkpoints:")
    print(f"  Cloud-only:  {CLOUD_CKPT_PATH}")
    print(f"  RGB rec:     {RGB_REC_CKPT_PATH}")
    print(f"  RGB overall: {RGB_REC_OVERALL_PATH}")
    print(f"  NDI rec:     {NDI_REC_CKPT_PATH}")
    print(f"  NDI overall: {NDI_REC_OVERALL_PATH}")

    free_gpu_memory()
