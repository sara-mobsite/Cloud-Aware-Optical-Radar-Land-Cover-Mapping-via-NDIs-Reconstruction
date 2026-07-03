

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
import csv
import gc
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import rasterio
from tqdm import tqdm




DATA_ROOT = Path("")
EXT_ROOT = Path("")

FULL_LUOJIA_ROOT = DATA_ROOT / "LuojiaSET-OSFCR"
SELECTED_TEST_ROOT = EXT_ROOT / "LuojiaSET_OSFCR_selected_100"

# Stage 1 simple Light SegFormer outputs.
SIMPLE_OUT_ROOT = EXT_ROOT / "luojia_lightsegformer_b0_segmentation_seed6_patchsplit_70_30"
SIMPLE_CSV_ROOT = SIMPLE_OUT_ROOT / "csv"
SIMPLE_PRED_ROOT = SIMPLE_OUT_ROOT / "test_predictions"
SIMPLE_SAVE_PATH = SIMPLE_OUT_ROOT / "lightsegformer_b0_luojia_seed6_patchsplit_70_30.pth"

# Stage 2 Light SegFormer + reconstruction outputs.
RECON_OUT_ROOT = EXT_ROOT / "luojia_lightsegformer_b0_multitask_ndi_rec040_from_simple_patchsplit_70_30_no_cache"
RECON_CSV_ROOT = RECON_OUT_ROOT / "csv"
RECON_PRED_ROOT = RECON_OUT_ROOT / "test_predictions"
RECON_BEST_PATH = RECON_OUT_ROOT / "best_lightsegformer_b0_multitask_ndi_rec040_from_simple.pth"

for p in [
    SIMPLE_OUT_ROOT,
    SIMPLE_CSV_ROOT,
    SIMPLE_PRED_ROOT,
    RECON_OUT_ROOT,
    RECON_CSV_ROOT,
    RECON_PRED_ROOT,
]:
    p.mkdir(parents=True, exist_ok=True)

TARGET_PERCENTAGE_RANGES = [
    (0, 20),
    (20, 40),
    (40, 60),
    (60, 80),
    (80, 100),
]

BASE_SEED = 6
IN_CHANNELS = 12
NUM_CLASSES = 6
IGNORE_INDEX = 255

BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS_STAGE1 = 200
NUM_EPOCHS_STAGE2 = 200
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-3
PATIENCE_STAGE1 = 20
PATIENCE_STAGE2 = 20

TRAIN_RATIO = 0.70
VAL_RATIO = 0.30

# Fixed reconstruction weight only.
W_REC = 0.4
W_SEG = 0.6

USE_STRICT_DETERMINISM = True

S1_BANDS = [1, 2]
S2_INPUT_BANDS = [2, 3, 4, 5, 6, 7, 8, 9, 12, 13]

# NDI bands, rasterio 1-based indexing.
B3 = 3
B4 = 4
B8 = 8
B11 = 11

MEAN_VALS = torch.tensor([
    -12.64,
    -19.35,
    438.37,
    614.05,
    588.40,
    942.84,
    1769.93,
    2049.55,
    2193.29,
    2235.55,
    1568.22,
    997.73,
], dtype=torch.float32).view(12, 1, 1)

STD_VALS = torch.tensor([
    5.13,
    5.59,
    607.02,
    603.29,
    684.56,
    738.43,
    1100.45,
    1275.80,
    1369.37,
    1356.54,
    1070.16,
    813.52,
], dtype=torch.float32).view(12, 1, 1)

# Run switches.
RUN_STAGE_1_SIMPLE_SEGFORMER = True
RUN_STAGE_2_RECON_FROM_SIMPLE = True

# Optional selected-test evaluation. Off by default to save disk/time.
RUN_STAGE_1_TEST_EVAL = False
RUN_STAGE_2_TEST_EVAL = False
SAVE_TEST_PREDICTIONS = False




def free_gpu_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def cleanup_gpu_cpu_resources(note=""):
    print("\n" + "=" * 80)
    print("Cleaning GPU/CPU resources", f"- {note}" if note else "")
    print("=" * 80)

    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print("CUDA cleanup warning:", e)

        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"CUDA memory allocated: {allocated:.2f} MB")
        print(f"CUDA memory reserved:  {reserved:.2f} MB")
    else:
        print("CUDA is not available.")

    print("Cleanup done.\n")


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
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return _seed_worker


set_seed_everywhere(BASE_SEED)



def check_paths():
    if not FULL_LUOJIA_ROOT.exists():
        raise FileNotFoundError(f"Missing full Luojia root: {FULL_LUOJIA_ROOT}")

    if not SELECTED_TEST_ROOT.exists():
        raise FileNotFoundError(f"Missing selected test root: {SELECTED_TEST_ROOT}")

    print("Full Luojia root:", FULL_LUOJIA_ROOT)
    print("Selected test root:", SELECTED_TEST_ROOT)
    print("Simple Light SegFormer output root:", SIMPLE_OUT_ROOT)
    print("Reconstruction Light SegFormer output root:", RECON_OUT_ROOT)
    print("NDI mode: on-the-fly, no disk cache")


check_paths()




def safe_name(text):
    return (
        str(text)
        .replace("%", "pct")
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def detect_percentage_folders(root):
    folders = []

    for p in root.iterdir():
        if not p.is_dir():
            continue

        name = p.name.strip()
        m = re.match(r"^(\d+)%?-(\d+)%?$", name)

        if not m:
            continue

        start = int(m.group(1))
        end = int(m.group(2))

        if (start, end) not in TARGET_PERCENTAGE_RANGES:
            continue

        folders.append((start, end, p))

    folders = sorted(folders, key=lambda x: (x[0], x[1]))

    if len(folders) == 0:
        raise RuntimeError(f"No target percentage folders found in: {root}")

    print(f"\nDetected percentage folders in {root}:")
    for start, end, p in folders:
        print(f"  {start}-{end}% -> {p}")

    return folders


def extract_sample_key(path):
    """
    Example:
      ROIs_12_s2_cloudy_xxx_p0034.tif

    Returns:
      ROIs_12_p0034
    """
    stem = Path(path).stem
    parts = stem.split("_")

    patch_token = None
    patch_index = None

    for i, part in enumerate(parts):
        if re.fullmatch(r"p\d+", part, flags=re.IGNORECASE):
            patch_token = part
            patch_index = i
            break

    if patch_token is None:
        return None

    before_patch = parts[:patch_index]

    modality_tokens = {
        "s1",
        "s2",
        "sar",
        "clear",
        "cloudy",
        "cloud",
        "cd",
        "lulc",
        "mask",
        "cloudmask",
        "cloud_detection",
        "result",
        "results",
        "land",
        "cover",
        "maps",
        "landcover",
        "lc",
        "dfc",
    }

    roi_parts = []

    for part in before_patch:
        if part.lower() in modality_tokens:
            break
        roi_parts.append(part)

    if len(roi_parts) == 0:
        return None

    roi_id = "_".join(roi_parts)
    return f"{roi_id}_{patch_token}"


def extract_roi_id(sample_key):
    m = re.search(r"(.+)_p\d+$", sample_key)
    if m:
        return m.group(1)
    return sample_key


def collect_files_by_key(folder):
    files_by_key = {}

    if not folder.exists():
        return files_by_key

    files = sorted(list(folder.glob("*.tif")) + list(folder.glob("*.tiff")))

    for p in files:
        key = extract_sample_key(p)
        if key is None:
            continue

        files_by_key[key] = p

    return files_by_key


def collect_samples_from_root(root):
    """
    Required:
      s1/
      s2/
      s2_cloudy/
      land_cover_maps/

    Optional:
      cloud_detection_results/
    """
    folders = detect_percentage_folders(root)
    samples = []

    for start, end, percentage_dir in folders:
        percentage_name = percentage_dir.name

        s1_dir = percentage_dir / "s1"
        clean_s2_dir = percentage_dir / "s2"
        cloudy_s2_dir = percentage_dir / "s2_cloudy"
        cloud_mask_dir = percentage_dir / "cloud_detection_results"
        label_dir = percentage_dir / "land_cover_maps"

        s1_files = collect_files_by_key(s1_dir)
        clean_s2_files = collect_files_by_key(clean_s2_dir)
        cloudy_s2_files = collect_files_by_key(cloudy_s2_dir)
        cloud_mask_files = collect_files_by_key(cloud_mask_dir)
        label_files = collect_files_by_key(label_dir)

        matched_keys = sorted(
            set(s1_files.keys())
            & set(clean_s2_files.keys())
            & set(cloudy_s2_files.keys())
            & set(label_files.keys())
        )

        no_mask_count = sum(1 for k in matched_keys if k not in cloud_mask_files)

        print("\n" + "=" * 80)
        print(f"Collecting: {root} | {percentage_name}")
        print("=" * 80)
        print("S1 files:", len(s1_files))
        print("Clean S2 files:", len(clean_s2_files))
        print("Cloudy S2 files:", len(cloudy_s2_files))
        print("Cloud-mask files:", len(cloud_mask_files))
        print("Label files:", len(label_files))
        print("Matched samples:", len(matched_keys))
        print("Matched samples WITHOUT cloud-mask file:", no_mask_count)

        for key in matched_keys:
            samples.append({
                "percentage_name": percentage_name,
                "percentage_range": f"{start}-{end}",
                "sample_key": key,
                "roi_id": extract_roi_id(key),
                "s1_path": s1_files[key],
                "clean_s2_path": clean_s2_files[key],
                "cloudy_s2_path": cloudy_s2_files[key],
                "cloud_mask_path": cloud_mask_files.get(key, None),
                "label_path": label_files[key],
            })

    return samples




print("\nCollecting selected TEST samples...")
test_samples = collect_samples_from_root(SELECTED_TEST_ROOT)

test_patch_keys = set(s["sample_key"] for s in test_samples)

print("\nSelected test samples:", len(test_samples))
print("Unique selected test patch keys:", len(test_patch_keys))

if len(test_samples) == 0:
    raise RuntimeError("Selected test set is empty.")

print("\nCollecting FULL Luojia samples...")
all_full_samples = collect_samples_from_root(FULL_LUOJIA_ROOT)

print("\nFull Luojia matched samples:", len(all_full_samples))

trainval_samples = [
    s for s in all_full_samples
    if s["sample_key"] not in test_patch_keys
]

print("Train/val samples after removing selected test patch keys:", len(trainval_samples))

if len(trainval_samples) == 0:
    raise RuntimeError("No train/validation samples left after excluding selected test patches.")


def split_train_val_by_patch_key(samples, seed=0, train_ratio=0.70):
    by_patch = defaultdict(list)

    for s in samples:
        by_patch[s["sample_key"]].append(s)

    patch_keys = sorted(by_patch.keys())

    rng = random.Random(seed)
    rng.shuffle(patch_keys)

    train_n = int(train_ratio * len(patch_keys))

    train_patch_keys = set(patch_keys[:train_n])
    val_patch_keys = set(patch_keys[train_n:])

    train_samples_local = []
    val_samples_local = []

    for key in train_patch_keys:
        train_samples_local.extend(by_patch[key])

    for key in val_patch_keys:
        val_samples_local.extend(by_patch[key])

    if len(train_samples_local) == 0:
        raise RuntimeError("Training split is empty.")

    if len(val_samples_local) == 0:
        raise RuntimeError("Validation split is empty.")

    return train_samples_local, val_samples_local, train_patch_keys, val_patch_keys


train_samples, val_samples, train_patch_keys, val_patch_keys = split_train_val_by_patch_key(
    trainval_samples,
    seed=BASE_SEED,
    train_ratio=TRAIN_RATIO,
)

print("\nPatch-level train/validation split:")
print("Train unique patch keys:", len(train_patch_keys))
print("Val unique patch keys:", len(val_patch_keys))
print("Train samples:", len(train_samples))
print("Val samples:", len(val_samples))


def assert_no_leakage():
    train_keys = set(s["sample_key"] for s in train_samples)
    val_keys = set(s["sample_key"] for s in val_samples)
    test_keys = set(s["sample_key"] for s in test_samples)

    train_test_overlap = train_keys & test_keys
    val_test_overlap = val_keys & test_keys
    train_val_overlap = train_keys & val_keys

    print("\nLeakage check:")
    print("Train/test overlap patch keys:", len(train_test_overlap))
    print("Val/test overlap patch keys:", len(val_test_overlap))
    print("Train/val overlap patch keys:", len(train_val_overlap))

    if len(train_test_overlap) > 0:
        print("Examples:", sorted(list(train_test_overlap))[:20])
        raise RuntimeError("Leakage detected: training patches appear in selected test.")

    if len(val_test_overlap) > 0:
        print("Examples:", sorted(list(val_test_overlap))[:20])
        raise RuntimeError("Leakage detected: validation patches appear in selected test.")

    if len(train_val_overlap) > 0:
        print("Examples:", sorted(list(train_val_overlap))[:20])
        raise RuntimeError("Leakage detected: same patch appears in train and validation.")

    print("No leakage detected.")


assert_no_leakage()

print("\nCloud percentage distribution:")
print("Train:", Counter([s["percentage_name"] for s in train_samples]))
print("Val:  ", Counter([s["percentage_name"] for s in val_samples]))
print("Test: ", Counter([s["percentage_name"] for s in test_samples]))


def save_split_manifest(csv_root, tag):
    split_csv = csv_root / f"split_manifest_{tag}.csv"

    with open(split_csv, "w", newline="") as f:
        fieldnames = [
            "split",
            "percentage_name",
            "sample_key",
            "roi_id",
            "s1_path",
            "clean_s2_path",
            "cloudy_s2_path",
            "cloud_mask_path",
            "label_path",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for split_name, rows in [
            ("train", train_samples),
            ("val", val_samples),
            ("test_selected", test_samples),
        ]:
            for s in rows:
                writer.writerow({
                    "split": split_name,
                    "percentage_name": s["percentage_name"],
                    "sample_key": s["sample_key"],
                    "roi_id": s["roi_id"],
                    "s1_path": str(s["s1_path"]),
                    "clean_s2_path": str(s["clean_s2_path"]),
                    "cloudy_s2_path": str(s["cloudy_s2_path"]),
                    "cloud_mask_path": "" if s["cloud_mask_path"] is None else str(s["cloud_mask_path"]),
                    "label_path": str(s["label_path"]),
                })

    print("Saved split manifest:", split_csv)


save_split_manifest(SIMPLE_CSV_ROOT, "simple")
save_split_manifest(RECON_CSV_ROOT, "recon")


# ============================================================
# 6) ON-THE-FLY NDI COMPUTATION
# ============================================================

def compute_ndi_from_s2_file(path):
    with rasterio.open(path) as src:
        b3 = src.read(B3).astype("float32")
        b4 = src.read(B4).astype("float32")
        b8 = src.read(B8).astype("float32")
        b11 = src.read(B11).astype("float32")

    eps = 1e-6

    ndvi = (b8 - b4) / (b8 + b4 + eps)
    ndwi = (b3 - b8) / (b3 + b8 + eps)
    ndbi = (b11 - b8) / (b11 + b8 + eps)

    ndi = np.stack([ndvi, ndwi, ndbi], axis=0)
    ndi = np.nan_to_num(ndi, nan=0.0, posinf=0.0, neginf=0.0)
    ndi = np.clip(ndi, -1.0, 1.0).astype("float32")

    return ndi


def read_cloud_mask_for_loss(path, shape):
    if path is None:
        # Missing mask means reconstruct the whole image.
        return np.ones(shape, dtype="float32")

    try:
        with rasterio.open(path) as src:
            arr = src.read(1)

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        mask = (arr > 0).astype("float32")

        if mask.shape != shape:
            return np.ones(shape, dtype="float32")

        return mask

    except Exception:
        return np.ones(shape, dtype="float32")


# ============================================================
# 7) DATASETS
# ============================================================

class LuojiaSegFormerDataset(Dataset):
    """
    Stage 1 simple segmentation dataset:
      input: S1 + cloudy S2
      target: segmentation label
    """
    def __init__(self, samples, return_index=False):
        self.samples = samples
        self.return_index = return_index
        self.mean = MEAN_VALS
        self.std = STD_VALS

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        with rasterio.open(item["s1_path"]) as src1:
            s1 = src1.read(S1_BANDS).astype("float32")

        with rasterio.open(item["cloudy_s2_path"]) as src2:
            s2_cloudy_input = src2.read(S2_INPUT_BANDS).astype("float32")

        x = torch.from_numpy(
            np.concatenate([s1, s2_cloudy_input], axis=0)
        ).float()

        x = (x - self.mean) / self.std

        with rasterio.open(item["label_path"]) as src:
            label = src.read(1).astype("int64")

        valid = (label >= 0) & (label < NUM_CLASSES)
        clean_label = np.full(label.shape, IGNORE_INDEX, dtype=np.int64)
        clean_label[valid] = label[valid]
        y = torch.from_numpy(clean_label).long()

        if self.return_index:
            return x, y, idx

        return x, y


class LuojiaSegFormerMultitaskNDIDataset(Dataset):
    """
    Stage 2 reconstruction dataset:
      input: S1 + cloudy S2; auxiliary input: cloudy NDI
      targets: segmentation label + clean NDI
    """
    def __init__(self, samples, return_index=False):
        self.samples = samples
        self.return_index = return_index
        self.mean = MEAN_VALS
        self.std = STD_VALS

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        with rasterio.open(item["s1_path"]) as src1:
            s1 = src1.read(S1_BANDS).astype("float32")

        with rasterio.open(item["cloudy_s2_path"]) as src2:
            s2_cloudy_input = src2.read(S2_INPUT_BANDS).astype("float32")

        x = torch.from_numpy(
            np.concatenate([s1, s2_cloudy_input], axis=0)
        ).float()

        x = (x - self.mean) / self.std

        ndi_cloudy = torch.from_numpy(
            compute_ndi_from_s2_file(item["cloudy_s2_path"])
        ).float()

        ndi_clean = torch.from_numpy(
            compute_ndi_from_s2_file(item["clean_s2_path"])
        ).float()

        h, w = ndi_clean.shape[1], ndi_clean.shape[2]
        mask_np = read_cloud_mask_for_loss(item["cloud_mask_path"], shape=(h, w))
        mask = torch.from_numpy(mask_np).float().unsqueeze(0)

        with rasterio.open(item["label_path"]) as src:
            label = src.read(1).astype("int64")

        valid = (label >= 0) & (label < NUM_CLASSES)
        clean_label = np.full(label.shape, IGNORE_INDEX, dtype=np.int64)
        clean_label[valid] = label[valid]
        y = torch.from_numpy(clean_label).long()

        if self.return_index:
            return x, y, ndi_cloudy, ndi_clean, mask, idx

        return x, y, ndi_cloudy, ndi_clean, mask



# ============================================================
# 8) LIGHT SEGFORMER-B0 MODEL BLOCKS
# ============================================================

class DropPath(nn.Module):
    """
    Stochastic depth. With drop_path_rate=0.0 in this script, it is an identity,
    but keeping the module makes the SegFormer blocks complete.
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
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
        num_classes=NUM_CLASSES,
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
# 9) SIMPLE LIGHT SEGFORMER-B0
# ============================================================

class LightSegFormerCloud(nn.Module):
    """
    Light SegFormer-B0-style segmentation-only model for 12-channel input.
    """
    def __init__(
        self,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
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


# ============================================================
# 10) MULTITASK LIGHT SEGFORMER-B0 + NDI RECONSTRUCTION
# ============================================================

class LightSegFormerRecon(nn.Module):
    """
    Light SegFormer-B0-style model with auxiliary NDI reconstruction.

    Stage 2 protocol:
      - load compatible weights from the segmentation-only checkpoint
      - freeze the encoder/backbone only
      - keep the SegFormer decoder, segmentation fusion head, and reconstruction head trainable
    """
    def __init__(
        self,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        recon_name="NDI",
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

        # Segmentation branch: use full-resolution shared decoder feature fused
        # with detached reconstruction features, matching your prior setup.
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

    def forward(self, x, ndi_cloudy, return_shapes=False):
        input_size = x.shape[-2:]

        feats = self.backbone(x)
        early_feat = feats[0]  # H/4, 32

        # Shared SegFormer decoder feature.
        head_feat = self.head.forward_features(feats)  # H/4, 128

        # Reconstruction branch.
        shared_up = F.interpolate(
            head_feat,
            size=early_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        r128 = self.recon_pre(shared_up)

        ndi_cloudy_early = F.interpolate(
            ndi_cloudy,
            size=early_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        recon_concat = torch.cat(
            [r128, early_feat, ndi_cloudy_early],
            dim=1,
        )

        r = self.recon_inject(recon_concat)

        r_full = F.interpolate(
            r,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        r_feat32 = self.recon_final(r_full)
        ndi_pred = self.recon_out(r_feat32)

        # Segmentation branch.
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
            return seg_logits, ndi_pred, {
                "input": tuple(x.shape),
                "f0_early_H4_32": tuple(feats[0].shape),
                "f1_H8_64": tuple(feats[1].shape),
                "f2_H16_160": tuple(feats[2].shape),
                "f3_H32_256": tuple(feats[3].shape),
                "head_feat_H4_128": tuple(head_feat.shape),
                "shared_up_H4_128": tuple(shared_up.shape),
                "r128_after_two_3x3": tuple(r128.shape),
                "ndi_cloudy_H4_3": tuple(ndi_cloudy_early.shape),
                "concat_128_plus_32_plus_3": tuple(recon_concat.shape),
                "r_after_1x1_128": tuple(r.shape),
                "r_feat32": tuple(r_feat32.shape),
                "ndi_pred": tuple(ndi_pred.shape),
                "shared_full": tuple(shared_full.shape),
                "seg_feat_256": tuple(seg_feat.shape),
                "seg_logits": tuple(seg_logits.shape),
            }

        return seg_logits, ndi_pred


# ============================================================
# 11) CHECKPOINT LOADING + FREEZING

# ============================================================

def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def strip_prefix_if_present(state_dict, prefix):
    out = {}

    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v

    return out


def load_simple_segformer_into_multitask(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = extract_state_dict(ckpt)
    sd = strip_prefix_if_present(sd, "module.")

    current = model.state_dict()
    loadable = {}
    skipped = []

    for k, v in sd.items():
        # Load compatible backbone and SegFormer decoder/head tensors.
        # The cloud-only classifier head.cls_seg is intentionally not loaded
        # because the reconstruction model uses seg_fuse + seg_out instead.
        if k.startswith("head.cls_seg."):
            skipped.append(k)
            continue

        if k.startswith("backbone.") or k.startswith("head."):
            if k in current and current[k].shape == v.shape:
                loadable[k] = v
            else:
                skipped.append(k)

    missing, unexpected = model.load_state_dict(loadable, strict=False)

    print("\nLoaded simple Light SegFormer checkpoint into multitask Light SegFormer:")
    print("Checkpoint:", ckpt_path)
    print("Loaded compatible keys:", len(loadable))
    print("Skipped keys:", len(skipped))
    print("Missing keys are expected for reconstruction head and segmentation fusion head.")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    if skipped:
        print("First skipped keys:", skipped[:20])


def freeze_segformer_encoder(model):
    """
    Freeze the Light SegFormer encoder/backbone only.
    The decoder, segmentation fusion head, and reconstruction branch stay trainable.
    """
    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone()
    else:
        for p in model.backbone.parameters():
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print("\nLight SegFormer encoder/backbone frozen.")
    print("Frozen module: backbone")
    print("Trainable parameters:", trainable)
    print("Frozen parameters:", frozen)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


# ============================================================
# 12) LOSSES + METRICS
# ============================================================

def masked_l1(pred, target, mask):
    m = mask.expand_as(pred)
    denom = m.sum()

    if denom.item() == 0:
        return pred.new_tensor(0.0)

    loss = torch.abs(pred - target) * m
    loss = loss.sum() / (denom + 1e-6)

    return loss


def calculate_iou(logits, labels, num_classes=NUM_CLASSES):
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        valid_mask = labels != IGNORE_INDEX

        ious = []

        for cls in range(num_classes):
            pred_cls = (preds == cls) & valid_mask
            true_cls = (labels == cls) & valid_mask

            intersection = (pred_cls & true_cls).float().sum()
            union = (pred_cls | true_cls).float().sum()

            if union > 0:
                ious.append(intersection / (union + 1e-6))

        if len(ious) == 0:
            return torch.tensor(0.0, device=logits.device)

        return torch.mean(torch.stack(ious))


# ============================================================
# 13) DATALOADERS
# ============================================================

def make_stage1_loaders():
    worker_init_fn = seed_worker_factory(BASE_SEED)

    train_dataset = LuojiaSegFormerDataset(train_samples)
    val_dataset = LuojiaSegFormerDataset(val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(BASE_SEED),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(BASE_SEED),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=False,
    )

    return train_loader, val_loader


def make_stage2_loaders():
    worker_init_fn = seed_worker_factory(BASE_SEED)

    train_dataset = LuojiaSegFormerMultitaskNDIDataset(train_samples)
    val_dataset = LuojiaSegFormerMultitaskNDIDataset(val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(BASE_SEED),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(BASE_SEED),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=False,
    )

    return train_loader, val_loader


# ============================================================
# 14) STAGE 1 TRAINING: SIMPLE LIGHT SEGFORMER-B0
# ============================================================

def train_simple_segformer(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=NUM_EPOCHS_STAGE1,
    lr=LR_STAGE1,
    save_path=SIMPLE_SAVE_PATH,
    patience=PATIENCE_STAGE1,
):
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    best_val_iou = -1.0
    best_epoch = 0
    bad_epochs = 0
    hist_rows = []

    for epoch in range(num_epochs):
        model.train()

        train_loss_sum = 0.0
        train_iou_sum = 0.0
        train_batches = 0

        for imgs, labels in tqdm(train_loader, desc=f"Stage 1 | Epoch {epoch + 1} train"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(imgs)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.detach().cpu())
            train_iou_sum += calculate_iou(logits.detach(), labels).item()
            train_batches += 1

            del imgs, labels, logits, loss

        model.eval()

        val_loss_sum = 0.0
        val_iou_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Stage 1 | Epoch {epoch + 1} val"):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(imgs)
                loss = criterion(logits, labels)

                val_loss_sum += float(loss.detach().cpu())
                val_iou_sum += calculate_iou(logits, labels).item()
                val_batches += 1

                del imgs, labels, logits, loss

        avg_train_loss = train_loss_sum / max(1, train_batches)
        avg_val_loss = val_loss_sum / max(1, val_batches)
        avg_train_iou = train_iou_sum / max(1, train_batches)
        avg_val_iou = val_iou_sum / max(1, val_batches)

        hist_rows.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_iou": avg_train_iou,
            "val_iou": avg_val_iou,
        })

        hist_csv = SIMPLE_CSV_ROOT / "training_history_lightsegformer_b0_simple.csv"

        with open(hist_csv, "w", newline="") as f:
            fieldnames = [
                "epoch",
                "train_loss",
                "val_loss",
                "train_iou",
                "val_iou",
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(hist_rows)

        print(
            f"\nStage 1 | Epoch {epoch + 1}/{num_epochs} | "
            f"Train loss={avg_train_loss:.4f} IoU={avg_train_iou:.4f} | "
            f"Val loss={avg_val_loss:.4f} IoU={avg_val_iou:.4f}"
        )

        if avg_val_iou > best_val_iou + 1e-6:
            best_val_iou = avg_val_iou
            best_epoch = epoch + 1
            bad_epochs = 0

            torch.save({
                "epoch": epoch + 1,
                "val_iou": best_val_iou,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "hist": hist_rows,
                "model": "LightSegFormerCloud",
                "num_classes": NUM_CLASSES,
                "in_channels": IN_CHANNELS,
                "seed": BASE_SEED,
                "train_ratio": TRAIN_RATIO,
                "val_ratio": VAL_RATIO,
                "stage": "simple_segmentation",
            }, save_path)

            print(f"✅ Saved best Stage 1 checkpoint to: {save_path}")

        else:
            bad_epochs += 1
            print(f"No Stage 1 validation IoU improvement: {bad_epochs}/{patience}")

            if bad_epochs >= patience:
                print("Stage 1 early stopping.")
                break

        free_gpu_memory()

    print("\nStage 1 training finished.")
    print("Best epoch:", best_epoch)
    print("Best validation IoU:", best_val_iou)
    print("Best checkpoint:", save_path)


# ============================================================
# 15) STAGE 2 TRAINING: LIGHT SEGFORMER-B0 + RECONSTRUCTION
# ============================================================

def train_segformer_reconstruction(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=NUM_EPOCHS_STAGE2,
    lr=LR_STAGE2,
    best_path=RECON_BEST_PATH,
    patience=PATIENCE_STAGE2,
):
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    model.to(device)

    best_val_iou = -1.0
    best_epoch = 0
    bad_epochs = 0
    hist_rows = []

    printed_shapes = False

    for epoch in range(num_epochs):
        model.train()

        tr_iou_sum = 0.0
        tr_seg_sum = 0.0
        tr_rec_sum = 0.0
        tr_total_sum = 0.0
        tr_batches = 0
        tr_rec_batches = 0

        for x, y, ndi_cloudy, ndi_clean, mask in tqdm(
            train_loader,
            desc=f"Stage 2 | Epoch {epoch + 1} train",
        ):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            ndi_cloudy = ndi_cloudy.to(device, non_blocking=True)
            ndi_clean = ndi_clean.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if not printed_shapes:
                seg_logits, ndi_pred, shape_info = model(x, ndi_cloudy, return_shapes=True)
                print("\nShape check:")
                for k, v in shape_info.items():
                    print(f"  {k:28s}: {v}")
                printed_shapes = True
            else:
                seg_logits, ndi_pred = model(x, ndi_cloudy)

            seg_loss = ce(seg_logits, y)

            use_rec = mask.sum().item() > 0.0
            rec_loss = masked_l1(ndi_pred, ndi_clean, mask) if use_rec else seg_loss.new_tensor(0.0)

            loss = W_SEG * seg_loss + W_REC * rec_loss

            loss.backward()
            optimizer.step()

            tr_iou_sum += calculate_iou(seg_logits.detach(), y).item()
            tr_seg_sum += float(seg_loss.detach().cpu())
            tr_total_sum += float(loss.detach().cpu())
            tr_batches += 1

            if use_rec:
                tr_rec_sum += float(rec_loss.detach().cpu())
                tr_rec_batches += 1

            del x, y, ndi_cloudy, ndi_clean, mask
            del seg_logits, ndi_pred, seg_loss, rec_loss, loss

        train_iou = tr_iou_sum / max(1, tr_batches)
        train_seg = tr_seg_sum / max(1, tr_batches)
        train_rec = tr_rec_sum / max(1, tr_rec_batches) if tr_rec_batches > 0 else 0.0
        train_total = tr_total_sum / max(1, tr_batches)

        model.eval()

        va_iou_sum = 0.0
        va_seg_sum = 0.0
        va_rec_sum = 0.0
        va_total_sum = 0.0
        va_batches = 0
        va_rec_batches = 0

        with torch.no_grad():
            for x, y, ndi_cloudy, ndi_clean, mask in tqdm(
                val_loader,
                desc=f"Stage 2 | Epoch {epoch + 1} val",
            ):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                ndi_cloudy = ndi_cloudy.to(device, non_blocking=True)
                ndi_clean = ndi_clean.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                seg_logits, ndi_pred = model(x, ndi_cloudy)

                seg_loss = ce(seg_logits, y)

                use_rec = mask.sum().item() > 0.0
                rec_loss = masked_l1(ndi_pred, ndi_clean, mask) if use_rec else seg_loss.new_tensor(0.0)

                loss = W_SEG * seg_loss + W_REC * rec_loss

                va_iou_sum += calculate_iou(seg_logits, y).item()
                va_seg_sum += float(seg_loss.detach().cpu())
                va_total_sum += float(loss.detach().cpu())
                va_batches += 1

                if use_rec:
                    va_rec_sum += float(rec_loss.detach().cpu())
                    va_rec_batches += 1

                del x, y, ndi_cloudy, ndi_clean, mask
                del seg_logits, ndi_pred, seg_loss, rec_loss, loss

        val_iou = va_iou_sum / max(1, va_batches)
        val_seg = va_seg_sum / max(1, va_batches)
        val_rec = va_rec_sum / max(1, va_rec_batches) if va_rec_batches > 0 else 0.0
        val_total = va_total_sum / max(1, va_batches)

        hist_rows.append({
            "epoch": epoch + 1,
            "train_iou": train_iou,
            "val_iou": val_iou,
            "train_seg_loss": train_seg,
            "val_seg_loss": val_seg,
            "train_rec_loss": train_rec,
            "val_rec_loss": val_rec,
            "train_total_loss": train_total,
            "val_total_loss": val_total,
            "w_seg": W_SEG,
            "w_rec": W_REC,
        })

        hist_csv = RECON_CSV_ROOT / "training_history_lightsegformer_b0_recon_rec040.csv"

        with open(hist_csv, "w", newline="") as f:
            fieldnames = [
                "epoch",
                "train_iou",
                "val_iou",
                "train_seg_loss",
                "val_seg_loss",
                "train_rec_loss",
                "val_rec_loss",
                "train_total_loss",
                "val_total_loss",
                "w_seg",
                "w_rec",
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(hist_rows)

        print(
            f"\nStage 2 | Epoch {epoch + 1}/{num_epochs} | "
            f"Train IoU={train_iou:.4f} Val IoU={val_iou:.4f} | "
            f"Train Total={train_total:.4f} Val Total={val_total:.4f} | "
            f"Train Seg={train_seg:.4f} Val Seg={val_seg:.4f} | "
            f"Train Rec={train_rec:.4f} Val Rec={val_rec:.4f}"
        )

        if val_iou > best_val_iou + 1e-6:
            best_val_iou = val_iou
            best_epoch = epoch + 1
            bad_epochs = 0

            torch.save({
                "epoch": epoch + 1,
                "val_iou": best_val_iou,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "hist": hist_rows,
                "model": "LightSegFormerRecon",
                "source_simple_checkpoint": str(SIMPLE_SAVE_PATH),
                "num_classes": NUM_CLASSES,
                "in_channels": IN_CHANNELS,
                "w_seg": W_SEG,
                "w_rec": W_REC,
                "seed": BASE_SEED,
                "train_ratio": TRAIN_RATIO,
                "val_ratio": VAL_RATIO,
                "encoder_backbone_frozen": True,
                "frozen_modules": ["backbone"],
                "ndi_mode": "on_the_fly_no_disk_cache",
            }, best_path)

            print(f"✅ Saved best Stage 2 checkpoint to: {best_path}")

        else:
            bad_epochs += 1
            print(f"No Stage 2 validation IoU improvement: {bad_epochs}/{patience}")

            if bad_epochs >= patience:
                print("Stage 2 early stopping.")
                break

        free_gpu_memory()

    print("\nStage 2 training finished.")
    print("Best epoch:", best_epoch)
    print("Best validation IoU:", best_val_iou)
    print("Best checkpoint:", best_path)


# ============================================================
# 16) SIMPLE SELECTED-TEST EVALUATION HELPERS
# ============================================================

def confusion_matrix_update(cm, pred, target, num_classes=NUM_CLASSES):
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    valid = (
        (target != IGNORE_INDEX)
        & (target >= 0)
        & (target < num_classes)
        & (pred >= 0)
        & (pred < num_classes)
    )

    pred = pred[valid]
    target = target[valid]

    if target.size == 0:
        return cm

    idx = num_classes * target.astype(np.int64) + pred.astype(np.int64)
    bincount = np.bincount(idx, minlength=num_classes * num_classes)
    cm += bincount.reshape(num_classes, num_classes)

    return cm


def metrics_from_confusion(cm):
    ious = []
    f1s = []
    per_class = {}

    for cls in range(NUM_CLASSES):
        tp = float(cm[cls, cls])
        fp = float(cm[:, cls].sum() - cm[cls, cls])
        fn = float(cm[cls, :].sum() - cm[cls, cls])

        iou_denom = tp + fp + fn
        f1_denom = 2.0 * tp + fp + fn

        iou = tp / (iou_denom + 1e-6) if iou_denom > 0 else np.nan
        f1 = (2.0 * tp) / (f1_denom + 1e-6) if f1_denom > 0 else np.nan

        per_class[f"IoU_class_{cls}"] = float(iou) if not np.isnan(iou) else ""
        per_class[f"F1_class_{cls}"] = float(f1) if not np.isnan(f1) else ""

        if not np.isnan(iou):
            ious.append(iou)
        if not np.isnan(f1):
            f1s.append(f1)

    total = cm.sum()
    correct = np.trace(cm)
    pixel_acc = correct / (total + 1e-6) if total > 0 else np.nan

    return {
        "mIoU": float(np.mean(ious)) if len(ious) > 0 else "",
        "macro_F1": float(np.mean(f1s)) if len(f1s) > 0 else "",
        "pixel_accuracy": float(pixel_acc) if not np.isnan(pixel_acc) else "",
        **per_class,
    }


def save_prediction_mask(out_path, pred_mask, reference_path):
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()

    profile.update(
        count=1,
        dtype="uint8",
        compress="deflate",
        nodata=IGNORE_INDEX,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(pred_mask.astype("uint8"), 1)


def write_seg_test_csv(prefix, rows, cm_by_percentage, cm_overall, csv_root):
    class_fields = []
    for i in range(NUM_CLASSES):
        class_fields.extend([f"IoU_class_{i}", f"F1_class_{i}"])

    per_sample_csv = csv_root / f"{prefix}_selected_test_per_sample.csv"

    with open(per_sample_csv, "w", newline="") as f:
        fieldnames = [
            "cloud_percentage",
            "sample_key",
            "roi_id",
            "prediction_path",
            "mIoU",
            "macro_F1",
            "pixel_accuracy",
            *class_fields,
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    summary_rows = []

    for percentage_name, cm in sorted(cm_by_percentage.items()):
        summary_rows.append({
            "cloud_percentage": percentage_name,
            "samples": sum(1 for r in rows if r["cloud_percentage"] == percentage_name),
            **metrics_from_confusion(cm),
        })

    summary_rows.append({
        "cloud_percentage": "ALL",
        "samples": len(rows),
        **metrics_from_confusion(cm_overall),
    })

    summary_csv = csv_root / f"{prefix}_selected_test_summary_by_cloud_percentage.csv"

    with open(summary_csv, "w", newline="") as f:
        fieldnames = [
            "cloud_percentage",
            "samples",
            "mIoU",
            "macro_F1",
            "pixel_accuracy",
            *class_fields,
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print("\nSelected test summary:", prefix)
    for row in summary_rows:
        print(
            f"{row['cloud_percentage']:10s} | "
            f"n={row['samples']:4d} | "
            f"mIoU={row['mIoU']} | "
            f"macro_F1={row['macro_F1']} | "
            f"pixel_acc={row['pixel_accuracy']}"
        )

    print("Saved per-sample CSV:", per_sample_csv)
    print("Saved summary CSV:", summary_csv)


def evaluate_simple_selected_test(model, device):
    dataset = LuojiaSegFormerDataset(test_samples, return_index=True)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    cm_overall = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    cm_by_percentage = {}
    rows = []

    with torch.no_grad():
        for x, y, indices in tqdm(loader, desc="Stage 1 selected test eval"):
            x = x.to(device, non_blocking=True)
            logits = model(x)

            preds = torch.argmax(logits, dim=1).cpu().numpy().astype("uint8")
            labels_np = y.cpu().numpy().astype("int64")
            indices_np = indices.cpu().numpy().tolist()

            for b in range(preds.shape[0]):
                item = test_samples[indices_np[b]]
                percentage_name = item["percentage_name"]
                sample_key = item["sample_key"]

                if percentage_name not in cm_by_percentage:
                    cm_by_percentage[percentage_name] = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

                cm_sample = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
                cm_sample = confusion_matrix_update(cm_sample, preds[b], labels_np[b], NUM_CLASSES)
                cm_overall = confusion_matrix_update(cm_overall, preds[b], labels_np[b], NUM_CLASSES)
                cm_by_percentage[percentage_name] = confusion_matrix_update(
                    cm_by_percentage[percentage_name],
                    preds[b],
                    labels_np[b],
                    NUM_CLASSES,
                )

                pred_path = ""

                if SAVE_TEST_PREDICTIONS:
                    pred_path = (
                        SIMPLE_PRED_ROOT
                        / percentage_name
                        / "pred_masks"
                        / f"lightsegformer_b0_simple_{safe_name(percentage_name)}_{safe_name(sample_key)}.tif"
                    )
                    save_prediction_mask(pred_path, preds[b], item["label_path"])
                    pred_path = str(pred_path)

                rows.append({
                    "cloud_percentage": percentage_name,
                    "sample_key": sample_key,
                    "roi_id": item["roi_id"],
                    "prediction_path": pred_path,
                    **metrics_from_confusion(cm_sample),
                })

    write_seg_test_csv("lightsegformer_b0_simple", rows, cm_by_percentage, cm_overall, SIMPLE_CSV_ROOT)


def evaluate_recon_selected_test(model, device):
    dataset = LuojiaSegFormerMultitaskNDIDataset(test_samples, return_index=True)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    cm_overall = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    cm_by_percentage = {}
    rows = []

    with torch.no_grad():
        for x, y, ndi_cloudy, ndi_clean, mask, indices in tqdm(loader, desc="Stage 2 selected test eval"):
            x = x.to(device, non_blocking=True)
            ndi_cloudy = ndi_cloudy.to(device, non_blocking=True)

            seg_logits, _ = model(x, ndi_cloudy)

            preds = torch.argmax(seg_logits, dim=1).cpu().numpy().astype("uint8")
            labels_np = y.cpu().numpy().astype("int64")
            indices_np = indices.cpu().numpy().tolist()

            for b in range(preds.shape[0]):
                item = test_samples[indices_np[b]]
                percentage_name = item["percentage_name"]
                sample_key = item["sample_key"]

                if percentage_name not in cm_by_percentage:
                    cm_by_percentage[percentage_name] = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

                cm_sample = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
                cm_sample = confusion_matrix_update(cm_sample, preds[b], labels_np[b], NUM_CLASSES)
                cm_overall = confusion_matrix_update(cm_overall, preds[b], labels_np[b], NUM_CLASSES)
                cm_by_percentage[percentage_name] = confusion_matrix_update(
                    cm_by_percentage[percentage_name],
                    preds[b],
                    labels_np[b],
                    NUM_CLASSES,
                )

                pred_path = ""

                if SAVE_TEST_PREDICTIONS:
                    pred_path = (
                        RECON_PRED_ROOT
                        / percentage_name
                        / "pred_masks"
                        / f"lightsegformer_b0_recon_rec040_{safe_name(percentage_name)}_{safe_name(sample_key)}.tif"
                    )
                    save_prediction_mask(pred_path, preds[b], item["label_path"])
                    pred_path = str(pred_path)

                rows.append({
                    "cloud_percentage": percentage_name,
                    "sample_key": sample_key,
                    "roi_id": item["roi_id"],
                    "prediction_path": pred_path,
                    **metrics_from_confusion(cm_sample),
                })

    write_seg_test_csv("lightsegformer_b0_recon_rec040", rows, cm_by_percentage, cm_overall, RECON_CSV_ROOT)


# ============================================================
# 17) RUNNERS
# ============================================================

def run_stage1_simple_segformer():
    print("\n" + "#" * 80)
    print("STAGE 1: SIMPLE LIGHT SEGFORMER-B0 SEGMENTATION")
    print("#" * 80)

    set_seed_everywhere(BASE_SEED)
    cleanup_gpu_cpu_resources("before Stage 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Stage 1 checkpoint:", SIMPLE_SAVE_PATH)

    train_loader, val_loader = make_stage1_loaders()

    model = LightSegFormerCloud(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
    )

    total_params, trainable_params = count_parameters(model)
    print("\nModel parameter count:")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    train_simple_segformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=NUM_EPOCHS_STAGE1,
        lr=LR_STAGE1,
        save_path=SIMPLE_SAVE_PATH,
        patience=PATIENCE_STAGE1,
    )

    if RUN_STAGE_1_TEST_EVAL:
        print("\nLoading best Stage 1 checkpoint for selected test evaluation...")
        ckpt = torch.load(SIMPLE_SAVE_PATH, map_location=device)
        model.load_state_dict(extract_state_dict(ckpt))
        model.to(device)
        model.eval()
        evaluate_simple_selected_test(model, device)

    del model
    del train_loader
    del val_loader
    cleanup_gpu_cpu_resources("after Stage 1")


def run_stage2_reconstruction_from_simple():
    print("\n" + "#" * 80)
    print("STAGE 2: LIGHT SEGFORMER-B0 + NDI RECONSTRUCTION FROM SIMPLE CHECKPOINT")
    print("#" * 80)

    if not SIMPLE_SAVE_PATH.exists():
        raise FileNotFoundError(
            "Stage 1 checkpoint is missing. Run Stage 1 first:\n"
            f"  {SIMPLE_SAVE_PATH}"
        )

    set_seed_everywhere(BASE_SEED)
    cleanup_gpu_cpu_resources("before Stage 2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Stage 1 source checkpoint:", SIMPLE_SAVE_PATH)
    print("Stage 2 best checkpoint:", RECON_BEST_PATH)
    print(f"Loss weights: W_SEG={W_SEG}, W_REC={W_REC}")
    print("NDI mode: on-the-fly, no disk cache")

    train_loader, val_loader = make_stage2_loaders()

    model = LightSegFormerRecon(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
    )

    print("\nLoading saved Stage 1 simple Light SegFormer weights...")
    load_simple_segformer_into_multitask(model, SIMPLE_SAVE_PATH)

    print("\nFreezing Light SegFormer encoder for Stage 2...")
    freeze_segformer_encoder(model)

    total_params, trainable_params = count_parameters(model)
    print("\nModel parameter count after freezing:")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    train_segformer_reconstruction(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=NUM_EPOCHS_STAGE2,
        lr=LR_STAGE2,
        best_path=RECON_BEST_PATH,
        patience=PATIENCE_STAGE2,
    )

    if RUN_STAGE_2_TEST_EVAL:
        print("\nLoading best Stage 2 checkpoint for selected test evaluation...")
        ckpt = torch.load(RECON_BEST_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        evaluate_recon_selected_test(model, device)

    del model
    del train_loader
    del val_loader
    cleanup_gpu_cpu_resources("after Stage 2")



# ============================================================
# 18) DEEPLABV3 RECONSTRUCTION-ONLY STAGE, NO NDI CACHE
# ============================================================
# This section is intentionally cache-free. It reuses the already collected
# Luojia samples, computes NDI on the fly in __getitem__, and never writes an
# ndi_cache directory. This avoids filling disk storage when running the
# combined pipeline multiple times.

from reben_publication.BigEarthNetv2_0_ImageClassifier import (
    BigEarthNetv2_0_ImageClassifier
)

DEEPLAB_SEED = 0
DEEPLAB_OUT_ROOT = EXT_ROOT / "luojia_deeplabv3_reconstruction_only_ndi_rec040_on_the_fly_patchsplit_70_30"
DEEPLAB_CSV_ROOT = DEEPLAB_OUT_ROOT / "csv"
DEEPLAB_PRED_ROOT = DEEPLAB_OUT_ROOT / "test_predictions"
DEEPLAB_BEST_PATH = DEEPLAB_OUT_ROOT / "best_deeplabv3_reconstruction_only_ndi_rec040_from_segmentation_freeze_encoder_no_cache.pth"

DEEPLAB_SEGMENTATION_CKPT_CANDIDATES = [
    EXT_ROOT / "luojia_deeplab_segmentation_seed0_patchsplit_70_30" / "deeplabv3_luojia_seed0_patchsplit_70_30.pth",
    EXT_ROOT / "luojia_deeplab_segmentation_seed0" / "deeplabv3_luojia_seed0.pth",
    DATA_ROOT / "deeplabv3_luojia_seed0_patchsplit_70_30.pth",
    DATA_ROOT / "deeplabv3_luojia_seed0.pth",
]

DEEPLAB_NUM_EPOCHS = 200
DEEPLAB_LR = 1e-3
DEEPLAB_PATIENCE = 20

RUN_DEEPLAB_RECON_FROM_EXISTING_SEG = True
RUN_DEEPLAB_SELECTED_TEST_EVAL = False

for _p in [DEEPLAB_OUT_ROOT, DEEPLAB_CSV_ROOT, DEEPLAB_PRED_ROOT]:
    _p.mkdir(parents=True, exist_ok=True)


def resolve_deeplab_segmentation_checkpoint():
    for p in DEEPLAB_SEGMENTATION_CKPT_CANDIDATES:
        if p.exists():
            return p

    msg = "Could not find saved DeepLab segmentation checkpoint. Checked:\n"
    for p in DEEPLAB_SEGMENTATION_CKPT_CANDIDATES:
        msg += f"  {p}\n"
    raise FileNotFoundError(msg)


def prepare_deeplab_split():
    """
    DeepLab reconstruction uses the same selected-test exclusion protocol but
    preserves the DeepLab seed-0 patch split, because the existing DeepLab
    segmentation checkpoint was trained with seed 0.
    """
    deeplab_train_samples, deeplab_val_samples, deeplab_train_keys, deeplab_val_keys = split_train_val_by_patch_key(
        trainval_samples,
        seed=DEEPLAB_SEED,
        train_ratio=TRAIN_RATIO,
    )

    train_keys = set(s["sample_key"] for s in deeplab_train_samples)
    val_keys = set(s["sample_key"] for s in deeplab_val_samples)
    selected_test_keys = set(s["sample_key"] for s in test_samples)

    print("\nDeepLab patch-level train/validation split:")
    print("DeepLab train unique patch keys:", len(deeplab_train_keys))
    print("DeepLab val unique patch keys:", len(deeplab_val_keys))
    print("DeepLab train samples:", len(deeplab_train_samples))
    print("DeepLab val samples:", len(deeplab_val_samples))
    print("DeepLab train/test overlap patch keys:", len(train_keys & selected_test_keys))
    print("DeepLab val/test overlap patch keys:", len(val_keys & selected_test_keys))
    print("DeepLab train/val overlap patch keys:", len(train_keys & val_keys))

    if train_keys & selected_test_keys:
        raise RuntimeError("Leakage detected: DeepLab train patches appear in selected test.")
    if val_keys & selected_test_keys:
        raise RuntimeError("Leakage detected: DeepLab val patches appear in selected test.")
    if train_keys & val_keys:
        raise RuntimeError("Leakage detected: same patch appears in DeepLab train and validation.")

    split_csv = DEEPLAB_CSV_ROOT / "split_manifest_deeplab_recon_seed0.csv"
    with open(split_csv, "w", newline="") as f:
        fieldnames = [
            "split",
            "percentage_name",
            "sample_key",
            "roi_id",
            "s1_path",
            "clean_s2_path",
            "cloudy_s2_path",
            "cloud_mask_path",
            "label_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for split_name, rows in [
            ("train", deeplab_train_samples),
            ("val", deeplab_val_samples),
            ("test_selected", test_samples),
        ]:
            for s in rows:
                writer.writerow({
                    "split": split_name,
                    "percentage_name": s["percentage_name"],
                    "sample_key": s["sample_key"],
                    "roi_id": s["roi_id"],
                    "s1_path": str(s["s1_path"]),
                    "clean_s2_path": str(s["clean_s2_path"]),
                    "cloudy_s2_path": str(s["cloudy_s2_path"]),
                    "cloud_mask_path": "" if s["cloud_mask_path"] is None else str(s["cloud_mask_path"]),
                    "label_path": str(s["label_path"]),
                })

    print("Saved DeepLab split manifest:", split_csv)
    print("DeepLab NDI mode: on-the-fly, no disk cache")

    return deeplab_train_samples, deeplab_val_samples


class LuojiaDeepLabMultitaskNDIOnTheFlyDataset(Dataset):
    """
    DeepLab reconstruction dataset without disk cache:
      input: S1 + cloudy S2
      auxiliary input: cloudy NDI computed from cloudy S2
      targets: segmentation label + clean NDI computed from clean S2
    """
    def __init__(self, samples, return_index=False):
        self.samples = samples
        self.return_index = return_index
        self.mean = MEAN_VALS
        self.std = STD_VALS

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        with rasterio.open(item["s1_path"]) as src1:
            s1 = src1.read(S1_BANDS).astype("float32")

        with rasterio.open(item["cloudy_s2_path"]) as src2:
            s2_cloudy_input = src2.read(S2_INPUT_BANDS).astype("float32")

        x = torch.from_numpy(
            np.concatenate([s1, s2_cloudy_input], axis=0)
        ).float()
        x = (x - self.mean) / self.std

        ndi_cloudy = torch.from_numpy(
            compute_ndi_from_s2_file(item["cloudy_s2_path"])
        ).float()
        ndi_clean = torch.from_numpy(
            compute_ndi_from_s2_file(item["clean_s2_path"])
        ).float()

        h, w = ndi_clean.shape[1], ndi_clean.shape[2]
        mask_np = read_cloud_mask_for_loss(item["cloud_mask_path"], shape=(h, w))
        mask = torch.from_numpy(mask_np).float().unsqueeze(0)

        with rasterio.open(item["label_path"]) as src:
            label = src.read(1).astype("int64")

        valid = (label >= 0) & (label < NUM_CLASSES)
        clean_label = np.full(label.shape, IGNORE_INDEX, dtype=np.int64)
        clean_label[valid] = label[valid]
        y = torch.from_numpy(clean_label).long()

        if self.return_index:
            return x, y, ndi_cloudy, ndi_clean, mask, idx

        return x, y, ndi_cloudy, ndi_clean, mask


class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(6, 12, 18)):
        super().__init__()

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        ])

        for r in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = [b(x) for b in self.branches]

        gp = self.global_pool(x)
        gp = F.interpolate(
            gp,
            size=(h, w),
            mode="bilinear",
            align_corners=True,
        )
        feats.append(gp)

        return self.project(torch.cat(feats, dim=1))


class MultiTaskDeepLabV3(nn.Module):
    def __init__(self, pretrained, num_classes=NUM_CLASSES):
        super().__init__()

        enc = pretrained.model.vision_encoder

        self.conv1 = enc.conv1
        self.bn1 = enc.bn1
        self.act1 = enc.act1
        self.maxpool = enc.maxpool

        self.layer1 = enc.layer1
        self.layer2 = enc.layer2
        self.layer3 = enc.layer3
        self.layer4 = enc.layer4

        self.aspp = AtrousSpatialPyramidPooling(2048, 256)

        self.r64 = nn.Sequential(
            CBR(256, 128),
            CBR(128, 128),
        )

        self.inject = nn.Sequential(
            CBR(128 + 3 + 256, 128, k=1, p=0),
        )

        self.r256 = nn.Sequential(
            CBR(128, 64),
            CBR(64, 32),
        )

        self.r_out = nn.Sequential(
            nn.Conv2d(32, 3, 1),
            nn.Tanh(),
        )

        self.seg_pre = CBR(256, 256)
        self.seg_fuse = CBR(256 + 32, 256, k=1, p=0)
        self.seg_out = nn.Conv2d(256, num_classes, 1)

    def forward(self, x, ndi_cloudy):
        input_size = x.shape[-2:]

        x1 = self.act1(self.bn1(self.conv1(x)))
        x1 = self.maxpool(x1)

        l1 = self.layer1(x1)
        x2 = self.layer2(l1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        f = self.aspp(x4)

        r = F.interpolate(
            f,
            size=l1.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        r = self.r64(r)

        ndi_low = F.interpolate(
            ndi_cloudy,
            size=l1.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

        r = self.inject(torch.cat([r, ndi_low, l1], dim=1))

        r = F.interpolate(
            r,
            size=input_size,
            mode="bilinear",
            align_corners=True,
        )

        r_feat = self.r256(r)
        ndi_pred = self.r_out(r_feat)

        s = F.interpolate(
            self.seg_pre(f),
            size=input_size,
            mode="bilinear",
            align_corners=True,
        )

        s = self.seg_fuse(torch.cat([s, r_feat.detach()], dim=1))
        seg_logits = self.seg_out(s)

        return seg_logits, ndi_pred


def load_luojia_deeplab_seg_checkpoint_with_remap(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = extract_state_dict(ckpt)
    sd = strip_prefix_if_present(sd, "module.")

    idx_to_name = {
        "0": "conv1",
        "1": "bn1",
        "4": "layer1",
        "5": "layer2",
        "6": "layer3",
        "7": "layer4",
    }

    model_sd = model.state_dict()
    new_sd = {}
    skipped = []

    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            new_sd[k] = v
            continue

        if k.startswith("encoder_layers."):
            parts = k.split(".")
            idx = parts[1]
            if idx in idx_to_name:
                new_key = ".".join([idx_to_name[idx]] + parts[2:])
                if new_key in model_sd and model_sd[new_key].shape == v.shape:
                    new_sd[new_key] = v
                else:
                    skipped.append(k)
            else:
                skipped.append(k)
            continue

        if k.startswith("encoder."):
            parts = k.split(".")
            idx = parts[1]
            if idx in idx_to_name:
                new_key = ".".join([idx_to_name[idx]] + parts[2:])
                if new_key in model_sd and model_sd[new_key].shape == v.shape:
                    new_sd[new_key] = v
                else:
                    skipped.append(k)
            else:
                skipped.append(k)
            continue

        if k.startswith("aspp.") and k in model_sd and model_sd[k].shape == v.shape:
            new_sd[k] = v
            continue

        if k.startswith("classifier.0."):
            new_key = k.replace("classifier.0.", "seg_pre.0.")
            if new_key in model_sd and model_sd[new_key].shape == v.shape:
                new_sd[new_key] = v
            else:
                skipped.append(k)
            continue

        if k.startswith("classifier.1."):
            new_key = k.replace("classifier.1.", "seg_pre.1.")
            if new_key in model_sd and model_sd[new_key].shape == v.shape:
                new_sd[new_key] = v
            else:
                skipped.append(k)
            continue

        if k.startswith("classifier.4."):
            new_key = k.replace("classifier.4.", "seg_out.")
            if new_key in model_sd and model_sd[new_key].shape == v.shape:
                new_sd[new_key] = v
            else:
                skipped.append(k)
            continue

        skipped.append(k)

    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    print("\nLoaded DeepLab segmentation checkpoint with remap:")
    print("Checkpoint:", ckpt_path)
    print("Loaded keys:", len(new_sd))
    print("Skipped keys:", len(skipped))
    print("Missing keys are expected for reconstruction branch and seg_fuse.")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))
    if skipped:
        print("First skipped keys:", skipped[:20])


def freeze_deeplab_encoder(model):
    for module in [
        model.conv1,
        model.bn1,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    ]:
        for p in module.parameters():
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print("\nDeepLab encoder/backbone frozen.")
    print("Frozen modules: conv1, bn1, layer1, layer2, layer3, layer4")
    print("Trainable parameters:", trainable)
    print("Frozen parameters:", frozen)


def make_deeplab_recon_loaders(deeplab_train_samples, deeplab_val_samples):
    worker_init_fn = seed_worker_factory(DEEPLAB_SEED)

    train_dataset = LuojiaDeepLabMultitaskNDIOnTheFlyDataset(deeplab_train_samples)
    val_dataset = LuojiaDeepLabMultitaskNDIOnTheFlyDataset(deeplab_val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(DEEPLAB_SEED),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(DEEPLAB_SEED),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=False,
    )

    return train_loader, val_loader


def train_deeplab_reconstruction(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=DEEPLAB_NUM_EPOCHS,
    lr=DEEPLAB_LR,
    best_path=DEEPLAB_BEST_PATH,
    patience=DEEPLAB_PATIENCE,
):
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    model.to(device)

    best_val_iou = -1.0
    best_epoch = 0
    bad_epochs = 0
    hist_rows = []

    for epoch in range(num_epochs):
        model.train()

        tr_iou_sum = 0.0
        tr_seg_sum = 0.0
        tr_rec_sum = 0.0
        tr_total_sum = 0.0
        tr_batches = 0
        tr_rec_batches = 0

        for x, y, ndi_cloudy, ndi_clean, mask in tqdm(
            train_loader,
            desc=f"DeepLab reconstruction | Epoch {epoch + 1} train",
        ):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            ndi_cloudy = ndi_cloudy.to(device, non_blocking=True)
            ndi_clean = ndi_clean.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            seg_logits, ndi_pred = model(x, ndi_cloudy)
            seg_loss = ce(seg_logits, y)

            use_rec = mask.sum().item() > 0.0
            rec_loss = masked_l1(ndi_pred, ndi_clean, mask) if use_rec else seg_loss.new_tensor(0.0)
            loss = W_SEG * seg_loss + W_REC * rec_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            tr_iou_sum += calculate_iou(seg_logits.detach(), y).item()
            tr_seg_sum += float(seg_loss.detach().cpu())
            tr_total_sum += float(loss.detach().cpu())
            tr_batches += 1

            if use_rec:
                tr_rec_sum += float(rec_loss.detach().cpu())
                tr_rec_batches += 1

            del x, y, ndi_cloudy, ndi_clean, mask, seg_logits, ndi_pred, seg_loss, rec_loss, loss

        train_iou = tr_iou_sum / max(1, tr_batches)
        train_seg = tr_seg_sum / max(1, tr_batches)
        train_rec = tr_rec_sum / max(1, tr_rec_batches) if tr_rec_batches > 0 else 0.0
        train_total = tr_total_sum / max(1, tr_batches)

        model.eval()

        va_iou_sum = 0.0
        va_seg_sum = 0.0
        va_rec_sum = 0.0
        va_total_sum = 0.0
        va_batches = 0
        va_rec_batches = 0

        with torch.no_grad():
            for x, y, ndi_cloudy, ndi_clean, mask in tqdm(
                val_loader,
                desc=f"DeepLab reconstruction | Epoch {epoch + 1} val",
            ):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                ndi_cloudy = ndi_cloudy.to(device, non_blocking=True)
                ndi_clean = ndi_clean.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                seg_logits, ndi_pred = model(x, ndi_cloudy)
                seg_loss = ce(seg_logits, y)

                use_rec = mask.sum().item() > 0.0
                rec_loss = masked_l1(ndi_pred, ndi_clean, mask) if use_rec else seg_loss.new_tensor(0.0)
                loss = W_SEG * seg_loss + W_REC * rec_loss

                va_iou_sum += calculate_iou(seg_logits, y).item()
                va_seg_sum += float(seg_loss.detach().cpu())
                va_total_sum += float(loss.detach().cpu())
                va_batches += 1

                if use_rec:
                    va_rec_sum += float(rec_loss.detach().cpu())
                    va_rec_batches += 1

                del x, y, ndi_cloudy, ndi_clean, mask, seg_logits, ndi_pred, seg_loss, rec_loss, loss

        val_iou = va_iou_sum / max(1, va_batches)
        val_seg = va_seg_sum / max(1, va_batches)
        val_rec = va_rec_sum / max(1, va_rec_batches) if va_rec_batches > 0 else 0.0
        val_total = va_total_sum / max(1, va_batches)

        hist_rows.append({
            "epoch": epoch + 1,
            "train_iou": train_iou,
            "val_iou": val_iou,
            "train_seg_loss": train_seg,
            "val_seg_loss": val_seg,
            "train_rec_loss": train_rec,
            "val_rec_loss": val_rec,
            "train_total_loss": train_total,
            "val_total_loss": val_total,
            "w_seg": W_SEG,
            "w_rec": W_REC,
            "ndi_mode": "on_the_fly_no_cache",
        })

        hist_csv = DEEPLAB_CSV_ROOT / "training_history_deeplabv3_recon_ndi_rec040_no_cache.csv"
        with open(hist_csv, "w", newline="") as f:
            fieldnames = [
                "epoch",
                "train_iou",
                "val_iou",
                "train_seg_loss",
                "val_seg_loss",
                "train_rec_loss",
                "val_rec_loss",
                "train_total_loss",
                "val_total_loss",
                "w_seg",
                "w_rec",
                "ndi_mode",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(hist_rows)

        print(
            f"\nDeepLab reconstruction | Epoch {epoch + 1}/{num_epochs} | "
            f"Train IoU={train_iou:.4f} Val IoU={val_iou:.4f} | "
            f"Train Total={train_total:.4f} Val Total={val_total:.4f} | "
            f"Train Seg={train_seg:.4f} Val Seg={val_seg:.4f} | "
            f"Train Rec={train_rec:.4f} Val Rec={val_rec:.4f}"
        )

        if val_iou > best_val_iou + 1e-6:
            best_val_iou = val_iou
            best_epoch = epoch + 1
            bad_epochs = 0

            torch.save({
                "epoch": epoch + 1,
                "val_iou": best_val_iou,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "w_seg": W_SEG,
                "w_rec": W_REC,
                "seed": DEEPLAB_SEED,
                "train_ratio": TRAIN_RATIO,
                "val_ratio": VAL_RATIO,
                "uses_cached_ndi": False,
                "ndi_mode": "on_the_fly_no_cache",
                "stage": "deeplabv3_reconstruction_from_existing_segmentation",
            }, best_path)

            print(f" Saved best DeepLab reconstruction checkpoint to: {best_path}")

        else:
            bad_epochs += 1
            print(f"No DeepLab validation IoU improvement: {bad_epochs}/{patience}")
            if bad_epochs >= patience:
                print("DeepLab reconstruction early stopping.")
                break

        free_gpu_memory()

    print("\nDeepLab reconstruction training finished.")
    print("Best epoch:", best_epoch)
    print("Best validation IoU:", best_val_iou)
    print("Best checkpoint:", best_path)


def run_deeplab_reconstruction_from_existing_segmentation():
    print("\n" + "#" * 80)
    print("STAGE 3: DEEPLABV3 + NDI RECONSTRUCTION FROM EXISTING SEGMENTATION CHECKPOINT")
    print("#" * 80)

    set_seed_everywhere(DEEPLAB_SEED)
    cleanup_gpu_cpu_resources("before DeepLab reconstruction")

    ckpt_path = resolve_deeplab_segmentation_checkpoint()
    print("DeepLab segmentation checkpoint:", ckpt_path)
    print("DeepLab reconstruction checkpoint:", DEEPLAB_BEST_PATH)
    print(f"Loss weights: W_SEG={W_SEG}, W_REC={W_REC}")
    print("DeepLab NDI mode: on-the-fly, no disk cache")

    deeplab_train_samples, deeplab_val_samples = prepare_deeplab_split()
    train_loader, val_loader = make_deeplab_recon_loaders(
        deeplab_train_samples,
        deeplab_val_samples,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("\nLoading BigEarthNet pretrained encoder...")
    pretrained = BigEarthNetv2_0_ImageClassifier.from_pretrained(
        "BIFOLD-BigEarthNetv2-0/resnet50-all-v0.2.0"
    )

    print("\nBuilding DeepLabV3 multitask reconstruction model...")
    model = MultiTaskDeepLabV3(
        pretrained,
        num_classes=NUM_CLASSES,
    )

    print("\nLoading saved DeepLab segmentation weights...")
    load_luojia_deeplab_seg_checkpoint_with_remap(
        model,
        ckpt_path,
    )

    print("\nFreezing DeepLab encoder for reconstruction training...")
    freeze_deeplab_encoder(model)

    total_params, trainable_params = count_parameters(model)
    print("\nDeepLab parameter count after freezing:")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    train_deeplab_reconstruction(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=DEEPLAB_NUM_EPOCHS,
        lr=DEEPLAB_LR,
        best_path=DEEPLAB_BEST_PATH,
        patience=DEEPLAB_PATIENCE,
    )

    # Selected-test evaluation is intentionally disabled by default to avoid
    # extra disk usage from predictions. Enable RUN_DEEPLAB_SELECTED_TEST_EVAL
    # only if you really need it.
    if RUN_DEEPLAB_SELECTED_TEST_EVAL:
        print("\nRUN_DEEPLAB_SELECTED_TEST_EVAL=True, but test prediction saving is not implemented in this no-cache combined script.")
        print("Skipping to avoid accidental disk growth.")

    del model
    del pretrained
    del train_loader
    del val_loader
    cleanup_gpu_cpu_resources("after DeepLab reconstruction")



if __name__ == "__main__":
    print("=" * 80)
    print("Combined Luojia pipeline")
    print("1) Light SegFormer-B0 segmentation-only")
    print("2) Light SegFormer-B0 + NDI reconstruction from Stage 1 checkpoint")
    print("3) DeepLabV3 + NDI reconstruction from existing DeepLab segmentation checkpoint")
    print("=" * 80)
    print("SegFormer Stage 1 checkpoint:", SIMPLE_SAVE_PATH)
    print("SegFormer Stage 2 checkpoint:", RECON_BEST_PATH)
    print("DeepLab reconstruction checkpoint:", DEEPLAB_BEST_PATH)
    print(f"SegFormer seed: {BASE_SEED}")
    print(f"DeepLab seed: {DEEPLAB_SEED}")
    print(f"Fixed reconstruction weights: W_SEG={W_SEG}, W_REC={W_REC}")
    print("NDI mode for reconstruction stages: on-the-fly, no disk cache")
    print("No NDI cache folders are created by this combined script.")
    print("=" * 80)

    if RUN_STAGE_1_SIMPLE_SEGFORMER:
        run_stage1_simple_segformer()
    else:
        print("\nSkipping SegFormer Stage 1 because RUN_STAGE_1_SIMPLE_SEGFORMER = False")

    cleanup_gpu_cpu_resources("between SegFormer Stage 1 and Stage 2")

    if RUN_STAGE_2_RECON_FROM_SIMPLE:
        run_stage2_reconstruction_from_simple()
    else:
        print("\nSkipping SegFormer Stage 2 because RUN_STAGE_2_RECON_FROM_SIMPLE = False")

    cleanup_gpu_cpu_resources("between SegFormer Stage 2 and DeepLab")

    if RUN_DEEPLAB_RECON_FROM_EXISTING_SEG:
        run_deeplab_reconstruction_from_existing_segmentation()
    else:
        print("\nSkipping DeepLab reconstruction because RUN_DEEPLAB_RECON_FROM_EXISTING_SEG = False")

    cleanup_gpu_cpu_resources("final cleanup")

    print("\nDONE.")
    print("SegFormer Stage 1 checkpoint:", SIMPLE_SAVE_PATH)
    print("SegFormer Stage 2 checkpoint:", RECON_BEST_PATH)
    print("DeepLab reconstruction checkpoint:", DEEPLAB_BEST_PATH)
    print("NDI cache: NOT USED")
