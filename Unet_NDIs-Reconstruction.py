import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
DATA_ROOT = Path('')
EXT_ROOT = Path('')
FULL_LUOJIA_ROOT = DATA_ROOT / 'LuojiaSET-OSFCR'
SELECTED_TEST_ROOT = EXT_ROOT / 'LuojiaSET_OSFCR_selected_100'
SIMPLE_OUT_ROOT = EXT_ROOT / 'luojia_unet_segmentation_seed0_patchsplit_70_30'
SIMPLE_CSV_ROOT = SIMPLE_OUT_ROOT / 'csv'
SIMPLE_PRED_ROOT = SIMPLE_OUT_ROOT / 'test_predictions'
SIMPLE_SAVE_PATH = SIMPLE_OUT_ROOT / 'unet_resnet50_luojia_seed0_patchsplit_70_30.pth'
RECON_OUT_ROOT = EXT_ROOT / 'luojia_unet_multitask_ndi_rec040_from_simple_patchsplit_70_30_safe_v2'
RECON_CSV_ROOT = RECON_OUT_ROOT / 'csv'
RECON_PRED_ROOT = RECON_OUT_ROOT / 'test_predictions'
RECON_BEST_PATH = RECON_OUT_ROOT / 'best_unet_multitask_ndi_rec040_from_simple_safe_v2.pth'
for p in [SIMPLE_OUT_ROOT, SIMPLE_CSV_ROOT, SIMPLE_PRED_ROOT, RECON_OUT_ROOT, RECON_CSV_ROOT, RECON_PRED_ROOT]:
    p.mkdir(parents=True, exist_ok=True)
TARGET_PERCENTAGE_RANGES = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
BASE_SEED = 0
IN_CHANNELS = 12
NUM_CLASSES = 6
IGNORE_INDEX = 255
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS_STAGE1 = 200
NUM_EPOCHS_STAGE2 = 200
LR_STAGE1 = 0.001
LR_STAGE2 = 0.001
PATIENCE_STAGE1 = 20
PATIENCE_STAGE2 = 20
TRAIN_RATIO = 0.7
VAL_RATIO = 0.3
W_REC = 0.4
W_SEG = 0.6
USE_STRICT_DETERMINISM = True
S1_BANDS = [1, 2]
S2_INPUT_BANDS = [2, 3, 4, 5, 6, 7, 8, 9, 12, 13]
B3 = 3
B4 = 4
B8 = 8
B11 = 11
MEAN_VALS = torch.tensor([-12.64, -19.35, 438.37, 614.05, 588.4, 942.84, 1769.93, 2049.55, 2193.29, 2235.55, 1568.22, 997.73], dtype=torch.float32).view(12, 1, 1)
STD_VALS = torch.tensor([5.13, 5.59, 607.02, 603.29, 684.56, 738.43, 1100.45, 1275.8, 1369.37, 1356.54, 1070.16, 813.52], dtype=torch.float32).view(12, 1, 1)
RUN_STAGE_1_SIMPLE_UNET = True
RUN_STAGE_2_RECON_FROM_SIMPLE = True
RUN_STAGE_1_TEST_EVAL = False
RUN_STAGE_2_TEST_EVAL = False
SAVE_TEST_PREDICTIONS = False

def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def cleanup_gpu_cpu_resources(note=''):
    print('\n' + '=' * 80)
    print('Cleaning GPU/CPU resources', f'- {note}' if note else '')
    print('=' * 80)
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print('CUDA cleanup warning:', e)
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        print(f'CUDA memory allocated: {allocated:.2f} MB')
        print(f'CUDA memory reserved:  {reserved:.2f} MB')
    else:
        print('CUDA is not available.')
    print('Cleanup done.\n')

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
        raise FileNotFoundError(f'Missing full Luojia root: {FULL_LUOJIA_ROOT}')
    if not SELECTED_TEST_ROOT.exists():
        raise FileNotFoundError(f'Missing selected test root: {SELECTED_TEST_ROOT}')
    print('Full Luojia root:', FULL_LUOJIA_ROOT)
    print('Selected test root:', SELECTED_TEST_ROOT)
    print('Simple UNet output root:', SIMPLE_OUT_ROOT)
    print('Reconstruction UNet output root:', RECON_OUT_ROOT)
    print('NDI mode: on-the-fly, no disk cache')
check_paths()

def safe_name(text):
    return str(text).replace('%', 'pct').replace('-', '_').replace('/', '_').replace(' ', '_')

def detect_percentage_folders(root):
    folders = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        name = p.name.strip()
        m = re.match('^(\\d+)%?-(\\d+)%?$', name)
        if not m:
            continue
        start = int(m.group(1))
        end = int(m.group(2))
        if (start, end) not in TARGET_PERCENTAGE_RANGES:
            continue
        folders.append((start, end, p))
    folders = sorted(folders, key=lambda x: (x[0], x[1]))
    if len(folders) == 0:
        raise RuntimeError(f'No target percentage folders found in: {root}')
    print(f'\nDetected percentage folders in {root}:')
    for start, end, p in folders:
        print(f'  {start}-{end}% -> {p}')
    return folders

def extract_sample_key(path):
    stem = Path(path).stem
    parts = stem.split('_')
    patch_token = None
    patch_index = None
    for i, part in enumerate(parts):
        if re.fullmatch('p\\d+', part, flags=re.IGNORECASE):
            patch_token = part
            patch_index = i
            break
    if patch_token is None:
        return None
    before_patch = parts[:patch_index]
    modality_tokens = {'s1', 's2', 'sar', 'clear', 'cloudy', 'cloud', 'cd', 'lulc', 'mask', 'cloudmask', 'cloud_detection', 'result', 'results', 'land', 'cover', 'maps', 'landcover', 'lc', 'dfc'}
    roi_parts = []
    for part in before_patch:
        if part.lower() in modality_tokens:
            break
        roi_parts.append(part)
    if len(roi_parts) == 0:
        return None
    roi_id = '_'.join(roi_parts)
    return f'{roi_id}_{patch_token}'

def extract_roi_id(sample_key):
    m = re.search('(.+)_p\\d+$', sample_key)
    if m:
        return m.group(1)
    return sample_key

def collect_files_by_key(folder):
    files_by_key = {}
    if not folder.exists():
        return files_by_key
    files = sorted(list(folder.glob('*.tif')) + list(folder.glob('*.tiff')))
    for p in files:
        key = extract_sample_key(p)
        if key is None:
            continue
        files_by_key[key] = p
    return files_by_key

def collect_samples_from_root(root):
    folders = detect_percentage_folders(root)
    samples = []
    for start, end, percentage_dir in folders:
        percentage_name = percentage_dir.name
        s1_dir = percentage_dir / 's1'
        clean_s2_dir = percentage_dir / 's2'
        cloudy_s2_dir = percentage_dir / 's2_cloudy'
        cloud_mask_dir = percentage_dir / 'cloud_detection_results'
        label_dir = percentage_dir / 'land_cover_maps'
        s1_files = collect_files_by_key(s1_dir)
        clean_s2_files = collect_files_by_key(clean_s2_dir)
        cloudy_s2_files = collect_files_by_key(cloudy_s2_dir)
        cloud_mask_files = collect_files_by_key(cloud_mask_dir)
        label_files = collect_files_by_key(label_dir)
        matched_keys = sorted(set(s1_files.keys()) & set(clean_s2_files.keys()) & set(cloudy_s2_files.keys()) & set(label_files.keys()))
        no_mask_count = sum((1 for k in matched_keys if k not in cloud_mask_files))
        print('\n' + '=' * 80)
        print(f'Collecting: {root} | {percentage_name}')
        print('=' * 80)
        print('S1 files:', len(s1_files))
        print('Clean S2 files:', len(clean_s2_files))
        print('Cloudy S2 files:', len(cloudy_s2_files))
        print('Cloud-mask files:', len(cloud_mask_files))
        print('Label files:', len(label_files))
        print('Matched samples:', len(matched_keys))
        print('Matched samples WITHOUT cloud-mask file:', no_mask_count)
        for key in matched_keys:
            samples.append({'percentage_name': percentage_name, 'percentage_range': f'{start}-{end}', 'sample_key': key, 'roi_id': extract_roi_id(key), 's1_path': s1_files[key], 'clean_s2_path': clean_s2_files[key], 'cloudy_s2_path': cloudy_s2_files[key], 'cloud_mask_path': cloud_mask_files.get(key, None), 'label_path': label_files[key]})
    return samples
print('\nCollecting selected TEST samples...')
test_samples = collect_samples_from_root(SELECTED_TEST_ROOT)
test_patch_keys = set((s['sample_key'] for s in test_samples))
print('\nSelected test samples:', len(test_samples))
print('Unique selected test patch keys:', len(test_patch_keys))
if len(test_samples) == 0:
    raise RuntimeError('Selected test set is empty.')
print('\nCollecting FULL Luojia samples...')
all_full_samples = collect_samples_from_root(FULL_LUOJIA_ROOT)
print('\nFull Luojia matched samples:', len(all_full_samples))
trainval_samples = [s for s in all_full_samples if s['sample_key'] not in test_patch_keys]
print('Train/val samples after removing selected test patch keys:', len(trainval_samples))
if len(trainval_samples) == 0:
    raise RuntimeError('No train/validation samples left after excluding selected test patches.')

def split_train_val_by_patch_key(samples, seed=0, train_ratio=0.7):
    by_patch = defaultdict(list)
    for s in samples:
        by_patch[s['sample_key']].append(s)
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
        raise RuntimeError('Training split is empty.')
    if len(val_samples_local) == 0:
        raise RuntimeError('Validation split is empty.')
    return (train_samples_local, val_samples_local, train_patch_keys, val_patch_keys)
train_samples, val_samples, train_patch_keys, val_patch_keys = split_train_val_by_patch_key(trainval_samples, seed=BASE_SEED, train_ratio=TRAIN_RATIO)
print('\nPatch-level train/validation split:')
print('Train unique patch keys:', len(train_patch_keys))
print('Val unique patch keys:', len(val_patch_keys))
print('Train samples:', len(train_samples))
print('Val samples:', len(val_samples))

def assert_no_leakage():
    train_keys = set((s['sample_key'] for s in train_samples))
    val_keys = set((s['sample_key'] for s in val_samples))
    test_keys = set((s['sample_key'] for s in test_samples))
    train_test_overlap = train_keys & test_keys
    val_test_overlap = val_keys & test_keys
    train_val_overlap = train_keys & val_keys
    print('\nLeakage check:')
    print('Train/test overlap patch keys:', len(train_test_overlap))
    print('Val/test overlap patch keys:', len(val_test_overlap))
    print('Train/val overlap patch keys:', len(train_val_overlap))
    if len(train_test_overlap) > 0:
        print('Examples:', sorted(list(train_test_overlap))[:20])
        raise RuntimeError('Leakage detected: training patches appear in selected test.')
    if len(val_test_overlap) > 0:
        print('Examples:', sorted(list(val_test_overlap))[:20])
        raise RuntimeError('Leakage detected: validation patches appear in selected test.')
    if len(train_val_overlap) > 0:
        print('Examples:', sorted(list(train_val_overlap))[:20])
        raise RuntimeError('Leakage detected: same patch appears in train and validation.')
    print('No leakage detected.')
assert_no_leakage()
print('\nCloud percentage distribution:')
print('Train:', Counter([s['percentage_name'] for s in train_samples]))
print('Val:  ', Counter([s['percentage_name'] for s in val_samples]))
print('Test: ', Counter([s['percentage_name'] for s in test_samples]))

def save_split_manifest(csv_root, tag):
    split_csv = csv_root / f'split_manifest_{tag}.csv'
    with open(split_csv, 'w', newline='') as f:
        fieldnames = ['split', 'percentage_name', 'sample_key', 'roi_id', 's1_path', 'clean_s2_path', 'cloudy_s2_path', 'cloud_mask_path', 'label_path']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, rows in [('train', train_samples), ('val', val_samples), ('test_selected', test_samples)]:
            for s in rows:
                writer.writerow({'split': split_name, 'percentage_name': s['percentage_name'], 'sample_key': s['sample_key'], 'roi_id': s['roi_id'], 's1_path': str(s['s1_path']), 'clean_s2_path': str(s['clean_s2_path']), 'cloudy_s2_path': str(s['cloudy_s2_path']), 'cloud_mask_path': '' if s['cloud_mask_path'] is None else str(s['cloud_mask_path']), 'label_path': str(s['label_path'])})
    print('Saved split manifest:', split_csv)
save_split_manifest(SIMPLE_CSV_ROOT, 'simple')
save_split_manifest(RECON_CSV_ROOT, 'recon')

def compute_ndi_from_s2_file(path):
    with rasterio.open(path) as src:
        b3 = src.read(B3).astype('float32')
        b4 = src.read(B4).astype('float32')
        b8 = src.read(B8).astype('float32')
        b11 = src.read(B11).astype('float32')
    eps = 1e-06
    ndvi = (b8 - b4) / (b8 + b4 + eps)
    ndwi = (b3 - b8) / (b3 + b8 + eps)
    ndbi = (b11 - b8) / (b11 + b8 + eps)
    ndi = np.stack([ndvi, ndwi, ndbi], axis=0)
    ndi = np.nan_to_num(ndi, nan=0.0, posinf=0.0, neginf=0.0)
    ndi = np.clip(ndi, -1.0, 1.0).astype('float32')
    return ndi

def read_cloud_mask_for_loss(path, shape):
    if path is None:
        return np.ones(shape, dtype='float32')
    try:
        with rasterio.open(path) as src:
            arr = src.read(1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        mask = (arr > 0).astype('float32')
        if mask.shape != shape:
            return np.ones(shape, dtype='float32')
        return mask
    except Exception:
        return np.ones(shape, dtype='float32')

class LuojiaUNetDataset(Dataset):

    def __init__(self, samples, return_index=False):
        self.samples = samples
        self.return_index = return_index
        self.mean = MEAN_VALS
        self.std = STD_VALS

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        with rasterio.open(item['s1_path']) as src1:
            s1 = src1.read(S1_BANDS).astype('float32')
        with rasterio.open(item['cloudy_s2_path']) as src2:
            s2_cloudy_input = src2.read(S2_INPUT_BANDS).astype('float32')
        x = torch.from_numpy(np.concatenate([s1, s2_cloudy_input], axis=0)).float()
        x = (x - self.mean) / self.std
        with rasterio.open(item['label_path']) as src:
            label = src.read(1).astype('int64')
        valid = (label >= 0) & (label < NUM_CLASSES)
        clean_label = np.full(label.shape, IGNORE_INDEX, dtype=np.int64)
        clean_label[valid] = label[valid]
        y = torch.from_numpy(clean_label).long()
        if self.return_index:
            return (x, y, idx)
        return (x, y)

class LuojiaUNetMultitaskNDIDataset(Dataset):

    def __init__(self, samples, return_index=False):
        self.samples = samples
        self.return_index = return_index
        self.mean = MEAN_VALS
        self.std = STD_VALS

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        with rasterio.open(item['s1_path']) as src1:
            s1 = src1.read(S1_BANDS).astype('float32')
        with rasterio.open(item['cloudy_s2_path']) as src2:
            s2_cloudy_input = src2.read(S2_INPUT_BANDS).astype('float32')
        x = torch.from_numpy(np.concatenate([s1, s2_cloudy_input], axis=0)).float()
        x = (x - self.mean) / self.std
        ndi_cloudy = torch.from_numpy(compute_ndi_from_s2_file(item['cloudy_s2_path'])).float()
        ndi_clean = torch.from_numpy(compute_ndi_from_s2_file(item['clean_s2_path'])).float()
        h, w = (ndi_clean.shape[1], ndi_clean.shape[2])
        mask_np = read_cloud_mask_for_loss(item['cloud_mask_path'], shape=(h, w))
        mask = torch.from_numpy(mask_np).float().unsqueeze(0)
        with rasterio.open(item['label_path']) as src:
            label = src.read(1).astype('int64')
        valid = (label >= 0) & (label < NUM_CLASSES)
        clean_label = np.full(label.shape, IGNORE_INDEX, dtype=np.int64)
        clean_label[valid] = label[valid]
        y = torch.from_numpy(clean_label).long()
        if self.return_index:
            return (x, y, ndi_cloudy, ndi_clean, mask, idx)
        return (x, y, ndi_cloudy, ndi_clean, mask)

def CBR(i, o, k=3, p=1):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=p, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

def conv_block_relu(in_ch, out_ch):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), nn.ReLU(inplace=True))

class UNetResNetBackbone(nn.Module):

    def __init__(self, pretrained_model, out_channels=NUM_CLASSES):
        super().__init__()
        encoder = pretrained_model.model.vision_encoder
        self.enc1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.act1)
        self.enc2 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.enc3 = encoder.layer2
        self.enc4 = encoder.layer3
        self.enc5 = encoder.layer4
        self.uptrans1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.decoder1 = conv_block_relu(2048, 1024)
        self.uptrans2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder2 = conv_block_relu(1024, 512)
        self.uptrans3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block_relu(512, 256)
        self.uptrans4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder4 = conv_block_relu(192, 128)
        self.uptrans5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        d1 = self.uptrans1(x5)
        d1 = torch.cat([d1, x4], dim=1)
        d1 = self.decoder1(d1)
        d2 = self.uptrans2(d1)
        d2 = torch.cat([d2, x3], dim=1)
        d2 = self.decoder2(d2)
        d3 = self.uptrans3(d2)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.decoder3(d3)
        d4 = self.uptrans4(d3)
        d4 = torch.cat([d4, x1], dim=1)
        d4 = self.decoder4(d4)
        d5 = self.uptrans5(d4)
        out = self.final_conv(d5)
        return out

class MultiTaskUNetResNetBackbone(nn.Module):

    def __init__(self, pretrained_model, num_classes=NUM_CLASSES):
        super().__init__()
        encoder = pretrained_model.model.vision_encoder
        self.enc1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.act1)
        self.enc2 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.enc3 = encoder.layer2
        self.enc4 = encoder.layer3
        self.enc5 = encoder.layer4
        self.uptrans1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.decoder1 = conv_block_relu(2048, 1024)
        self.uptrans2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder2 = conv_block_relu(1024, 512)
        self.uptrans3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block_relu(512, 256)
        self.uptrans4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder4 = conv_block_relu(192, 128)
        self.uptrans5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.r64 = nn.Sequential(CBR(256, 128), CBR(128, 128))
        self.inject = nn.Sequential(CBR(128 + 3 + 256, 128, k=1, p=0))
        self.r256 = nn.Sequential(CBR(128, 64), CBR(64, 32))
        self.r_out = nn.Sequential(nn.Conv2d(32, 3, 1), nn.Tanh())
        self.seg_fuse = CBR(64 + 32, 64, k=1, p=0)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x, ndi_cloudy):
        input_size = x.shape[-2:]
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        r = self.r64(x2)
        ndi64 = F.interpolate(ndi_cloudy, size=x2.shape[-2:], mode='bilinear', align_corners=True)
        r = self.inject(torch.cat([r, ndi64, x2], dim=1))
        r = F.interpolate(r, size=input_size, mode='bilinear', align_corners=True)
        r_feat = self.r256(r)
        ndi_pred = self.r_out(r_feat)
        d1 = self.uptrans1(x5)
        d1 = torch.cat([d1, x4], dim=1)
        d1 = self.decoder1(d1)
        d2 = self.uptrans2(d1)
        d2 = torch.cat([d2, x3], dim=1)
        d2 = self.decoder2(d2)
        d3 = self.uptrans3(d2)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.decoder3(d3)
        d4 = self.uptrans4(d3)
        d4 = torch.cat([d4, x1], dim=1)
        d4 = self.decoder4(d4)
        d5 = self.uptrans5(d4)
        fused = self.seg_fuse(torch.cat([d5, r_feat.detach()], dim=1))
        seg_logits = self.final_conv(fused)
        return (seg_logits, ndi_pred)

def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            return ckpt['model_state_dict']
        if 'state_dict' in ckpt:
            return ckpt['state_dict']
    return ckpt

def strip_prefix_if_present(state_dict, prefix):
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out

def load_simple_unet_into_multitask(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = extract_state_dict(ckpt)
    sd = strip_prefix_if_present(sd, 'module.')
    current = model.state_dict()
    loadable = {}
    skipped = []
    for k, v in sd.items():
        if k in current and current[k].shape == v.shape:
            loadable[k] = v
        else:
            skipped.append(k)
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    print('\nLoaded simple UNet checkpoint into multitask UNet:')
    print('Checkpoint:', ckpt_path)
    print('Loaded compatible keys:', len(loadable))
    print('Skipped keys:', len(skipped))
    print('Missing keys are expected for reconstruction branch and seg_fuse.')
    print('Missing keys:', len(missing))
    print('Unexpected keys:', len(unexpected))
    if skipped:
        print('First skipped keys:', skipped[:20])

def freeze_unet_encoder(model):
    for module in [model.enc1, model.enc2, model.enc3, model.enc4, model.enc5]:
        for p in module.parameters():
            p.requires_grad = False
    trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
    frozen = sum((p.numel() for p in model.parameters() if not p.requires_grad))
    print('\nUNet encoder frozen.')
    print('Frozen modules: enc1, enc2, enc3, enc4, enc5')
    print('Trainable parameters:', trainable)
    print('Frozen parameters:', frozen)

def count_parameters(model):
    total = sum((p.numel() for p in model.parameters()))
    trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
    return (total, trainable)

def masked_l1(pred, target, mask):
    m = mask.expand_as(pred)
    denom = m.sum()
    if denom.item() == 0:
        return pred.new_tensor(0.0)
    loss = torch.abs(pred - target) * m
    loss = loss.sum() / (denom + 1e-06)
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
                ious.append(intersection / (union + 1e-06))
        if len(ious) == 0:
            return torch.tensor(0.0, device=logits.device)
        return torch.mean(torch.stack(ious))

def make_stage1_loaders():
    worker_init_fn = seed_worker_factory(BASE_SEED)
    train_dataset = LuojiaUNetDataset(train_samples)
    val_dataset = LuojiaUNetDataset(val_samples)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn, generator=torch.Generator().manual_seed(BASE_SEED), pin_memory=torch.cuda.is_available(), persistent_workers=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn, generator=torch.Generator().manual_seed(BASE_SEED), pin_memory=torch.cuda.is_available(), persistent_workers=False, drop_last=False)
    return (train_loader, val_loader)

def make_stage2_loaders():
    worker_init_fn = seed_worker_factory(BASE_SEED)
    train_dataset = LuojiaUNetMultitaskNDIDataset(train_samples)
    val_dataset = LuojiaUNetMultitaskNDIDataset(val_samples)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn, generator=torch.Generator().manual_seed(BASE_SEED), pin_memory=torch.cuda.is_available(), persistent_workers=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn, generator=torch.Generator().manual_seed(BASE_SEED), pin_memory=torch.cuda.is_available(), persistent_workers=False, drop_last=False)
    return (train_loader, val_loader)

def train_simple_unet(model, train_loader, val_loader, device, num_epochs=NUM_EPOCHS_STAGE1, lr=LR_STAGE1, save_path=SIMPLE_SAVE_PATH, patience=PATIENCE_STAGE1):
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
        for imgs, labels in tqdm(train_loader, desc=f'Stage 1 | Epoch {epoch + 1} train'):
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
            for imgs, labels in tqdm(val_loader, desc=f'Stage 1 | Epoch {epoch + 1} val'):
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
        hist_rows.append({'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'train_iou': avg_train_iou, 'val_iou': avg_val_iou})
        hist_csv = SIMPLE_CSV_ROOT / 'training_history_unet_simple.csv'
        with open(hist_csv, 'w', newline='') as f:
            fieldnames = ['epoch', 'train_loss', 'val_loss', 'train_iou', 'val_iou']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(hist_rows)
        print(f'\nStage 1 | Epoch {epoch + 1}/{num_epochs} | Train loss={avg_train_loss:.4f} IoU={avg_train_iou:.4f} | Val loss={avg_val_loss:.4f} IoU={avg_val_iou:.4f}')
        if avg_val_iou > best_val_iou + 1e-06:
            best_val_iou = avg_val_iou
            best_epoch = epoch + 1
            bad_epochs = 0
            torch.save({'epoch': epoch + 1, 'val_iou': best_val_iou, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'hist': hist_rows, 'model': 'UNetResNetBackbone', 'num_classes': NUM_CLASSES, 'in_channels': IN_CHANNELS, 'seed': BASE_SEED, 'train_ratio': TRAIN_RATIO, 'val_ratio': VAL_RATIO, 'stage': 'simple_segmentation'}, save_path)
            print(f'✅ Saved best Stage 1 checkpoint to: {save_path}')
        else:
            bad_epochs += 1
            print(f'No Stage 1 validation IoU improvement: {bad_epochs}/{patience}')
            if bad_epochs >= patience:
                print('Stage 1 early stopping.')
                break
        free_gpu_memory()
    print('\nStage 1 training finished.')
    print('Best epoch:', best_epoch)
    print('Best validation IoU:', best_val_iou)
    print('Best checkpoint:', save_path)

def train_unet_reconstruction(model, train_loader, val_loader, device, num_epochs=NUM_EPOCHS_STAGE2, lr=LR_STAGE2, best_path=RECON_BEST_PATH, patience=PATIENCE_STAGE2):
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
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
        for x, y, ndi_cloudy, ndi_clean, mask in tqdm(train_loader, desc=f'Stage 2 | Epoch {epoch + 1} train'):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            ndi_cloudy = ndi_cloudy.to(device, non_blocking=True)
            ndi_clean = ndi_clean.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
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
            for x, y, ndi_cloudy, ndi_clean, mask in tqdm(val_loader, desc=f'Stage 2 | Epoch {epoch + 1} val'):
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
        hist_rows.append({'epoch': epoch + 1, 'train_iou': train_iou, 'val_iou': val_iou, 'train_seg_loss': train_seg, 'val_seg_loss': val_seg, 'train_rec_loss': train_rec, 'val_rec_loss': val_rec, 'train_total_loss': train_total, 'val_total_loss': val_total, 'w_seg': W_SEG, 'w_rec': W_REC})
        hist_csv = RECON_CSV_ROOT / 'training_history_unet_recon_rec040_safe_v2.csv'
        with open(hist_csv, 'w', newline='') as f:
            fieldnames = ['epoch', 'train_iou', 'val_iou', 'train_seg_loss', 'val_seg_loss', 'train_rec_loss', 'val_rec_loss', 'train_total_loss', 'val_total_loss', 'w_seg', 'w_rec']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(hist_rows)
        print(f'\nStage 2 | Epoch {epoch + 1}/{num_epochs} | Train IoU={train_iou:.4f} Val IoU={val_iou:.4f} | Train Total={train_total:.4f} Val Total={val_total:.4f} | Train Seg={train_seg:.4f} Val Seg={val_seg:.4f} | Train Rec={train_rec:.4f} Val Rec={val_rec:.4f}')
        if val_iou > best_val_iou + 1e-06:
            best_val_iou = val_iou
            best_epoch = epoch + 1
            bad_epochs = 0
            torch.save({'epoch': epoch + 1, 'val_iou': best_val_iou, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'hist': hist_rows, 'model': 'MultiTaskUNetResNetBackbone', 'source_simple_checkpoint': str(SIMPLE_SAVE_PATH), 'num_classes': NUM_CLASSES, 'in_channels': IN_CHANNELS, 'w_seg': W_SEG, 'w_rec': W_REC, 'seed': BASE_SEED, 'train_ratio': TRAIN_RATIO, 'val_ratio': VAL_RATIO, 'encoder_frozen': True, 'frozen_modules': ['enc1', 'enc2', 'enc3', 'enc4', 'enc5'], 'ndi_mode': 'on_the_fly_no_disk_cache'}, best_path)
            print(f'✅ Saved best Stage 2 checkpoint to: {best_path}')
        else:
            bad_epochs += 1
            print(f'No Stage 2 validation IoU improvement: {bad_epochs}/{patience}')
            if bad_epochs >= patience:
                print('Stage 2 early stopping.')
                break
        free_gpu_memory()
    print('\nStage 2 training finished.')
    print('Best epoch:', best_epoch)
    print('Best validation IoU:', best_val_iou)
    print('Best checkpoint:', best_path)

def confusion_matrix_update(cm, pred, target, num_classes=NUM_CLASSES):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    valid = (target != IGNORE_INDEX) & (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
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
        iou = tp / (iou_denom + 1e-06) if iou_denom > 0 else np.nan
        f1 = 2.0 * tp / (f1_denom + 1e-06) if f1_denom > 0 else np.nan
        per_class[f'IoU_class_{cls}'] = float(iou) if not np.isnan(iou) else ''
        per_class[f'F1_class_{cls}'] = float(f1) if not np.isnan(f1) else ''
        if not np.isnan(iou):
            ious.append(iou)
        if not np.isnan(f1):
            f1s.append(f1)
    total = cm.sum()
    correct = np.trace(cm)
    pixel_acc = correct / (total + 1e-06) if total > 0 else np.nan
    return {'mIoU': float(np.mean(ious)) if len(ious) > 0 else '', 'macro_F1': float(np.mean(f1s)) if len(f1s) > 0 else '', 'pixel_accuracy': float(pixel_acc) if not np.isnan(pixel_acc) else '', **per_class}

def save_prediction_mask(out_path, pred_mask, reference_path):
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype='uint8', compress='deflate', nodata=IGNORE_INDEX)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(pred_mask.astype('uint8'), 1)

def write_seg_test_csv(prefix, rows, cm_by_percentage, cm_overall, csv_root):
    class_fields = []
    for i in range(NUM_CLASSES):
        class_fields.extend([f'IoU_class_{i}', f'F1_class_{i}'])
    per_sample_csv = csv_root / f'{prefix}_selected_test_per_sample.csv'
    with open(per_sample_csv, 'w', newline='') as f:
        fieldnames = ['cloud_percentage', 'sample_key', 'roi_id', 'prediction_path', 'mIoU', 'macro_F1', 'pixel_accuracy', *class_fields]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    summary_rows = []
    for percentage_name, cm in sorted(cm_by_percentage.items()):
        summary_rows.append({'cloud_percentage': percentage_name, 'samples': sum((1 for r in rows if r['cloud_percentage'] == percentage_name)), **metrics_from_confusion(cm)})
    summary_rows.append({'cloud_percentage': 'ALL', 'samples': len(rows), **metrics_from_confusion(cm_overall)})
    summary_csv = csv_root / f'{prefix}_selected_test_summary_by_cloud_percentage.csv'
    with open(summary_csv, 'w', newline='') as f:
        fieldnames = ['cloud_percentage', 'samples', 'mIoU', 'macro_F1', 'pixel_accuracy', *class_fields]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    print('\nSelected test summary:', prefix)
    for row in summary_rows:
        print(f"{row['cloud_percentage']:10s} | n={row['samples']:4d} | mIoU={row['mIoU']} | macro_F1={row['macro_F1']} | pixel_acc={row['pixel_accuracy']}")
    print('Saved per-sample CSV:', per_sample_csv)
    print('Saved summary CSV:', summary_csv)

def evaluate_simple_selected_test(model, device):
    dataset = LuojiaUNetDataset(test_samples, return_index=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    model.eval()
    cm_overall = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    cm_by_percentage = {}
    rows = []
    with torch.no_grad():
        for x, y, indices in tqdm(loader, desc='Stage 1 selected test eval'):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy().astype('uint8')
            labels_np = y.cpu().numpy().astype('int64')
            indices_np = indices.cpu().numpy().tolist()
            for b in range(preds.shape[0]):
                item = test_samples[indices_np[b]]
                percentage_name = item['percentage_name']
                sample_key = item['sample_key']
                if percentage_name not in cm_by_percentage:
                    cm_by_percentage[percentage_name] = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
                cm_sample = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
                cm_sample = confusion_matrix_update(cm_sample, preds[b], labels_np[b], NUM_CLASSES)
                cm_overall = confusion_matrix_update(cm_overall, preds[b], labels_np[b], NUM_CLASSES)
                cm_by_percentage[percentage_name] = confusion_matrix_update(cm_by_percentage[percentage_name], preds[b], labels_np[b], NUM_CLASSES)
                pred_path = ''
                if SAVE_TEST_PREDICTIONS:
                    pred_path = SIMPLE_PRED_ROOT / percentage_name / 'pred_masks' / f'unet_simple_{safe_name(percentage_name)}_{safe_name(sample_key)}.tif'
                    save_prediction_mask(pred_path, preds[b], item['label_path'])
                    pred_path = str(pred_path)
                rows.append({'cloud_percentage': percentage_name, 'sample_key': sample_key, 'roi_id': item['roi_id'], 'prediction_path': pred_path, **metrics_from_confusion(cm_sample)})
    write_seg_test_csv('unet_simple', rows, cm_by_percentage, cm_overall, SIMPLE_CSV_ROOT)

def evaluate_recon_selected_test(model, device):
    dataset = LuojiaUNetMultitaskNDIDataset(test_samples, return_index=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    model.eval()
    cm_overall = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    cm_by_percentage = {}
    rows = []
    with torch.no_grad():
        for x, y, ndi_cloudy, ndi_clean, mask, indices in tqdm(loader, desc='Stage 2 selected test eval'):
            x = x.to(device, non_blocking=True)
            ndi_cloudy = ndi_cloudy.to(device, non_blocking=True)
            seg_logits, _ = model(x, ndi_cloudy)
            preds = torch.argmax(seg_logits, dim=1).cpu().numpy().astype('uint8')
            labels_np = y.cpu().numpy().astype('int64')
            indices_np = indices.cpu().numpy().tolist()
            for b in range(preds.shape[0]):
                item = test_samples[indices_np[b]]
                percentage_name = item['percentage_name']
                sample_key = item['sample_key']
                if percentage_name not in cm_by_percentage:
                    cm_by_percentage[percentage_name] = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
                cm_sample = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
                cm_sample = confusion_matrix_update(cm_sample, preds[b], labels_np[b], NUM_CLASSES)
                cm_overall = confusion_matrix_update(cm_overall, preds[b], labels_np[b], NUM_CLASSES)
                cm_by_percentage[percentage_name] = confusion_matrix_update(cm_by_percentage[percentage_name], preds[b], labels_np[b], NUM_CLASSES)
                pred_path = ''
                if SAVE_TEST_PREDICTIONS:
                    pred_path = RECON_PRED_ROOT / percentage_name / 'pred_masks' / f'unet_recon_rec040_{safe_name(percentage_name)}_{safe_name(sample_key)}.tif'
                    save_prediction_mask(pred_path, preds[b], item['label_path'])
                    pred_path = str(pred_path)
                rows.append({'cloud_percentage': percentage_name, 'sample_key': sample_key, 'roi_id': item['roi_id'], 'prediction_path': pred_path, **metrics_from_confusion(cm_sample)})
    write_seg_test_csv('unet_recon_rec040', rows, cm_by_percentage, cm_overall, RECON_CSV_ROOT)

def run_stage1_simple_unet():
    print('\n' + '#' * 80)
    print('STAGE 1: SIMPLE UNET SEGMENTATION')
    print('#' * 80)
    set_seed_everywhere(BASE_SEED)
    cleanup_gpu_cpu_resources('before Stage 1')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    print('Stage 1 checkpoint:', SIMPLE_SAVE_PATH)
    train_loader, val_loader = make_stage1_loaders()
    print('\nLoading BigEarthNet pretrained encoder...')
    pretrained = BigEarthNetv2_0_ImageClassifier.from_pretrained('BIFOLD-BigEarthNetv2-0/resnet50-all-v0.2.0')
    model = UNetResNetBackbone(pretrained, out_channels=NUM_CLASSES)
    total_params, trainable_params = count_parameters(model)
    print('\nModel parameter count:')
    print(f'Total parameters:     {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    train_simple_unet(model=model, train_loader=train_loader, val_loader=val_loader, device=device, num_epochs=NUM_EPOCHS_STAGE1, lr=LR_STAGE1, save_path=SIMPLE_SAVE_PATH, patience=PATIENCE_STAGE1)
    if RUN_STAGE_1_TEST_EVAL:
        print('\nLoading best Stage 1 checkpoint for selected test evaluation...')
        ckpt = torch.load(SIMPLE_SAVE_PATH, map_location=device)
        model.load_state_dict(extract_state_dict(ckpt))
        model.to(device)
        model.eval()
        evaluate_simple_selected_test(model, device)
    del model
    del pretrained
    del train_loader
    del val_loader
    cleanup_gpu_cpu_resources('after Stage 1')

def run_stage2_reconstruction_from_simple():
    print('\n' + '#' * 80)
    print('STAGE 2: UNET + NDI RECONSTRUCTION FROM SIMPLE CHECKPOINT')
    print('#' * 80)
    if not SIMPLE_SAVE_PATH.exists():
        raise FileNotFoundError(f'Stage 1 checkpoint is missing. Run Stage 1 first:\n  {SIMPLE_SAVE_PATH}')
    set_seed_everywhere(BASE_SEED)
    cleanup_gpu_cpu_resources('before Stage 2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    print('Stage 1 source checkpoint:', SIMPLE_SAVE_PATH)
    print('Stage 2 best checkpoint:', RECON_BEST_PATH)
    print(f'Loss weights: W_SEG={W_SEG}, W_REC={W_REC}')
    print('NDI mode: on-the-fly, no disk cache')
    train_loader, val_loader = make_stage2_loaders()
    print('\nLoading BigEarthNet pretrained encoder...')
    pretrained = BigEarthNetv2_0_ImageClassifier.from_pretrained('BIFOLD-BigEarthNetv2-0/resnet50-all-v0.2.0')
    model = MultiTaskUNetResNetBackbone(pretrained, num_classes=NUM_CLASSES)
    print('\nLoading saved Stage 1 simple UNet weights...')
    load_simple_unet_into_multitask(model, SIMPLE_SAVE_PATH)
    print('\nFreezing UNet encoder for Stage 2...')
    freeze_unet_encoder(model)
    total_params, trainable_params = count_parameters(model)
    print('\nModel parameter count after freezing:')
    print(f'Total parameters:     {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    train_unet_reconstruction(model=model, train_loader=train_loader, val_loader=val_loader, device=device, num_epochs=NUM_EPOCHS_STAGE2, lr=LR_STAGE2, best_path=RECON_BEST_PATH, patience=PATIENCE_STAGE2)
    if RUN_STAGE_2_TEST_EVAL:
        print('\nLoading best Stage 2 checkpoint for selected test evaluation...')
        ckpt = torch.load(RECON_BEST_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()
        evaluate_recon_selected_test(model, device)
    del model
    del pretrained
    del train_loader
    del val_loader
    cleanup_gpu_cpu_resources('after Stage 2')
if __name__ == '__main__':
    print('=' * 80)
    print('Two-stage UNet training on LuojiaSET-OSFCR')
    print('=' * 80)
    print('Stage 1 checkpoint:', SIMPLE_SAVE_PATH)
    print('Stage 2 checkpoint:', RECON_BEST_PATH)
    print(f'Fixed reconstruction weight: W_REC={W_REC}, W_SEG={W_SEG}')
    print('NDI mode: on-the-fly, no disk cache')
    print('=' * 80)
    if RUN_STAGE_1_SIMPLE_UNET:
        run_stage1_simple_unet()
    else:
        print('\nSkipping Stage 1 because RUN_STAGE_1_SIMPLE_UNET = False')
    if RUN_STAGE_2_RECON_FROM_SIMPLE:
        run_stage2_reconstruction_from_simple()
    else:
        print('\nSkipping Stage 2 because RUN_STAGE_2_RECON_FROM_SIMPLE = False')
    print('\nDONE.')
    print('Stage 1 checkpoint:', SIMPLE_SAVE_PATH)
    print('Stage 2 checkpoint:', RECON_BEST_PATH)
