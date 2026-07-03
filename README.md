# Cloud Aware Optical Radar Land Cover Mapping via NDIs Reconstruction

This repository contains the code for cloud-aware optical–radar land-cover mapping using Normalized Difference Indices (NDIs) reconstruction. The objective is to improve semantic segmentation under cloud contamination by combining Sentinel-1 radar information, cloudy Sentinel-2 optical information, and an auxiliary NDI reconstruction branch.

## Synthetic Cloud Generation

For synthetic cloud generation on the DFC2020 data, we used the cloud simulation strategy from [SatelliteCloudGenerator: Controllable Cloud and Shadow Synthesis for Multi-Spectral Optical Satellite Images](https://www.mdpi.com/2072-4292/15/17/4138). The generator provides controllable cloud configurations that simulate different cloud appearances by changing cloud thickness, transparency, and locality.

The cloud configurations follow the four cloud types reported in the SatelliteCloudGenerator paper:

| Parameter | Thick | Local | Thin | Fog |
|---|---:|---:|---:|---:|
| `min_lvl` | `0.0` | `0.0` | `[0.0, 0.1]` | `[0.3, 0.6]` |
| `max_lvl` | `1.0` | `1.0` | `[0.4, 0.7]` | `[0.6, 0.7]` |
| `threshold` | `[0.0, 0.2]` | `[0.0, 0.2]` | `0.0` | `0.0` |
| `locality_degree` | `1` | `[2, 4]` | `[1, 3]` | `1` |
| `decay_factor` | `1.0` | `1.0` | `1.0` | `1.0` |
| `cloud_color` | `True` | `True` | `True` | `True` |
| `channel_offset` | `2` | `2` | `2` | `2` |
| `blur_scaling` | `2.0` | `2.0` | `2.0` | `2.0` |

In our training protocol, the cloud type was changed periodically during training so that the model was exposed to all cloud conditions. The training loop cycles through the cloud configurations, for example:

```text
fog → thin → thick → local → cloud-free → fog → thick → thin → local → ...
```

This periodic pass continues until the end of training. The goal is to avoid training on only one cloud distribution and to force the model to learn robust features across different cloud thicknesses and spatial patterns.

The test set is kept separate from this periodic training schedule. Instead of mixing cloud types during evaluation, the full test set is evaluated under each cloud condition separately. For example, one test run uses fog clouds for the whole test set, another uses thick clouds, another uses thin clouds, another uses local clouds, and another uses cloud-free samples. This makes the final results easier to compare by cloud type.

## Main Idea

Clouds affect the quality of optical Sentinel-2 imagery and can reduce land-cover classification performance. To improve robustness under cloudy conditions, we combine:

- Sentinel-1 radar input
- Cloudy Sentinel-2 optical input
- Land-cover segmentation supervision
- Auxiliary NDI reconstruction supervision

The reconstruction branch predicts clean NDIs from cloudy observations. This auxiliary task helps the model learn cloud-aware representations that improve segmentation performance under cloud contamination.

The reconstructed indices are:

- NDVI
- NDWI
- NDBI

## Training Strategy

Each `.py` file contains two training steps for one model:

1. **Training without NDI reconstruction**  
   The model is first trained for land-cover segmentation only.

2. **Training with NDI reconstruction**  
   The model is then trained with an auxiliary NDI reconstruction branch using a reconstruction loss weight of **40%**.

For the second step, the model weights are initialized from the checkpoint obtained in the first step without NDI reconstruction. This allows the reconstruction-based model to start from a segmentation-trained baseline.

### Step 1: Training without NDI reconstruction

The model is first trained for land-cover segmentation only.

```text
Input: Sentinel-1 + cloudy Sentinel-2
Output: land-cover segmentation mask
Loss: cross-entropy segmentation loss
```

### Step 2: Training with NDI reconstruction

The model is then trained with an auxiliary NDI reconstruction branch.

```text
Input: Sentinel-1 + cloudy Sentinel-2 + cloudy NDIs
Output 1: land-cover segmentation mask
Output 2: reconstructed clean NDIs
Loss: 60% segmentation loss + 40% NDI reconstruction loss
```

## Encoder Backbone

For **UNet** and **DeepLabV3**, we initialize the encoder using a pretrained **ResNet-50** model from **BigEarthNet v2.0**, a large-scale remote sensing dataset.

BigEarthNet v2.0:  
https://bigearth.net/

The implementation relies on:

```python
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
```

This pretrained ResNet-50 encoder is used for UNet and DeepLabV3. Other models do not use this BigEarthNet-pretrained ResNet-50 initialization.

## Datasets

Cloud injection was performed on the DFC2020 dataset.

DFC2020 dataset:  
https://www.grss-ieee.org/community/technical-committees/2020-ieee-grss-data-fusion-contest/

Real cloudy experiments are performed using the LuojiaSET-OSFCR dataset.

LuojiaSET-OSFCR dataset:  
https://github.com/RSIIPAC/LuojiaSET-OSFCR

## Reference

Czerkawski, M., Atkinson, R., Michie, C., & Tachtatzis, C. (2023). [SatelliteCloudGenerator: Controllable Cloud and Shadow Synthesis for Multi-Spectral Optical Satellite Images](https://www.mdpi.com/2072-4292/15/17/4138). *Remote Sensing*, 15(17), 4138.
