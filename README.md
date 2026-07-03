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
