# Cloud Aware Optical Radar Land Cover Mapping via NDIs Reconstruction

This repository contains the code for cloud-aware optical–radar land-cover mapping using Normalized Difference Indices (NDIs) reconstruction. The objective is to improve semantic segmentation under cloud contamination by combining Sentinel-1 radar information, cloudy Sentinel-2 optical information, and an auxiliary NDI reconstruction branch.

## Training Strategy

Each `.py` file contains two training steps for one model:

1. **Training without NDI reconstruction**  
   The model is first trained for land-cover segmentation only.

2. **Training with NDI reconstruction**  
   The model is then trained with an auxiliary NDI reconstruction branch using a reconstruction loss weight of **40%**.

For the second step, the model weights are initialized from the checkpoint obtained in the first step without NDI reconstruction. This allows the reconstruction-based model to start from a segmentation-trained baseline.

## Encoder Backbone

For **UNet** and **DeepLabV3**, we initialize the encoder using a pretrained **ResNet-50** model from **BigEarthNet v2.0**, a large-scale remote sensing dataset.

BigEarthNet v2.0:  
https://bigearth.net/

The implementation relies on:

```python
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
