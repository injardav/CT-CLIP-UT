# CT-CLIP-UT

**CT-CLIP-UT** is a customized extension of the [CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP) framework, developed as part of a [Master's thesis](https://drive.google.com/file/d/1mESSdczyXtrmz_XSWyzl8XepzOZJI-PI/view?usp=sharing) at the University of Tartu. While the foundational architecture is based on the original CT-CLIP implementation, the codebase has been extensively modified to support new functionality. This project investigates the potential of the pre-trained CT-CLIP model to perform weakly supervised segmentation on chest CT scans.

## Overview

This work focuses on evaluating the interpretability and spatial localization capabilities of CT-CLIP through the application of various attribution methods. The core contributions include the implementation and comparison of:

- Raw Attention Maps
- Attention Rollout
- Integrated Gradients
- Grad-CAM
- Occlusion Sensitivity

## Results

The following visualizations demonstrate the model’s response under different attribution techniques.

### Raw Attention Maps
#### Spatial
![Spatial Raw Attention Weights](results/spatial_raw_attention_maps.gif)

#### Temporal
![Temporal Raw Attention Weights](results/temporal_raw_attention_maps.gif)

### Attention Rollout
#### Spatial
![Spatial Attention Rollout](results/spatial_attention_rollout.gif)

#### Temporal
![Temporal Attention Rollout](results/temporal_attention_rollout.gif)

### Integrated Gradients
![Integrated Gradients](results/integrated_gradients.gif)

### Grad-CAM
#### Spatial
![Spatial Grad-CAM](results/spatial_grad_cam.gif)

#### Temporal
![Temporal Grad-CAM](results/temporal_grad_cam.gif)

#### Combined
![Combined Grad-CAM](results/combined_grad_cam.gif)

#### VQ
![VQ Grad-CAM](results/vq_grad_cam.gif)

### Occlusion Sensitivity
coming soon!
![Occlusion Sensitivity](results/occlusion_sensitivity.gif)
