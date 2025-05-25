# CT-CLIP-UT

**CT-CLIP-UT** is a customized extension of the [CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP) framework, developed as part of a [Master's thesis](thesis.pdf) at the University of Tartu. While the foundational architecture is based on the original CT-CLIP implementation, the codebase has been extensively modified to support new functionality. This project investigates the potential of the pre-trained CT-CLIP model to perform weakly supervised segmentation on chest CT scans.

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
![Raw Attention Weights](results/raw_attention_weights.gif)

### Attention Rollout
![Attention Rollout](results/attention_rollout.gif)

### Integrated Gradients
![Integrated Gradients](results/integrated_gradients.gif)

### Grad-CAM
![Grad-CAM](results/grad_cam.gif)

### Occlusion Sensitivity
![Occlusion Sensitivity](results/occlusion_sensitivity.gif)
