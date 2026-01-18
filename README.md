<h2 align="center">
Physics-Consistent Universal Imaging Inverse Problem Solving for Heterogeneous Systems via Latent Space Optimization

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8-blue.svg" alt="Python 3.8">
  <img src="https://img.shields.io/badge/PyTorch-2.1.2-red.svg" alt="PyTorch 2.1.2">
  <img src="https://img.shields.io/badge/TorchVision-0.16.2-orange.svg" alt="TorchVision 0.16.2">
</p>
<p align="center">
  <img src="https://github.com/LyJune03/Uni_inverse/blob/main/figs/Graphical%20Abstract%20Image.png" width="90%" alt="Concept figure placeholder">
</p>




## 📰 News

[1] **2025.09.29**  We have uploaded the key implementation of the proposed method on the **single-pixel imaging** task.

## ✨ Key Ideas

- We propose an unsupervised framework for solving inverse problems, including nonlinear cases, which eliminates the need for independent pretraining tailored to specific imaging mechanisms or degradation levels. 
- By leveraging structural optimization in latent space, the framework can reconstruct scenes beyond the original generator manifold, achieving significant advantages over other methods in cross‐domain imaging tasks.
- For dynamic imaging involving temporally varying targets, a latent space prior module is constructed based on the latent representation of inter-frame correlations, effectively enhancing both imaging quality and efficiency. 

## 🖼️ Visual Results

<p align="center">
  <img src="https://github.com/LyJune03/Uni_inverse/blob/main/figs/visual.png" width="90%" alt="Concept figure placeholder">
</p>




## 🧰 Environment
- Python **3.8**
- PyTorch **2.1.2**, TorchVision **0.16.2**

### Installation
```bash
# 1) Create env
conda create -n uni-inverse python=3.8 -y
conda activate uni-inverse

# 2) Install PyTorch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# 3) Other deps
pip install numpy opencv-python matplotlib lpips wandb dlib
```

## 📂 Repo Structure
```
├── README.md
├── utils
├── figs
├── single-pixel-imaging.py
├── CMOS.py
├── SPAD.py
├── hadamard_measure.py
├── preprocess.py
```

## 🧷 Pretrained Weights

- NVIDIA official weights (PyTorch/ADA)：
  - https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/
- NVIDIA NGC model repository (multi-resolution):
  - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan2/files?version=1

---

## 📦 StyleGAN2-ADA as a submodule

- NVIDIA official implementation (PyTorch, recommended):  
  https://github.com/NVlabs/stylegan2-ada-pytorch
- TensorFlow version (if needed):  
  https://github.com/NVlabs/stylegan2-ada

## 📕Code Overview

- `single_pixel_imaging.py`: A working example that implements the imaging method proposed in this paper (single-pixel imaging case).
- `hadamard_measure.py`: Implements the basic forward model for **single-pixel imaging**, using **cake cutting Hadamard patterns** (structured illumination).
- `SPAD.py`: Implements the basic forward model for SPAD imaging and provides an implementation of the imaging method proposed in this paper.
- `CMOS.py`: Implements the CMOS forward model under the settings defined in this paper and provides an implementation of the proposed imaging method.
- Note: For better isolation between different imaging models, we intentionally minimize shared dependencies across files. As a result, the three imaging models share little to no common code, meaning **each script is relatively long while the total number of files is small**.
