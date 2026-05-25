# HFG-Neural-Holo
Ringing-Free Neural Holography with High-Frequency-Guided Generative Priors

# Dataset
The dataset was rendered using [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and [Dr. Bokeh](https://shengcn.github.io/DrBokeh/). We sincerely thank the authors for their excellent open-source contributions.

# Training
```
python train.py
```
### 📦 Preparation of Pre-trained Weights

> **Note:** The checkpoints of our proposed method will be released here after paper acceptance. 

Our method builds upon several excellent open-source models. Before running the code, please download the following pre-trained weights from their original repositories and place them into the `./weights/` directory.

- **Stable Diffusion v2.1**: Download `v2-1_512-ema-pruned.ckpt` from [Stability AI's Hugging Face](https://huggingface.co/sd-research/stable-diffusion-2-1-base).
- **ControlNet**: Download `controlnet_sample0160000.pt` from [ControlNet's Repo](https://github.com/lllyasviel/ControlNet).
- **VAE / SeVAE**: Download `vae_sample0012000.pt` from [Original Repo].

The directory structure should look like this:
```
├── weights/
│   ├── controlnet_sample0160000.pt
│   ├── v2-1_512-ema-pruned.ckpt
│   └── vae_sample0012000.pt
└── train.py
```
