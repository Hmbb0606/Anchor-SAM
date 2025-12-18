# Anchor-SAM: Active Mining of Latent Anchors from SAM Encoder for Road Extraction

<div align="center">

**Wenhai Li**, **Xiaohui Huang**, **Xiaofei Yang**, **Yicong Zhou** (Senior Member, IEEE), **Jiangtao Peng** (Senior Member, IEEE), **Yifang Ban** (Senior Member, IEEE), **Nan Jiang**

*Institute of Data Science and Deep Learning, East China Jiaotong University*

</div>

## 📖 Introduction

This repository is the official implementation of the paper **"Anchor-SAM: Active Mining of Latent Anchors from SAM Encoder for Road Extraction"**, currently under review at **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)**.

**Anchor-SAM** leverages the powerful feature representation of the **Segment Anything Model (SAM) 2.1**, proposing a novel framework to actively mine latent semantic anchors from the encoder. This approach significantly enhances the connectivity and topological completeness of road extraction in high-resolution remote sensing imagery.

## 📂 Project Structure

The directory structure is organized as follows:

```text
./
├── config/            # Model configurations (YAML)
├── model/             # Model architectures and SOTA comparisons
├── utils/             # Utility functions (Data loading, losses, metrics)
├── pretrain_weight/   # Pre-trained weights storage
├── train.py           # Main training script
├── inference.py       # Inference/Testing script
├── model_profiler.py  # Performance profiling (FLOPs, FPS)
└── demo.py            # Demo script
```

## 🛠️ Environment Setup

### 1. Installation

Our environment setup follows the official implementation of [RS-Mamba](https://github.com/NJU-LHRS/Official_Remote_Sensing_Mamba) and [VMamba](https://github.com/MzeroMiko/VMamba).

**Step 1: Create a virtual environment**

```bash
conda create -n anchorsam python=3.10
conda activate anchorsam
```

**Step 2: Install PyTorch and CUDA**

```bash
# Install PyTorch
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.1.0
```

**Step 3: Install Dependencies**

```bash
pip install -r requirements.txt
pip install albumentations comet_ml thop prettytable openpyxl pyyaml
```

### 2. Download Pre-trained Weights

**SAM 2.1 Weights**

Please refer to the [official SAM 2 repository](https://github.com/facebookresearch/sam2) to download the pre-trained weights (e.g., `sam2.1_hiera_tiny.pt`, `sam2.1_hiera_large.pt`) and place them in `./pretrain_weight/`.

**Backbones (Optional)**

If you plan to train comparison models (e.g., D-LinkNet, DeepLabV3+), run the following script to download ResNet weights:

```bash
bash pretrain_weight/download.bash
```

## 📊 Data Preparation

Please organize your datasets (e.g., DeepGlobe, Massachusetts) as follows. The `root_dir` in `config/*.yaml` should point to the parent folder.

```text
/path/to/dataset/
    ├── DeepGlobe/
    │   ├── train/
    │   │   ├── image/   # .jpg or .png
    │   │   └── label/   # Binary masks (0 and 255/1)
    │   ├── val/
    │   │   ├── image/
    │   │   └── label/
    │   └── test/
    │       ├── image/
    │       └── label/
```

Update the `root_dir` in `config/Anchor-SAM.yaml` to match your data path:

```yaml
dataset_name: 'DeepGlobe'
root_dir: '/your/custom/path/to/datasets'
image_size: 1024
```

## 🚀 Usage

### 1. Training

To train **Anchor-SAM**, use the `train.py` script. You can modify hyperparameters in the YAML config files.

```bash
# Train Anchor-SAM (Default)
python train.py -c config/Anchor-SAM.yaml

# Train Comparison Models (e.g., UNet)
python train.py -c config/UNet.yaml
```

* **Logging**: Training metrics are automatically logged to **Comet ML** (offline mode by default) and saved locally to `training_metrics.xlsx`.
* **Checkpoints**: Best models are saved in `./logs_weights/`.

### 2. Inference

Run inference on the test set to generate binary masks and evaluate performance.

```bash
python inference.py -c config/Anchor-SAM.yaml
```

* **Outputs**: Results (segmentation masks, probability maps, error maps) are saved in the `./results/` folder.

### 3. Profiling

Calculate **FLOPs**, **Parameters**, and **FPS** for the model using the profiler script.

```bash
python model_profiler.py
```

*Note: Ensure `CONFIG_FILE_PATH` inside `model_profiler.py` points to the desired model config.*

## ⚙️ Configuration Details

The project uses `YAML` files for flexible configuration. Example (`config/Anchor-SAM.yaml`):

```yaml
model_classname: 'SAM2Point1Net'
model_args:
  model_size: 'large'
  # Path to the SAM 2.1 checkpoint
  pretrained_weights_path: './pretrain_weight/sam2.1_hiera_large.pt'

training_strategy:
  mode: 'train_all'  # Options: 'train_all' or 'adapter_only'

inference:
  threshold: 0.5
  save_outputs: true
  save_error_map: true  # Visualizes TP, FP, TN, FN
```

## 📝 Citation

If you find this code or research helpful, please consider citing our paper (BibTeX will be updated upon acceptance):

```bibtex
@article{li2025anchorsam,
  title={Anchor-SAM: Active Mining of Latent Anchors from SAM Encoder for Road Extraction},
  author={Li, Wenhai and Peng, Jiangtao and Huang, Xiaohui and Ban, Yifang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)},
  year={2025},
  note={Under Review}
}
```

## 🤝 Acknowledgements

* Thanks to the authors of [Segment Anything Model 2](https://github.com/facebookresearch/segment-anything-2) for their open-source contribution.
* We also acknowledge the open-source implementations of comparison methods used in this study.

<div align="center">
<i>Created by Wenhai Li</i>
</div>
