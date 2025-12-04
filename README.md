# CT-Former: Reconstructing 3D CT Scans from Single X-Rays with Transformer-Enhanced NeRF and MLG Ray Sampling

A novel deep learning framework for reconstructing high-quality 3D CT volumes from sparse or single-view X-ray projections using Neural Radiance Fields (NeRF) enhanced with Transformer-based attention mechanisms and Masked Local-Global (MLG) ray sampling strategies.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Testing & Evaluation](#testing--evaluation)
- [3D Visualization](#3d-visualization)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Performance Metrics](#performance-metrics)
- [Hardware Requirements](#hardware-requirements)
- [Dependencies](#dependencies)
- [Citation](#citation)

## 🎯 Overview

CT-Former addresses the challenge of reconstructing 3D CT volumes from limited X-ray projections, which is critical for reducing patient radiation exposure while maintaining diagnostic quality. The framework combines:

- **NeRF-based Implicit Radiodensity Fields**: Continuous 3D scene representation for resolution-independent modeling
- **Line-Segment Transformer Architecture**: Captures long-range spatial dependencies along ray segments
- **Masked Local-Global (MLG) Ray Sampling**: Efficient sampling strategy focusing computational resources on clinically informative regions
- **Latent Code Optimization**: Patient-specific adaptation during inference without retraining
- **Adversarial Training**: GAN-based loss for enforcing global anatomical realism

## ✨ Key Features

### Core Capabilities
- **Sparse-View Reconstruction**: Reconstructs 3D CT volumes from as few as 50 X-ray projections
- **Single-View Inference**: Capable of generating 3D volumes from single X-ray inputs
- **Multi-Anatomical Support**: Trained models for chest, foot, and head regions
- **High-Quality Outputs**: Preserves fine anatomical details including bone boundaries and soft tissue contrast
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs with efficient memory management

### Technical Innovations
- **Transformer-Enhanced NeRF**: Integrates attention mechanisms for better spatial coherence
- **MLG Sampling**: Reduces computational cost by 40% while maintaining reconstruction quality
- **Hash Grid Encoding**: Efficient positional encoding for 3D coordinates
- **Volume Rendering**: Physics-based rendering using ray marching and volume integration

## 🏗️ Architecture

### System Components

```
CT-Former Architecture:
├── Input: Sparse X-ray Projections (50 views)
├── Encoder: Hash Grid Positional Encoding
├── Network: Line-Segment Transformer (Lineformer)
│   ├── Line Attention Blocks
│   ├── Multi-Head Self-Attention
│   └── Feed-Forward Networks
├── Rendering: Volume Rendering with Ray Marching
├── MLG Sampling: Masked Local-Global Ray Selection
└── Output: 3D CT Volume (256×256×128 voxels)
```

### Network Architecture Details

#### 1. **Hash Grid Encoder**
- **Input**: 3D coordinates (x, y, z)
- **Encoding**: Multi-resolution hash grid with 16 levels
- **Output Dimension**: 32-dimensional feature vectors
- **Benefits**: Efficient memory usage and fast feature lookup

#### 2. **Line-Segment Transformer (Lineformer)**
- **Architecture**: 4-layer MLP with Transformer attention blocks
- **Hidden Dimension**: 32-256 (configurable)
- **Attention Mechanism**: Multi-head self-attention along ray segments
- **Line Size**: 2-32 points per segment (configurable)
- **Skip Connections**: At layer 2 for gradient flow

#### 3. **Volume Rendering**
- **Ray Sampling**: 256 coarse samples per ray
- **Fine Sampling**: Optional hierarchical sampling
- **Integration**: Numerical integration using volume rendering equation
- **Output**: Attenuation coefficients (radiodensity values)

#### 4. **MLG Ray Sampling**
- **Window Partition**: Divides projection images into windows (e.g., 32×32)
- **Local Sampling**: Dense sampling within selected windows
- **Global Sampling**: Sparse sampling from remaining regions
- **Efficiency**: Reduces ray samples by ~40% while maintaining quality

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (NVIDIA GPU with Compute Capability 3.5+)
- CUDA Toolkit 11.0+
- cuDNN 8.0+

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd major-project-test1
```

### Step 2: Create Conda Environment
```bash
conda create -n ct-former python=3.8
conda activate ct-former
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install TIGRE
TIGRE (Tomographic Iterative GPU-based Reconstruction) is included as a submodule. For Python installation:
```bash
cd TIGRE/Python
python setup.py install
cd ../..
```

### Step 5: Compile Hash Grid Encoder (Optional)
If using hash grid encoding, compile the CUDA extensions:
```bash
cd src/encoder/hashencoder
python setup.py build_ext --inplace
cd ../../..
```

## 📊 Dataset Preparation

### Data Format
The project uses `.pickle` files containing preprocessed CT data. Each pickle file contains:
- **Projections**: X-ray projection images (50 views, 256×256 pixels)
- **Angles**: Projection angles in radians
- **Geometry**: Cone-beam CT geometry parameters (DSD, DSO, detector size, etc.)
- **Ground Truth**: Original 3D CT volume (256×256×128 voxels)
- **Metadata**: Training/validation split information

### Generating Data from Raw CT Scans

#### Input Format
- **Raw Data**: `.mat` files (MATLAB format) containing CT volume data
- **Configuration**: `.yml` files with geometry parameters

#### Data Generation Script
```bash
python dataGenerator/generateData_head.py \
    --ctName head \
    --outputName head_50 \
    --dataFolder raw_data \
    --outputFolder ./data
```

#### Data Processing Pipeline
1. **Load CT Volume**: Reads `.mat` file containing Hounsfield Unit (HU) values
2. **Convert HU to Attenuation**: Transforms HU values to linear attenuation coefficients
3. **Resize Volume**: Resamples to target resolution (256×256×128)
4. **Generate Projections**: Uses TIGRE to simulate X-ray projections at specified angles
5. **Normalize Data**: Normalizes projections and volumes to [0, 1] range
6. **Save Pickle**: Serializes all data into `.pickle` format

### Supported Anatomical Regions
- **Chest**: Thoracic cavity with rib cage and soft tissues
- **Foot**: Foot anatomy with metatarsals and phalanges
- **Head**: Skull and upper cervical spine

## 🚀 Training

### Training with Standard NeRF
```bash
python train.py \
    --config config/nerf/chest_50.yaml \
    --gpu_id 1
```

### Training with Lineformer (CT-Former)
```bash
python train_mlg.py \
    --config config/Lineformer/chest_50.yaml \
    --gpu_id 1
```

### Training Configuration
Configuration files (`.yaml`) specify:
- **Network Architecture**: Type, layers, hidden dimensions
- **Encoder Settings**: Encoding type, resolution levels
- **Rendering Parameters**: Number of samples, fine sampling
- **Training Hyperparameters**: Learning rate, batch size, epochs
- **MLG Sampling**: Window size, window number

### Training Process
1. **Data Loading**: Loads `.pickle` files and creates TIGRE dataset
2. **Ray Sampling**: Samples rays from projection images (MLG or uniform)
3. **Forward Pass**: Encodes 3D points, processes through network, renders projections
4. **Loss Calculation**: MSE loss between predicted and ground truth projections
5. **Backward Pass**: Updates network weights using Adam optimizer
6. **Evaluation**: Periodic evaluation on validation set (PSNR, SSIM)
7. **Checkpointing**: Saves best models based on 3D PSNR

### Training Time
- **Chest Model**: ~8 hours on NVIDIA Tesla T4 (200 epochs)
- **Foot Model**: ~8 hours on NVIDIA Tesla P100 (200 epochs)
- **Head Model**: ~8 hours on NVIDIA Tesla P100 (200 epochs)

## 🧪 Testing & Evaluation

### Single Model Evaluation
```bash
python test.py \
    --method Lineformer \
    --category chest \
    --config config/Lineformer/chest_50.yaml \
    --weights models/chest.tar \
    --output_path output \
    --gpu_id 1
```

### Evaluation Metrics
- **Projection PSNR**: Peak Signal-to-Noise Ratio for X-ray projections
- **Projection SSIM**: Structural Similarity Index for projections
- **3D PSNR**: Volume-level PSNR comparing reconstructed and ground truth CT
- **3D SSIM**: Volume-level SSIM for structural similarity

### Quantitative Results
| Category | PSNR (dB) | SSIM |
|----------|-----------|------|
| Chest    | 32.45     | 0.9521 |
| Foot     | 33.25     | 0.9642 |
| Head     | 35.12     | 0.9785 |

## 🎨 3D Visualization

### Static Volume Rendering
```bash
python 3D_vis/3D_vis_chest.py --gpu_id 1
```

### Animated GIF Generation
```bash
python 3D_vis/3D_vis_chest_gif.py --gpu_id 1
```

### Visualization Parameters
- **Elevation Angle**: Viewing angle for volume rendering
- **Sigma**: Opacity threshold for volume rendering
- **Alpha**: Transparency level for semi-transparent rendering
- **Rotation**: 360-degree rotation for GIF animations (120 frames)

### Output Locations
- **Static Images**: `3D_vis/{category}/` directory
- **GIF Animations**: `3D_vis/{category}/elevation_*_sigma_*_alpha_*/` directories

## 📁 Project Structure

```
major-project-test1/
├── config/                 # Configuration files for different methods
│   ├── Lineformer/        # CT-Former configurations
│   ├── nerf/              # Standard NeRF configs
│   ├── FDK/               # Filtered Back Projection configs
│   ├── SART/              # Iterative reconstruction configs
│   └── ...
├── data/                  # Preprocessed dataset (.pickle files)
│   ├── chest_50.pickle
│   ├── foot_50.pickle
│   └── head_50.pickle
├── dataGenerator/         # Data preprocessing scripts
│   ├── generateData_head.py
│   └── raw_data/         # Raw CT scan data
├── models/                # Trained model checkpoints
│   ├── chest.tar
│   ├── foot.tar
│   └── head.tar
├── src/                   # Source code
│   ├── config/           # Configuration loading
│   ├── dataset/          # Data loading and MLG sampling
│   ├── encoder/          # Positional encoders (hash grid, frequency)
│   ├── loss/             # Loss functions
│   ├── network/          # Network architectures (Lineformer, NeRF)
│   ├── render/           # Volume rendering
│   ├── trainer.py        # Standard training loop
│   ├── trainer_mlg.py    # MLG training loop
│   └── utils/            # Utility functions
├── 3D_vis/               # 3D visualization scripts
│   ├── 3D_vis_chest.py
│   ├── 3D_vis_chest_gif.py
│   └── ...
├── web_gui/              # Web-based GUI (see WEB_GUI_README.md)
├── logs/                 # Training logs and TensorBoard files
├── output/               # Test outputs and reconstructions
├── test.py               # Main testing script
├── train.py              # Standard training script
├── train_mlg.py          # MLG training script
└── requirements.txt      # Python dependencies
```

## 🔬 Technical Details

### Neural Radiance Fields for CT
CT-Former adapts NeRF for medical imaging by:
- **Radiodensity Mapping**: Maps 3D coordinates to attenuation coefficients (μ) instead of RGB colors
- **Volume Rendering**: Uses Beer-Lambert law for X-ray attenuation
- **Sparse Supervision**: Trained on sparse X-ray projections rather than dense views

### Line-Segment Transformer
The Transformer architecture processes ray segments:
1. **Ray Partitioning**: Divides sampled points along rays into segments
2. **Segment Encoding**: Each segment contains `line_size` consecutive points
3. **Self-Attention**: Computes attention within each segment
4. **Feature Aggregation**: Merges segment features back to point-level

### MLG Ray Sampling Strategy
1. **Window Partition**: Divides 256×256 projection into windows (e.g., 32×32)
2. **Window Selection**: Randomly selects `window_num` windows with valid projections
3. **Local Sampling**: Dense sampling from selected windows (all pixels)
4. **Global Sampling**: Sparse sampling from remaining windows (`n_rays` samples)
5. **Efficiency**: Reduces total samples while maintaining coverage

### Latent Code Optimization
During inference:
1. **Initial Prediction**: Generate volume using trained model
2. **Latent Refinement**: Optimize patient-specific latent codes
3. **Iterative Refinement**: Update latent codes to minimize projection error
4. **Final Reconstruction**: Generate refined 3D volume

### Adversarial Training
- **Discriminator**: CNN that distinguishes real vs. reconstructed CT volumes
- **Generator**: The NeRF network generating 3D volumes
- **Loss**: Combines MSE loss with adversarial loss
- **Benefit**: Enforces anatomical realism beyond pixel-level accuracy

## 📈 Performance Metrics

### Computational Efficiency
- **Training Time**: ~8 hours per model (200 epochs)
- **Inference Time**: 2-3 minutes per patient (including latent optimization)
- **Memory Usage**: ~4-8 GB GPU memory (depending on batch size)
- **MLG Efficiency**: 40% reduction in ray samples vs. uniform sampling

### Quality Metrics
- **PSNR Improvement**: +11.49 dB over FDK, +5.98 dB over baseline NeRF
- **SSIM Improvement**: Significant improvement in structural similarity
- **Visual Quality**: Preserves fine details (bone boundaries, soft tissue contrast)

## 💻 Hardware Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 4GB VRAM (e.g., GTX 1050 Ti)
- **RAM**: 16GB system RAM
- **Storage**: 50GB free space

### Recommended Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3060, Tesla T4, Tesla P100)
- **RAM**: 32GB system RAM
- **Storage**: 100GB+ SSD for faster data loading

### Tested Hardware
- **Training**: NVIDIA Tesla T4 (chest), NVIDIA Tesla P100 (foot, head)
- **Inference**: NVIDIA GeForce RTX 3050 Laptop GPU

## 📚 Dependencies

### Core Libraries
- **PyTorch**: Deep learning framework (CUDA-enabled)
- **TIGRE**: GPU-accelerated CT reconstruction library
- **NumPy**: Numerical computing
- **scikit-image**: Image processing
- **matplotlib**: Visualization
- **tqdm**: Progress bars

### See `requirements.txt` for complete list

## 📄 Citation

If you use CT-Former in your research, please cite:

```bibtex
@article{ctformer2025,
  title={CT-Former: Reconstructing 3D CT Scans from Single X-Rays with Transformer-Enhanced NeRF and MLG Ray Sampling},
  author={Gaikwad, Vidya and Nipanikar, Sanchitsai and Nagarkar, Shreyas and Padalkar, Shivam and Parkar, Muhammad},
  journal={IEEE Transactions on Medical Imaging},
  year={2025}
}
```

---

For web GUI documentation, see [WEB_GUI_README.md](WEB_GUI_README.md)
