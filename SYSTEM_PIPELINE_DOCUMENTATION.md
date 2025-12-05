# CT-Former System Pipeline: Complete Flow Documentation

## Overview
This document provides a detailed explanation of the entire CT-Former system pipeline, from raw CT data input through preprocessing, training, inference, and visualization, to final reconstructed 3D volumes and metric calculations.

---

## Table of Contents
1. [Data Preprocessing Pipeline](#1-data-preprocessing-pipeline)
2. [Training Pipeline](#2-training-pipeline)
3. [Inference/Testing Pipeline](#3-inferencetesting-pipeline)
4. [Visualization Pipeline](#4-visualization-pipeline)
5. [Metric Calculation](#5-metric-calculation)

---

## 1. Data Preprocessing Pipeline

### 1.0 Stage 0: Raw Input (Original CT Scan Data)

**Script**: None (raw data input)

**What it is:**
- Original 3D CT scan volume directly from medical scanner
- **Format**: Binary `.raw` files (e.g., `vis_male_128x256x256_uint8.raw`, `statue_leg_341x341x93_uint8.raw`)
- **Content**: Raw voxel values stored as sequential binary data
- **Data Type**: 
  - `uint8` (0-255) for some scans
  - `int16` (-32768 to 32767) for others
- **Dimensions**: Varies by scan:
  - Head: `128×256×256` (depth×height×width)
  - Foot: `341×341×93`
  - Chest: Varies
- **Location**: `dataGenerator/raw_data/{category}/` (before conversion)

**Why `.raw` format:**
- Direct output from CT scanners (no headers/metadata)
- Efficient storage format (just sequential voxel values)
- Requires external knowledge of dimensions and data type to read correctly
- Common in medical imaging workflows

---

### 1.1 Stage 1: Conversion to MATLAB Format (.mat)

**Script**: `dataGenerator/data_vis_{category}.py` (or manual conversion script)

**Process**:
```python
# Read raw binary file
raw_data = np.fromfile('vis_male_128x256x256_uint8.raw', dtype=np.uint8)

# Reshape to 3D volume (dimensions must be known)
img_data = raw_data.reshape(256, 256, 128)  # [height, width, depth]

# Normalize to [0, 1] range
img_data = np.float32(img_data) / 255.0

# Save as MATLAB .mat file
scipy.io.savemat('raw_data/head/img.mat', {'img': img_data})
```

**Output**:
- **File**: `dataGenerator/raw_data/{category}/img.mat`
- **Content**: Dictionary with key `'img'` containing 3D numpy array
- **Format**: Float32, normalized to [0, 1] range
- **Shape**: 3D array matching original scan dimensions

**Why `.mat` format:**
- Standard format for medical imaging workflows
- Easy to load with `scipy.io.loadmat()` in Python
- Preserves metadata and array structure
- Compatible with MATLAB/Python toolchains
- Required by the data generation pipeline (`generateData_*.py`)

---

### 1.2 Stage 2: Main Data Generation Pipeline

**Script Execution**: `dataGenerator/generateData_{category}.py` or `dataGenerator/generateData.py`
- For head: `dataGenerator/generateData_head.py`
- For PNG X-ray images: `dataGenerator/generateData_from_png.py`

#### Step 2.1: Load Configuration
**Script Location**: `dataGenerator/generateData_{category}.py` (lines 26-49, 163-177)
**File**: `dataGenerator/raw_data/{category}/config.yml`
- Defines CT scanner geometry parameters:
  - `DSD`: Distance Source to Detector (mm)
  - `DSO`: Distance Source to Origin (mm)
  - `nDetector`: Detector pixel dimensions [H, W]
  - `dDetector`: Detector pixel size (mm)
  - `nVoxel`: Target volume dimensions [X, Y, Z] (e.g., `[256, 256, 128]`)
  - `dVoxel`: Voxel size (mm)
  - `numTrain`: Number of training projections (default: 50)
  - `numVal`: Number of validation projections (default: 50)
  - `totalAngle`: Total scanning angle (degrees, typically 180°)
  - `startAngle`: Starting angle (degrees, typically 0°)
  - `convert`: Boolean flag to enable HU→attenuation conversion
  - `rescale_slope`: Rescale slope for HU calculation
  - `rescale_intercept`: Rescale intercept for HU calculation
  - `normalize`: Boolean flag to normalize to [0, 1]
  - `noise`: Noise level (0 = no noise)
  - `randomAngle`: Whether to use random or uniform angle distribution

#### Step 2.2: Load and Convert CT Image
**Script Location**: `dataGenerator/generateData_{category}.py` (lines 118-160)
**Function**: `loadImage(dirname, nVoxels, convert, rescale_slope, rescale_intercept, normalize=True)`
**Helper Function**: `convert_to_attenuation()` (lines 89-115)

**Process**:
1. **Load `.mat` file**:
   ```python
   test_data = scipy.io.loadmat(dirname)  # Loads img.mat
   image_ori = test_data["img"].astype(np.float32)  # [H, W, D]
   ```

2. **HU to Attenuation Conversion** (`convert_to_attenuation()`):
   ```python
   # Step 1: Calculate Hounsfield Units
   HU = data * rescale_slope + rescale_intercept
   
   # Step 2: Convert HU to linear attenuation coefficients (μ)
   mu_water = 0.206  # cm⁻¹
   mu_air = 0.0004   # cm⁻¹
   mu = mu_water + (mu_water - mu_air) / 1000 * HU
   ```
   
   **Why convert HU to attenuation:**
   - CT scanners output Hounsfield Units (HU): relative scale where water=0, air=-1000
   - X-ray physics requires linear attenuation coefficients (μ): physical units (cm⁻¹)
   - NeRF models learn μ values, not HU (physics-based representation)
   - Enables accurate forward projection simulation using Beer-Lambert law

3. **Volume Resizing/Resampling**:
   ```python
   zoom_x = nVoxels[0] / imageDim[0]  # e.g., 256 / 128 = 2.0
   zoom_y = nVoxels[1] / imageDim[1]  # e.g., 256 / 256 = 1.0
   zoom_z = nVoxels[2] / imageDim[2]  # e.g., 128 / 256 = 0.5
   
   image = scipy.ndimage.interpolation.zoom(
       image, (zoom_x, zoom_y, zoom_z), order=3, prefilter=False
   )
   ```
   
   **Target Shape**: `[256, 256, 128]` (standardized across all categories)
   
   **Why resize:**
   - Standardizes input dimensions for neural network (consistent batch processing)
   - Reduces memory usage (128 slices vs. 256+ slices)
   - Balances resolution vs. computational cost
   - Ensures consistent training across different scan types

4. **Normalization to [0, 1] Range**:
   ```python
   image_min = np.min(image)
   image_max = np.max(image)
   image = (image - image_min) / (image_max - image_min)
   ```
   
   **Why normalize:**
   - Neural networks train better with normalized inputs (prevents gradient issues)
   - Consistent scale across different CT scans (different scanners, different ranges)
   - Standard practice in deep learning (improves convergence)
   - Prevents numerical instability from large values

**Output**: Normalized 3D volume `[256, 256, 128]` with values in [0, 1] range

#### Step 2.3: Generate X-ray Projections (Forward Projection)
**Script Location**: `dataGenerator/generateData_{category}.py` (lines 183-205)
**Function**: `tigre.Ax()` (TIGRE library forward projection)
**Geometry Class**: `ConeGeometry_special` (lines 52-82)

**Input**: 
- CT volume: `[256, 256, 128]` (attenuation coefficients μ)
- Geometry: `ConeGeometry_special` object (defines scanner geometry)
- Angles: Array of projection angles (radians)

**Process** (for each angle):
1. **X-ray Source Positioning**:
   - Source position: `DSO * [cos(angle), sin(angle), 0]` (circular trajectory)
   - Detector position: Fixed plane at distance `DSD` from source

2. **Ray Casting**:
   - Cast rays from source through 3D volume to detector pixels
   - Each ray samples attenuation coefficients along its path

3. **Line Integral Computation** (Beer-Lambert Law):
   ```
   I = I₀ * exp(-∫ μ(s) ds)
   ```
   - Computes integral of attenuation along each ray
   - Simulates X-ray absorption through tissue

4. **Projection onto Detector**:
   - Maps line integrals to detector pixel grid `[512, 512]` or `[256, 256]`
   - Output: Single X-ray projection image

**Training Angles**: 
- **Uniform**: `np.linspace(0, totalAngle, numTrain)` (evenly spaced)
- **Random**: `np.random.rand(numTrain) * totalAngle` (random distribution)

**Validation Angles**: Randomly sampled from full range

**Output**: 
- Training projections: `[50, 512, 512]` or `[50, 256, 256]` (50 X-ray images)
- Validation projections: `[50, 512, 512]` or `[50, 256, 256]` (50 X-ray images)
- Angles: Array of angles in radians `[50]`

**Why generate projections:**
- NeRF learns to reconstruct 3D from sparse 2D views (inverse problem)
- 50 projections simulate sparse CT acquisition (real CT uses 1000+ projections)
- Training data: pairs of (projection, angle) → 3D volume
- Validation data: test reconstruction quality on unseen angles

#### Step 2.4: Add Noise (Optional)
**Script Location**: `dataGenerator/generateData_{category}.py` (lines 189-194, 200-205)
**Function**: `CTnoise.add()` (from TIGRE library)
- **Poisson Noise**: Simulates photon counting statistics (`Poisson=1e5`)
- **Gaussian Noise**: Simulates electronic noise (`Gaussian=[0, noise_level]`)
- **Purpose**: Makes projections more realistic (real X-ray images have noise)
- **When Applied**: Only if `noise > 0` in config and `normalize=True`

#### Step 2.5: Save Pickle File
**Script Location**: `dataGenerator/generateData_{category}.py` (lines 246-251)
**Output**: `data/{category}_50.pickle`

**Structure**:
```python
{
    "image": np.array([256, 256, 128]),           # Ground truth CT volume (attenuation μ)
    "train": {
        "projections": np.array([50, 512, 512]),  # Training X-ray projections
        "angles": np.array([50])                  # Projection angles (radians)
    },
    "val": {
        "projections": np.array([50, 512, 512]),  # Validation X-ray projections
        "angles": np.array([50])                   # Validation angles (radians)
    },
    "numTrain": 50,
    "numVal": 50,
    # Geometry parameters (for ray generation during training)
    "DSD": float,                                  # Distance Source to Detector (m)
    "DSO": float,                                  # Distance Source to Origin (m)
    "nDetector": [512, 512],                      # Detector pixel dimensions
    "dDetector": [float, float],                  # Detector pixel size (m)
    "nVoxel": [256, 256, 128],                    # Volume dimensions
    "dVoxel": [float, float, float],              # Voxel size (m)
    "offOrigin": [float, float, float],            # Volume offset from origin
    "offDetector": [float, float, float],          # Detector offset
    "mode": "cone",                                # "cone" or "parallel"
    "accuracy": float,                             # Forward projection accuracy
    "filter": str                                  # Reconstruction filter type
}
```

**Why `.pickle` format:**
- Stores everything needed for training in single file (volume + projections + geometry)
- Preserves numpy arrays and Python objects efficiently
- Fast loading during training (no parsing overhead)
- Standard format for PyTorch data loaders
- Contains both input (projections) and ground truth (volume) for supervised learning

---

### 1.3 Complete Preprocessing Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 0: Raw Input                                              │
│ • Format: .raw binary files                                    │
│ • Content: Sequential voxel values (uint8 or int16)            │
│ • Dimensions: Varies (e.g., 128×256×256 for head)             │
│ • Location: dataGenerator/raw_data/{category}/                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           │ [Binary read + reshape + normalize]
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: MATLAB Format                                          │
│ • Format: .mat files                                            │
│ • Content: 3D numpy array (float32, [0, 1] range)              │
│ • Structure: {'img': np.array([H, W, D])}                      │
│ • Location: dataGenerator/raw_data/{category}/img.mat          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           │ [generateData_{category}.py]
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Load & Convert                                        │
│ • Load .mat file                                               │
│ • Convert HU → Attenuation (μ)                                 │
│ • Resize to [256, 256, 128]                                    │
│ • Normalize to [0, 1]                                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           │ [TIGRE forward projection]
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Generate Projections                                  │
│ • Training: 50 X-ray projections [50, 512, 512]               │
│ • Validation: 50 X-ray projections [50, 512, 512]            │
│ • Angles: Array of projection angles (radians)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           │ [Package everything]
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4: Final Output                                          │
│ • Format: .pickle file                                         │
│ • Content: Volume + Projections + Angles + Geometry           │
│ • Location: data/{category}_50.pickle                         │
│ • Ready for training!                                          │
└─────────────────────────────────────────────────────────────────┘
```

**Key Transformations:**
1. **Raw → .mat**: Standardizes format, enables easy loading
2. **HU → μ**: Physics-based conversion for accurate simulation
3. **Resize**: Standardizes dimensions, reduces memory
4. **Normalize**: Improves training stability
5. **Generate Projections**: Creates training data (sparse views → 3D)
6. **Pickle**: Efficient storage for training pipeline

This preprocessing pipeline converts raw CT scans into a format suitable for training NeRF models to reconstruct 3D volumes from sparse X-ray projections.

---

## 2. Training Pipeline

### 2.1 Entry Point: `train.py`

**Script**: `train.py`

#### Step 2.1: Configuration Loading
**Script Location**: `train.py`
**Function**: `load_config(args.config)`
- **Config File**: `config/{method}/{category}_50.yaml`
  - Methods: `Lineformer`, `tensorf`, `nerf`
- **Key Parameters**:
  - `exp.datadir`: Path to `.pickle` file
  - `network.net_type`: `"Lineformer"` or `"mlp"`
  - `encoder.type`: `"hashgrid"`, `"tensorf"`, or `"freq"`
  - `train.n_rays`: Number of rays per batch (default: 1024)
  - `train.n_batch`: Batch size (default: 1)
  - `train.epoch`: Total epochs (default: 200)
  - `render.n_samples`: Coarse samples per ray (default: 256)
  - `render.n_fine`: Fine samples per ray (default: 0 or 128)
  - `render.netchunk`: Chunk size for network inference (default: 409600)

#### Step 2.2: Dataset Initialization
**Script Location**: `src/dataset/tigre.py`
**Class**: `TIGREDataset` (from `src/dataset/tigre.py`)
- **Loading**:
  - Opens `.pickle` file
  - Extracts geometry parameters → `ConeGeometry` object
  - Loads projections: `[50, 256, 256]` → GPU tensor
  - Loads ground truth volume: `[256, 256, 128]` → GPU tensor
- **Ray Generation** (`get_rays()`):
  - For each projection angle:
    - Computes camera pose: `angle2pose(DSO, angle)` → 4×4 transformation matrix
    - Creates pixel grid: `[256, 256]`
    - Converts pixel coordinates to 3D rays:
      - Origin: `rays_o = pose[:3, -1]` (X-ray source position)
      - Direction: `rays_d = pose[:3, :3] @ dirs` (normalized direction vectors)
    - Output: `[50, 256, 256, 6]` (origin + direction)
  - Appends near/far bounds: `[50, 256, 256, 8]`
- **Voxel Grid Generation** (`get_voxels()`):
  - Creates 3D coordinate grid: `[256, 256, 128, 3]`
  - Each voxel contains `[x, y, z]` world coordinates
- **Training Data Sampling** (`__getitem__()`):
  - Selects random valid pixels (where projection > 0)
  - Samples `n_rays` rays (default: 1024)
  - Returns: `{"rays": [1024, 8], "projs": [1024]}`

#### Step 2.3: Network Initialization
**Script Location**: `train.py`
**Function**: `get_network()` → `Lineformer` or `DensityNetwork`
- **Encoder** (`get_encoder()`):
  - **Hash Grid Encoder** (`src/encoder/hashgrid_pytorch.py`):
    - Multi-resolution hash tables (16 levels)
    - Input: 3D coordinates `[N, 3]`
    - Output: Feature vectors `[N, 32]`
  - **TensorF Encoder**: Tensor decomposition-based encoding
  - **Freq Encoder**: Fourier feature encoding
- **Network Architecture**:
  - **Lineformer** (`src/network/Lineformer.py`):
    **Script Location**: `src/network/Lineformer.py`
    - Input: `[N_rays * N_samples, 3]` (3D coordinates)
    - Encoder: `[N_rays * N_samples, 32]`
    - Line-segment Transformer:
      - **Ray Partitioning**: Groups points into segments of `line_size` (default: 16)
      - **Line Attention Block**:
        - Multi-head self-attention within each segment
        - Positional embeddings for segment positions
        - Feed-forward network (FFN)
      - **Skip Connections**: At layer 4 (concatenates input features)
    - Output: `[N_rays * N_samples, 1]` (attenuation coefficients μ)
  - **DensityNetwork** (MLP baseline):
    - 8-layer MLP with skip connections
    - Output: Attenuation coefficients

#### Step 2.4: Training Loop
**Script Location**: `train.py`
**Class**: `BasicTrainer` (inherits from `Trainer`)
**Render Function**: `src/render/` (volume rendering functions)

**Per Epoch**:
1. **Data Loading**:
   - `DataLoader` batches training samples
   - Each batch: `{"rays": [1024, 8], "projs": [1024]}`

2. **Forward Pass** (`compute_loss()`):
   - **Ray Sampling** (`render()` function):
     - Extracts ray origins, directions, near, far from `rays`
     - **Coarse Sampling**:
       - Uniformly samples `n_samples` points along each ray (default: 256)
       - Points: `pts = rays_o + rays_d * z_vals` → `[1024, 256, 3]`
     - **Network Inference** (`run_network()`):
       - Flattens points: `[1024, 256, 3]` → `[262144, 3]`
       - Processes in chunks of `netchunk` (409600)
       - Network forward: `[262144, 3]` → `[262144, 1]` (attenuation μ)
       - Reshapes: `[1024, 256, 1]`
     - **Volume Rendering** (`raw2outputs()`):
       - Computes distances: `dists = z_vals[1:] - z_vals[:-1]`
       - Integrates attenuation: `acc = sum(μ * dists)` (Beer-Lambert law)
       - Computes weights for fine sampling: `weights = |μ[i+1] - μ[i]|`
     - **Fine Sampling** (if `n_fine > 0`):
       - PDF sampling based on weights
       - Additional `n_fine` samples in high-density regions
       - Re-runs network on fine samples
     - **Output**: `acc` (projected attenuation) → `[1024]`

3. **Loss Calculation**:
   - **MSE Loss**: `loss = mean((projs_pred - projs_gt)^2)`
   - Backpropagation and optimizer step

4. **Evaluation** (every `i_eval` epochs):
   - **Projection Evaluation**:
     - Renders all validation projections: `[50, 256, 256]`
     - Computes PSNR: `get_psnr(projs_pred, projs_gt)`
   - **3D Volume Evaluation**:
     - Runs network on all voxels: `[256, 256, 128, 3]` → `[256, 256, 128, 1]`
     - Computes 3D PSNR: `get_psnr_3d(image_pred, image_gt)`
     - Saves best model if PSNR improves

5. **Checkpoint Saving**:
   - Saves: `{"epoch", "network", "network_fine", "optimizer"}`
   - Location: `exp/{expname}/{timestamp}/ckpt.tar`

---

## 3. Inference/Testing Pipeline

### 3.1 Entry Point: `test.py`

**Script**: `test.py`

#### Step 3.1: Load Model
**Script Location**: `test.py`
- Loads checkpoint: `models/{category}.tar`
- Initializes network with same architecture as training
- Loads weights: `model.load_state_dict(ckpt["network"])`

#### Step 3.2: Load Validation Dataset
**Script Location**: `test.py`
- Same as training: `TIGREDataset(path, type="val")` (from `src/dataset/tigre.py`)
- Loads validation projections and angles

#### Step 3.3: Inference (`eval_step()`)
**Script Location**: `test.py`

**3.3.1 Projection Rendering**:
- For each validation projection (50 total):
  - Flattens rays: `[256, 256, 8]` → `[65536, 8]`
  - Processes in batches of `n_rays` (1024)
  - Renders projection: `render()` → `acc` → `[65536]`
  - Reshapes: `[256, 256]`
- Concatenates: `[50, 256, 256]`

**3.3.2 3D Volume Reconstruction**:
- **Input**: Voxel grid `[256, 256, 128, 3]` (all 3D coordinates)
- **Network Forward**:
  - Flattens: `[256, 256, 128, 3]` → `[8388608, 3]`
  - Processes in chunks: `run_network()` → `[8388608, 1]`
  - Reshapes: `[256, 256, 128]`
- **Output**: Reconstructed CT volume `[256, 256, 128]`

#### Step 3.4: Metric Calculation
**Script Location**: `test.py`
**Metric Functions**: `src/utils/util.py`
- **Projection Metrics**:
  - `proj_psnr = get_psnr(projs_pred, projs_gt)` (from `src/utils/util.py`)
  - `proj_ssim = get_ssim(projs_pred, projs_gt)` (from `src/utils/util.py`)
- **3D Volume Metrics**:
  - `psnr_3d = get_psnr_3d(image_pred, image_gt)` (from `src/utils/util.py`)
  - `ssim_3d = get_ssim_3d(image_pred, image_gt)` (from `src/utils/util.py`)

#### Step 3.5: Save Results
**Script Location**: `test.py`
**Output Directory**: `output/{method}/{category}/`
- **Projections**:
  - `proj_pred/`: Predicted projections `[50, 256, 256]` → PNG files
  - `proj_gt/`: Ground truth projections → PNG files
- **3D Volume Slices**:
  - `CT/H/ct_pred/`, `CT/H/ct_gt/`: Height slices (256 images)
  - `CT/W/ct_pred/`, `CT/W/ct_gt/`: Width slices (256 images)
  - `CT/L/ct_pred/`, `CT/L/ct_gt/`: Length slices (128 images)

---

## 4. Visualization Pipeline

### 4.1 Static 3D Visualization: `3D_vis/3D_vis_{category}.py`

**Script**: `3D_vis/3D_vis_{category}.py` (e.g., `3D_vis/3D_vis_chest.py`, `3D_vis/3D_vis_head.py`)

#### Step 4.1: Load Reconstructed Volume
**Script Location**: `3D_vis/3D_vis_{category}.py`
- Loads from `output/{method}/{category}/CT/` or directly from model inference
- Volume shape: `[256, 256, 128]`

#### Step 4.2: Volume Rendering
**Script Location**: `3D_vis/3D_vis_{category}.py`
- **Marching Cubes** (`skimage.measure.marching_cubes`):
  - Input: Volume `[256, 256, 128]`, threshold (e.g., 0.55)
  - Output: Mesh vertices and faces
- **3D Plotting** (`matplotlib`):
  - Creates 3D mesh with transparency
  - Sets camera view (elevation, azimuth)
  - Saves PNG: `3D_vis/{category}/3d_{category}_{params}.png`

### 4.2 Dynamic GIF Visualization: `3D_vis/3D_vis_{category}_gif.py`

**Script**: `3D_vis/3D_vis_{category}_gif.py` (e.g., `3D_vis/3D_vis_chest_gif.py`, `3D_vis/3D_vis_head_gif.py`)

#### Step 4.1: Load Volume
**Script Location**: `3D_vis/3D_vis_{category}_gif.py`
- Same as static visualization

#### Step 4.2: Generate Frames
**Script Location**: `3D_vis/3D_vis_{category}_gif.py`
- **Loop** over rotation angles (e.g., 0° to 360° in 120 steps):
  - For each angle:
    - Applies rotation to mesh
    - Renders frame: `[width, height, 3]` RGB image
    - Saves: `angle_{i}.png`
  - Progress: `tqdm` shows `i/120 [time<remaining]`

#### Step 4.3: Create GIF
**Script Location**: `3D_vis/3D_vis_{category}_gif.py`
- **Function**: `imageio.mimsave()`
- Combines frames: `angle_*.png` → `{category}_animation.gif`
- Parameters: Duration per frame, loop count
- **Output**: `3D_vis/{category}/{category}_animation.gif`

---

## 5. Metric Calculation

### 5.1 PSNR (Peak Signal-to-Noise Ratio)

**Script Location**: `src/utils/util.py`

#### 2D PSNR: `get_psnr(x, y)`
**File**: `src/utils/util.py`
- **Input**: Two tensors `x`, `y` (same shape)
- **Process**:
  1. Normalize to [0, 1]: `x_norm = (x - min(x)) / (max(x) - min(x))`
  2. Compute MSE: `mse = mean((x_norm - y_norm)^2)`
  3. PSNR: `PSNR = -10 * log10(mse)`
- **Output**: Scalar PSNR value (dB)
- **Usage**: Projection quality assessment

#### 3D PSNR: `get_psnr_3d(arr1, arr2)`
**File**: `src/utils/util.py`
- **Input**: 3D volumes `[H, W, L]`
- **Process**:
  1. Compute MSE across all voxels: `mse = mean((arr1 - arr2)^2)`
  2. PSNR: `PSNR = 20 * log10(1.0 / sqrt(mse))`
- **Output**: Scalar PSNR value (dB)
- **Usage**: 3D volume reconstruction quality

### 5.2 SSIM (Structural Similarity Index)

**Script Location**: `src/utils/util.py`

#### 2D SSIM: `get_ssim(img1, img2)`
**File**: `src/utils/util.py`
- **Input**: Two images `[H, W]` or `[B, H, W]`
- **Process**:
  1. Convert to uint8: `[0, 1]` → `[0, 255]`
  2. Apply Gaussian filter (11×11, σ=1.5)
  3. Compute local means, variances, covariance
  4. SSIM formula:
     ```
     SSIM = (2*μ1*μ2 + C1) * (2*σ12 + C2) / ((μ1² + μ2² + C1) * (σ1² + σ2² + C2))
     where C1 = (0.01*255)², C2 = (0.03*255)²
     ```
- **Output**: Scalar SSIM value [0, 1] (higher is better)

#### 3D SSIM: `get_ssim_3d(arr1, arr2)`
**File**: `src/utils/util.py`
- **Input**: 3D volumes `[H, W, L]`
- **Process**:
  1. Computes SSIM along each axis:
     - Depth: Transpose to `[L, H, W]`, compute SSIM for each slice
     - Height: Transpose to `[H, L, W]`, compute SSIM for each slice
     - Width: Compute SSIM for each slice `[H, W]`
  2. Averages SSIM across all three axes
- **Output**: Scalar SSIM value [0, 1]

---

## 6. MLG (Masked Local-Global) Sampling Strategy

### 6.1 Training with MLG: `train_mlg.py`

**Script**: `train_mlg.py`
**Dataset**: `TIGREDataset_MLG` (from `src/dataset/tigre_mlg.py`)

#### Step 6.1: Window Partitioning
**Script Location**: `src/dataset/tigre_mlg.py`
- **Input**: Projection `[256, 256]`, Rays `[256, 256, 8]`
- **Window Size**: `[32, 32]` (configurable)
- **Partitioning**:
  - Projections: `[256, 256]` → `[64, 32, 32]` (64 windows)
  - Rays: `[256, 256, 8]` → `[64, 32, 32, 8]`

#### Step 6.2: Window Selection
**Script Location**: `src/dataset/tigre_mlg.py`
- **Valid Windows**: Windows where all pixels have projection > 0
- **Local Sampling**: Randomly selects `window_num` windows (default: 4)
  - Takes ALL pixels from selected windows: `4 * 32 * 32 = 4096` rays
- **Global Sampling**: From remaining windows
  - Filters valid pixels (projection > 0)
  - Randomly samples `n_rays` rays (default: 1024)

#### Step 6.3: Combine Samples
**Script Location**: `src/dataset/tigre_mlg.py`
- Concatenates: `[4096 + 1024, 8]` rays, `[4096 + 1024]` projections
- **Efficiency**: ~40% reduction in total samples vs. uniform sampling
- **Benefit**: Focuses computation on informative regions

---

## 7. Complete Execution Flow Summary

### Training Flow:
```
1. Raw CT (.mat) → dataGenerator/generateData_{category}.py
2. Generate projections → data/{category}_50.pickle
3. train.py → Load config → Initialize dataset (src/dataset/tigre.py)
4. For each epoch:
   a. Sample rays (uniform or MLG)
   b. Render projections (NeRF forward)
   c. Compute loss (MSE)
   d. Backprop + optimize
   e. Evaluate (PSNR/SSIM)
   f. Save checkpoint
5. Best model → models/{category}.tar
```

### Inference Flow:
```
1. test.py → Load model → Load validation data (src/dataset/tigre.py)
2. Render all projections → [50, 256, 256] (src/render/)
3. Reconstruct 3D volume → [256, 256, 128] (src/network/Lineformer.py)
4. Compute metrics (PSNR, SSIM) (src/utils/util.py)
5. Save results → output/{method}/{category}/
```

### Visualization Flow:
```
1. Load reconstructed volume [256, 256, 128] (from output/{method}/{category}/CT/)
2. Static: 3D_vis/3D_vis_{category}.py → marching_cubes → 3D mesh → PNG
3. Dynamic: 3D_vis/3D_vis_{category}_gif.py → Rotate mesh → 120 frames → GIF
4. Save → 3D_vis/{category}/
```

---

## 8. Key Data Structures

### Ray Format: `[N, 8]`
- `[..., 0:3]`: Ray origin (X-ray source position)
- `[..., 3:6]`: Ray direction (normalized vector)
- `[..., 6:7]`: Near bound
- `[..., 7:8]`: Far bound

### Volume Format: `[H, W, L]`
- Height (H): 256
- Width (W): 256
- Length (L): 128
- Values: Attenuation coefficients μ ∈ [0, 1]

### Projection Format: `[N_angles, H, W]`
- Number of angles: 50 (training/validation)
- Detector size: 256×256
- Values: Line integrals (attenuation sums) ∈ [0, 1]

---

## 9. Network Architecture Details

### Line-segment Transformer (Lineformer)
1. **Input**: 3D coordinates `[N, 3]`
2. **Encoder**: Hash grid → `[N, 32]`
3. **Line Partitioning**: Groups into segments of `line_size` (16)
4. **Attention**: Multi-head self-attention within segments
5. **FFN**: Feed-forward network
6. **Skip Connections**: Concatenate input at layer 4
7. **Output**: Attenuation `[N, 1]`

### Volume Rendering Equation
```
I = ∫[near, far] exp(-∫[near, t] μ(s) ds) * μ(t) dt
```
- Simplified to: `acc = sum(μ[i] * dists[i])` (Beer-Lambert law)

---

## 10. Web GUI Interface

### 10.1 Web GUI Entry Point
**Script**: `web_gui/app.py`
- **Frontend**: `web_gui/templates/index.html`, `web_gui/static/style.css`, `web_gui/static/script.js`
- **Backend API**: Flask server in `web_gui/app.py`
- **Features**:
  - Category selection and output type filtering
  - Real-time progress monitoring for GIF generation
  - Output file viewing and folder navigation
  - System information display (CPU, Memory, GPU, Storage)

### 10.2 GUI Operations
**Script Location**: `web_gui/app.py`
- **Static Reconstruction**: Calls `3D_vis/3D_vis_{category}.py` (lines 380-430)
- **Dynamic Reconstruction**: Calls `3D_vis/3D_vis_{category}_gif.py` (lines 454-504)
- **Output File Listing**: `get_output_files()` function (lines 692-750)
- **Real-time Progress**: `run_subprocess_with_realtime_output()` function (lines 520-580)

---

## 11. File Organization

```
project/
├── dataGenerator/
│   ├── generateData_{category}.py    # Data preprocessing (main pipeline)
│   ├── generateData_from_png.py      # PNG X-ray to pickle conversion
│   ├── data_vis_{category}.py        # Raw to .mat conversion
│   └── raw_data/{category}/          # Raw CT data
├── data/
│   └── {category}_50.pickle          # Processed data
├── config/
│   └── {method}/{category}_50.yaml   # Training configs
├── train.py                          # Training script
├── train_mlg.py                      # Training with MLG
├── test.py                           # Inference script
├── web_gui/
│   ├── app.py                        # Flask backend server
│   ├── templates/index.html          # Frontend HTML
│   └── static/                       # CSS and JavaScript
├── src/
│   ├── dataset/                      # Data loading
│   │   ├── tigre.py                  # TIGREDataset class
│   │   └── tigre_mlg.py              # TIGREDataset_MLG class
│   ├── encoder/                      # Position encoding
│   │   └── hashgrid_pytorch.py      # Hash grid encoder
│   ├── network/                      # Network architectures
│   │   └── Lineformer.py             # Line-segment Transformer
│   ├── render/                       # Volume rendering
│   ├── loss/                         # Loss functions
│   └── utils/                        # Metrics, utilities
│       └── util.py                   # PSNR, SSIM functions
├── models/
│   └── {category}.tar                # Trained models
├── output/
│   └── {method}/{category}/           # Inference results
└── 3D_vis/
    ├── 3D_vis_{category}.py          # Static visualization
    └── 3D_vis_{category}_gif.py      # Dynamic GIF visualization
```

---

This completes the comprehensive documentation of the CT-Former system pipeline from input to output, including all intermediate steps, data transformations, and metric calculations.

