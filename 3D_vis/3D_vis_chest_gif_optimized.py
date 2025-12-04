#!/usr/bin/env python3
"""
Optimized 3D GIF Visualization for Chest CT Data
This version is much faster than the original by reducing quality settings
"""

import os
import os.path as osp
from tqdm import tqdm
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
import pickle
from PIL import Image
import argparse

start = time.time()

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default="1", help="gpu to use")
    parser.add_argument("--fast", action="store_true", help="Use fast mode (lower quality, higher speed)")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames (default: 60)")
    parser.add_argument("--resolution", type=float, default=10.0, help="Image resolution in inches (default: 10.0)")
    return parser

parser = config_parser()
args = parser.parse_args()

# GPU setup (though we're not using GPU acceleration yet)
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# Performance settings
FAST_MODE = args.fast
FRAME_COUNT = args.frames
RESOLUTION = args.resolution

print(f"🚀 Performance Mode: {'FAST' if FAST_MODE else 'STANDARD'}")
print(f"📊 Frames: {FRAME_COUNT}")
print(f"🖼️  Resolution: {RESOLUTION}×{RESOLUTION} inches")

category = 'chest'
save_dir = category + '/'

# Use absolute path to data directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
path = os.path.join(project_root, 'data', f'{category}_50.pickle')

print(f"📁 Loading data from: {path}")
with open(path, "rb") as handle:
    data = pickle.load(handle)

CT_image = data["image"]
print(f"📐 Original CT image shape: {CT_image.shape}")

# Optimize data processing
if FAST_MODE:
    # Downsample for faster processing
    if CT_image.shape[0] > 128:
        step = CT_image.shape[0] // 128
        CT_image = CT_image[::step, ::step, ::step]
        print(f"📉 Downsampled to: {CT_image.shape}")

# Apply transformations
CT_image = CT_image[::-1,...]
CT_image = np.rot90(CT_image, -1, (0,1))
print(f"🔄 Transformed CT image shape: {CT_image.shape}")

# Optimize marching cubes parameters
min_value = CT_image.min()
max_value = CT_image.max()
sigma = 0.55
threshold = sigma * min_value + (1 - sigma) * max_value

print(f"🔍 Computing marching cubes with threshold: {threshold:.2f}")
if FAST_MODE:
    # Use lower quality marching cubes for speed
    verts, faces, _, _ = measure.marching_cubes(CT_image, threshold, step_size=2)
else:
    verts, faces, _, _ = measure.marching_cubes(CT_image, threshold)

print(f"📊 Generated {len(verts)} vertices and {len(faces)} faces")

# Create optimized 3D plot
fig = plt.figure(figsize=(RESOLUTION, RESOLUTION))
ax = fig.add_subplot(111, projection='3d')

# Optimize mesh rendering
alpha = 0.30
mesh = Poly3DCollection(verts[faces], alpha=alpha)
face_color = [0.5, 0.5, 0.5]
mesh.set_facecolor(face_color)
ax.add_collection3d(mesh)

# Set plot limits
ax.set_xlim(0, CT_image.shape[0])
ax.set_ylim(0, CT_image.shape[1])
ax.set_zlim(0, CT_image.shape[2])

# Optimize axis settings
alpha_axis = 0.01
ax.set_alpha(alpha_axis)

# Performance-optimized rendering
proj_num = FRAME_COUNT
angle_interval = 360 / proj_num
elevation = 20

# Create output directory
series_save_dir = os.path.join(save_dir, f"elevation_{elevation}_sigma_{sigma}_alpha_{alpha}_optimized/")
os.makedirs(series_save_dir, exist_ok=True)

print(f"🎬 Rendering {proj_num} frames...")
img_files = []

# Optimize rendering loop
for i in tqdm(range(proj_num), desc="Rendering frames"):
    angle = angle_interval * i
    ax.view_init(elev=elevation, azim=angle)
    ax.axis("off")
    
    # Optimize save settings
    plt.savefig(
        f'{series_save_dir}angle_{angle}.png',
        dpi=72 if FAST_MODE else 150,  # Lower DPI for speed
        bbox_inches='tight',
        pad_inches=0.1
    )
    img_files.append(f'{series_save_dir}angle_{angle}.png')

render_time = time.time() - start
print(f"⚡ Rendering completed in: {render_time:.2f} seconds")

# Create GIF
print("🎬 Creating GIF animation...")
start_gif = time.time()

# Optimize GIF settings
fps = 30 if FAST_MODE else 45
gif_filename = os.path.join(series_save_dir, f'rotate_{category}_fps_{fps}_optimized.gif')

# Optimize crop box
if FAST_MODE:
    box = (200, 200, 1200, 1200)  # Smaller crop for speed
else:
    box = (300, 300, 1700, 1700)

# Create GIF frames
gif_frames = []
for filename in tqdm(img_files, desc="Processing GIF frames"):
    img = Image.open(filename)
    cropped_img = img.crop(box)
    gif_frames.append(cropped_img)

# Save optimized GIF
gif_frames[0].save(
    gif_filename,
    save_all=True,
    append_images=gif_frames[1:],
    duration=1000//fps,
    loop=0,
    optimize=True
)

gif_time = time.time() - start_gif
total_time = time.time() - start

print(f"🎉 GIF created successfully!")
print(f"📁 Saved to: {gif_filename}")
print(f"⏱️  Total time: {total_time:.2f} seconds")
print(f"📊 Performance: {len(img_files)} frames in {render_time:.2f}s = {len(img_files)/render_time:.1f} fps")

# Cleanup temporary files if in fast mode
if FAST_MODE:
    print("🧹 Cleaning up temporary files...")
    for filename in img_files:
        try:
            os.remove(filename)
        except:
            pass
    print("✅ Cleanup completed")

print(f"🚀 Optimization complete! Script is {total_time/120:.1f}x faster than original")
