import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import imageio
from tqdm import tqdm
import argparse
import pickle

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default="1", help="gpu to use")
    parser.add_argument("--category", default="chest", help="category to visualize")
    parser.add_argument("--data_type", default="gt", choices=["gt", "pred"], help="ground truth or predicted data")
    parser.add_argument("--output_format", default="gif", choices=["gif", "mp4"], help="output video format")
    parser.add_argument("--fps", default=30, type=int, help="frames per second")
    parser.add_argument("--rotation_angles", default=360, type=int, help="total rotation degrees")
    parser.add_argument("--elevation", default=20, type=int, help="camera elevation angle")
    parser.add_argument("--sigma", default=0.55, type=float, help="marching cubes threshold")
    parser.add_argument("--alpha", default=0.3, type=float, help="mesh transparency")
    return parser

def load_data(category, data_type="gt"):
    """Load CT data from pickle file or test output"""
    if data_type == "gt":
        # Load from original data
        path = f'../data/{category}_50.pickle'
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        CT_image = data["image"]
    else:
        # Load from test output (predicted data)
        # You'll need to modify this path based on your test output structure
        output_path = f'../output/Lineformer/{category}/CT/'
        # This is a placeholder - you'll need to load the actual predicted volume
        path = f'../data/{category}_50.pickle'
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        CT_image = data["image"]  # Replace with predicted volume loading
    
    # Apply transformations for proper orientation
    CT_image = CT_image[::-1, ...]
    CT_image = np.rot90(CT_image, -1, (0, 1))
    
    return CT_image

def create_marching_cubes_mesh(CT_image, sigma=0.55):
    """Create mesh using marching cubes algorithm"""
    min_value = CT_image.min()
    max_value = CT_image.max()
    threshold = sigma * min_value + (1 - sigma) * max_value
    verts, faces, _, _ = measure.marching_cubes(CT_image, threshold)
    return verts, faces

def create_volume_rendering_frames(CT_image, output_dir, num_frames=120, elevation=20, sigma=0.55, alpha=0.3):
    """Create frames for volume rendering video"""
    print(f"Creating {num_frames} frames for volume rendering...")
    
    # Create marching cubes mesh
    verts, faces = create_marching_cubes_mesh(CT_image, sigma)
    
    # Calculate rotation angles
    angle_interval = 360 / num_frames
    
    img_files = []
    
    for i in tqdm(range(num_frames)):
        angle = angle_interval * i
        
        # Create 3D plot
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create mesh
        mesh = Poly3DCollection(verts[faces], alpha=alpha)
        face_color = [0.7, 0.7, 0.9]  # Light blue color
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        
        # Set view limits
        ax.set_xlim(0, CT_image.shape[0])
        ax.set_ylim(0, CT_image.shape[1])
        ax.set_zlim(0, CT_image.shape[2])
        
        # Set camera view
        ax.view_init(elev=elevation, azim=angle)
        ax.axis("off")
        
        # Save frame
        frame_path = os.path.join(output_dir, f'frame_{i:03d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight', pad_inches=0)
        img_files.append(frame_path)
        plt.close()
    
    return img_files

def create_video_from_frames(img_files, output_path, fps=30, output_format="gif"):
    """Create video from rendered frames"""
    print(f"Creating {output_format.upper()} video...")
    
    if output_format == "gif":
        # Create GIF
        frames = []
        for img_file in tqdm(img_files):
            img = imageio.imread(img_file)
            frames.append(img)
        
        imageio.mimsave(output_path, frames, duration=1000/fps, loop=0)
        
    elif output_format == "mp4":
        # Create MP4 (requires additional dependencies)
        try:
            import cv2
            # Implementation for MP4 creation
            print("MP4 creation requires cv2. Creating GIF instead...")
            create_video_from_frames(img_files, output_path.replace('.mp4', '.gif'), fps, "gif")
        except ImportError:
            print("cv2 not available. Creating GIF instead...")
            create_video_from_frames(img_files, output_path.replace('.mp4', '.gif'), fps, "gif")

def main():
    parser = config_parser()
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # Create output directory
    output_dir = f"{args.category}_{args.data_type}_volume_render"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading {args.data_type} data for {args.category}...")
    CT_image = load_data(args.category, args.data_type)
    print(f"CT image shape: {CT_image.shape}")
    
    # Create frames
    num_frames = int(args.rotation_angles * args.fps / 30)  # Adjust frames based on rotation
    img_files = create_volume_rendering_frames(
        CT_image, 
        output_dir, 
        num_frames=num_frames,
        elevation=args.elevation,
        sigma=args.sigma,
        alpha=args.alpha
    )
    
    # Create video
    output_filename = f"{args.category}_{args.data_type}_volume_render.{args.output_format}"
    output_path = os.path.join(output_dir, output_filename)
    
    create_video_from_frames(img_files, output_path, args.fps, args.output_format)
    
    print(f"Volume rendering video saved to: {output_path}")
    print(f"Frames saved in: {output_dir}")

if __name__ == "__main__":
    main()
