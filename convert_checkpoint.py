import os
import torch
import argparse
from pathlib import Path

def convert_lightning_checkpoint(lightning_dir, output_path):
    """
    Convert a PyTorch Lightning checkpoint directory to a regular PyTorch checkpoint file.
    
    Args:
        lightning_dir: Path to the Lightning checkpoint directory
        output_path: Path where to save the converted checkpoint
    """
    lightning_dir = Path(lightning_dir)
    
    if not lightning_dir.exists():
        raise FileNotFoundError(f"Lightning checkpoint directory not found: {lightning_dir}")
    
    # Check if it's a Lightning checkpoint - look for version file in archive subdirectory
    version_file = lightning_dir / "archive" / "version"
    if not version_file.exists():
        raise ValueError(f"Not a valid Lightning checkpoint directory: {lightning_dir}")
    
    # Try to load from the data.pkl file first
    data_pkl = lightning_dir / "archive" / "data.pkl"
    if data_pkl.exists():
        try:
            checkpoint = torch.load(data_pkl, map_location='cpu')
            print(f"Loaded checkpoint from {data_pkl}")
        except Exception as e:
            print(f"Failed to load from data.pkl: {e}")
            # Try to load the entire directory
            try:
                checkpoint = torch.load(lightning_dir, map_location='cpu')
                print(f"Loaded checkpoint from {lightning_dir}")
            except Exception as e2:
                raise ValueError(f"Could not load checkpoint from either {data_pkl} or {lightning_dir}: {e2}")
    else:
        # Try to load the entire directory
        try:
            checkpoint = torch.load(lightning_dir, map_location='cpu')
            print(f"Loaded checkpoint from {lightning_dir}")
        except Exception as e:
            raise ValueError(f"Could not load checkpoint from {lightning_dir}: {e}")
    
    # Save as a regular PyTorch checkpoint
    torch.save(checkpoint, output_path)
    print(f"Converted checkpoint saved to: {output_path}")
    
    # Print checkpoint contents
    print("\nCheckpoint contents:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  {key}: {type(checkpoint[key])} with keys: {list(checkpoint[key].keys())}")
        else:
            print(f"  {key}: {type(checkpoint[key])}")

def main():
    parser = argparse.ArgumentParser(description="Convert Lightning checkpoint to PyTorch checkpoint")
    parser.add_argument("--input", required=True, help="Path to Lightning checkpoint directory")
    parser.add_argument("--output", required=True, help="Path for output PyTorch checkpoint file")
    
    args = parser.parse_args()
    
    try:
        convert_lightning_checkpoint(args.input, args.output)
        print("Conversion completed successfully!")
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
