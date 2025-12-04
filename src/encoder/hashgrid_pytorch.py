import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HashGridPyTorch(nn.Module):
    """
    Pure PyTorch implementation of HashGrid encoder that doesn't require C++ compilation.
    This is a fallback implementation that should work on Windows without build tools.
    """
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        
        # Initialize hash tables for each level
        self.hash_tables = nn.ModuleList()
        self.offsets = []
        offset = 0
        
        for i in range(num_levels):
            resolution = base_resolution * (2 ** i)
            params_in_level = min(2 ** log2_hashmap_size, (resolution + 1) ** input_dim)
            
            # Create hash table for this level
            hash_table = nn.Embedding(params_in_level, level_dim)
            # Initialize with small random values
            nn.init.uniform_(hash_table.weight, -0.0001, 0.0001)
            
            self.hash_tables.append(hash_table)
            self.offsets.append(offset)
            offset += params_in_level
    
    def spatial_hash(self, coords, level, resolution):
        """Simple spatial hashing function"""
        # Scale coordinates to resolution
        scaled_coords = coords * resolution
        
        # Convert to integer coordinates
        int_coords = torch.floor(scaled_coords).long()
        
        # Clamp to valid range
        int_coords = torch.clamp(int_coords, 0, resolution - 1)
        
        # Simple hash function
        hash_val = (int_coords[..., 0] * 73856093 + 
                   int_coords[..., 1] * 19349663 + 
                   int_coords[..., 2] * 83492791) % (2 ** self.log2_hashmap_size)
        
        # Ensure hash values are within valid range for the embedding table
        hash_val = torch.clamp(hash_val, 0, self.hash_tables[level].num_embeddings - 1)
        
        return hash_val
    
    def forward(self, inputs, size=1):
        """
        inputs: [B, D] coordinates in [-size, size]
        """
        # Normalize inputs to [0, 1]
        inputs = (inputs + size) / (2 * size)
        
        B = inputs.shape[0]
        outputs = []
        
        for level in range(self.num_levels):
            resolution = self.base_resolution * (2 ** level)
            
            # Get hash indices for this level
            hash_indices = self.spatial_hash(inputs, level, resolution)
            
            # Get embeddings from hash table
            level_embeddings = self.hash_tables[level](hash_indices)  # [B, level_dim]
            outputs.append(level_embeddings)
        
        # Concatenate all levels
        outputs = torch.cat(outputs, dim=-1)  # [B, num_levels * level_dim]
        
        return outputs


# Create a compatibility wrapper
class HashEncoder(HashGridPyTorch):
    """Compatibility wrapper for the original HashEncoder interface"""
    pass
