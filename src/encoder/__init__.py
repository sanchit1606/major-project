# Import encoders only when needed to avoid C++ compilation issues
# from .hashencoder import HashEncoder
import torch
from .freqencoder import FreqEncoder
from .tensorf_encoder import TensorfEncoder

# 实际调用的时候是采用 get_encoder 返回对应的 encoder

def get_encoder(encoding, input_dim=3, 
                multires=6, 
                degree=4,
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19,
                **kwargs):

    if encoding == "None":
        return lambda x, **kwargs: x, input_dim
    
    elif encoding == "frequency":
        # For frequency encoding, calculate N_freqs to get exactly 32 dimensions
        # We need: 3 + (3 * N_freqs * 2) = 32
        # So: 3 + 6 * N_freqs = 32
        # Therefore: N_freqs = (32 - 3) / 6 = 29 / 6 ≈ 4.83
        # We'll use N_freqs=4 and add padding to get exactly 32 dimensions
        encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=3, N_freqs=4, log_sampling=True)
        # The encoder will output 3 + (3 * 4 * 2) = 3 + 24 = 27 dimensions
        # We need to pad it to 32 dimensions
        original_forward = encoder.forward
        def padded_forward(input, bound):
            output = original_forward(input, bound)
            print(f"Frequency encoder output shape: {output.shape}, expected: 32")
            # Pad to exactly 32 dimensions
            if output.shape[-1] < 32:
                padding = torch.zeros(*output.shape[:-1], 32 - output.shape[-1], device=output.device, dtype=output.dtype)
                output = torch.cat([output, padding], dim=-1)
                print(f"After padding, output shape: {output.shape}")
            return output
        encoder.forward = padded_forward
        encoder.output_dim = 32

    elif encoding == "hashgrid":
        # Try C++ HashGrid encoder first, fallback to PyTorch implementation if compilation fails
        try:
            from .hashencoder import HashEncoder
            encoder = HashEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, 
                base_resolution=base_resolution, 
                log2_hashmap_size=log2_hashmap_size)
            print("Using C++ HashGrid encoder")
        except Exception as e:
            print(f"Warning: C++ HashGrid encoder not available due to compilation issues: {e}")
            print("Falling back to PyTorch HashGrid implementation...")
            try:
                from .hashgrid_pytorch import HashEncoder
                encoder = HashEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, 
                    base_resolution=base_resolution, 
                    log2_hashmap_size=log2_hashmap_size)
                print("Using PyTorch HashGrid encoder")
            except Exception as e2:
                print(f"Warning: PyTorch HashGrid encoder also failed: {e2}")
                print("Falling back to frequency encoder...")
                # Calculate N_freqs to get exactly 32 dimensions (matching HashGrid's 16 * 2 = 32)
                # We need: 3 + (3 * N_freqs * 2) = 32
                # So: 3 + 6 * N_freqs = 32
                # Therefore: N_freqs = (32 - 3) / 6 = 29 / 6 ≈ 4.83
                # We'll use N_freqs=4 and add padding to get exactly 32 dimensions
                encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=3, N_freqs=4, log_sampling=True)
                # The encoder will output 3 + (3 * 4 * 2) = 3 + 24 = 27 dimensions
                # We need to pad it to 32 dimensions
                original_forward = encoder.forward
                def padded_forward(input, bound):
                    output = original_forward(input, bound)
                    # Pad to exactly 32 dimensions
                    if output.shape[-1] < 32:
                        padding = torch.zeros(*output.shape[:-1], 32 - output.shape[-1], device=output.device, dtype=output.dtype)
                        output = torch.cat([output, padding], dim=-1)
                    return output
                encoder.forward = padded_forward
                encoder.output_dim = 32
    
    elif encoding == "tensorf":
        encoder = TensorfEncoder(input_dim=input_dim, num_levels=num_levels, **kwargs)

    else:
        raise NotImplementedError()

    return encoder, encoder.output_dim