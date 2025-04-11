import torch

def preprocess_state(state, device=None):
    """Preprocess the state for the CNN model"""
    # Input shape: (4, 96, 96, 3) - 4 stacked RGB frames

    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32).to(device)

    n_frames, height, width, channels = state.shape
    
    grayscale_frames = torch.zeros((n_frames, height, width), dtype=torch.float32).to(device)
    rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).to(device)
    for i in range(n_frames):
        # Dot product along the RGB channels
        grayscale_frames[i] = torch.sum(state[i, :, :, :3] * rgb_weights, dim=2).to(device)
    
    # Transpose from (4, 96, 96) to (96, 96, 4)
    transposed = grayscale_frames.permute(1, 2, 0)
    normalized = transposed / 255.0
    return normalized