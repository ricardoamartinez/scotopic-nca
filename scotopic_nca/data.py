# Data utilities for video generation and processing
# To be implemented by Agent 1 

import torch
import numpy as np

def generate_moving_square_video(
    sequence_length: int = 32,
    frame_size: int = 64,
    square_size: int = 10
) -> torch.Tensor:
    """
    Generates a synthetic grayscale video of a white square moving diagonally
    on a black background.

    Args:
        sequence_length: The number of frames in the video.
        frame_size: The height and width of each frame.
        square_size: The size of the moving square.

    Returns:
        A tensor of shape (sequence_length, 1, frame_size, frame_size)
        representing the video, with pixel values in [0, 1].
    """
    video = torch.zeros((sequence_length, 1, frame_size, frame_size), dtype=torch.float32)
    velocity = (1, 1)  # Move down and right

    # Initial position
    pos = [frame_size // 4, frame_size // 4]

    for t in range(sequence_length):
        x_start, y_start = int(pos[0]), int(pos[1])
        x_end, y_end = x_start + square_size, y_start + square_size

        # Draw the square, handling boundaries
        video[t, 0, y_start:y_end, x_start:x_end] = 1.0

        # Update position
        pos[0] = (pos[0] + velocity[0]) % (frame_size - square_size)
        pos[1] = (pos[1] + velocity[1]) % (frame_size - square_size)

    return video

def create_sparse_mask(
    frame: torch.Tensor,
    keep_fraction: float = 0.01
) -> torch.Tensor:
    """
    Creates a binary mask to randomly drop pixels from a frame.

    Args:
        frame: A single frame tensor of shape (C, H, W).
        keep_fraction: The fraction of pixels to keep (e.g., 0.01 for 1%).

    Returns:
        A binary mask tensor of the same shape as the frame.
    """
    num_pixels = frame.numel()
    num_to_keep = int(num_pixels * keep_fraction)

    mask = torch.zeros_like(frame.flatten())
    indices = torch.randperm(num_pixels)[:num_to_keep]
    mask[indices] = 1.0

    return mask.view(frame.shape) 