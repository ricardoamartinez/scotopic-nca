# Training loop for ScotopicNCA
# To be implemented by Agent 1 

import torch
import torch.optim as optim
import torch.nn.functional as F
from .model import ScotopicNCA
from .data import generate_moving_square_video, create_sparse_mask
import os

def train_baseline_nca(
    epochs: int = 200,
    sequence_length: int = 32,
    frame_size: int = 64,
    learning_rate: float = 1e-3,
    updates_per_frame: int = 1,
    device: str = 'cpu',
    save_path: str = 'scotopic_nca_baseline.pth'
):
    """
    Main training loop for the baseline Scotopic NCA.
    Saves the trained model's state_dict to the specified path.
    """
    print(f"Starting training on device: {device}")
    model = ScotopicNCA(device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    video_data = generate_moving_square_video(sequence_length=sequence_length, frame_size=frame_size).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        initial_state = torch.zeros(1, model.state_channels, frame_size, frame_size, device=device)
        state_grid = initial_state
        total_loss = 0.0
        
        for t in range(sequence_length):
            target_frame = video_data[t:t+1]
            mask = create_sparse_mask(target_frame).to(device)
            sparse_input = target_frame * mask
            # Create new state to avoid in-place modification issues
            new_state = state_grid.clone()
            new_state[:, model.input_channel:model.input_channel+1, :, :] = sparse_input
            state_grid = model(new_state, steps=updates_per_frame)
            prediction = state_grid[:, model.pred_channel:model.pred_channel+1, :, :]
            loss = F.mse_loss(prediction * mask, sparse_input)
            total_loss += loss

        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.6f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    print("Training complete.")
    return model

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_baseline_nca(device=device)
    # You can add code here to save the model or visualize results 