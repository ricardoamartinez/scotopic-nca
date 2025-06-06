# ScotopicNCA model implementation
# To be implemented by Agent 1 

import torch
import torch.nn as nn

class ScotopicNCA(nn.Module):
    """
    A Neural Cellular Automaton for scotopic vision reconstruction.
    This model learns to reconstruct a full image from a sparse set of pixels.
    """
    def __init__(
        self,
        state_channels: int = 16,
        hidden_channels: int = 128,
        device: str = 'cpu'
    ):
        """
        Initializes the NCA model.

        Args:
            state_channels: The number of channels in each cell's state vector.
            hidden_channels: The number of hidden units in the update network.
            device: The device to run the model on ('cpu' or 'cuda').
        """
        super().__init__()
        self.state_channels = state_channels
        self.device = device

        # Channel 0: Prediction (pixel value)
        # Channel 1: Input Register (ground truth pixel)
        self.pred_channel = 0
        self.input_channel = 1

        self.update_net = nn.Sequential(
            nn.Conv2d(state_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, state_channels, kernel_size=1, bias=True)
        )
        
        with torch.no_grad():
            self.update_net[2].weight.zero_()
            self.update_net[2].bias.zero_()

        self.to(device)

    def forward(self, state_grid: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Performs the NCA update steps.

        Args:
            state_grid: The current state of the NCA grid, shape (B, C, H, W).
            steps: The number of update iterations to perform.

        Returns:
            The updated state grid.
        """
        current_state = state_grid
        for _ in range(steps):
            delta = self.update_net(current_state)
            current_state = current_state + delta
            # Clamp prediction channel without in-place modification
            pred_clamped = torch.clamp(
                current_state[:, self.pred_channel:self.pred_channel+1, :, :], 0.0, 1.0
            )
            # Create new state with clamped prediction channel
            new_state = current_state.clone()
            new_state[:, self.pred_channel:self.pred_channel+1, :, :] = pred_clamped
            current_state = new_state
        return current_state 