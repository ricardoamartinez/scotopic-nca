# ScotopicNCA model implementation
# To be implemented by Agent 1 

import torch
import torch.nn as nn

class RobustScotopicNCA(nn.Module):
    """
    Scotopic Vision NCA exactly as described by Aman Bhargava.
    One layer for prediction, one for input register, rest are learned.
    """
    def __init__(
        self,
        hidden_channels: int = 12,
        device: str = 'cpu'
    ):
        """
        Initializes the Scotopic NCA model following Aman's approach.

        Args:
            hidden_channels: Number of hidden channels for cell memory.
            device: The device to run the model on.
        """
        super().__init__()
        # State layout: [R_pred, G_pred, B_pred, R_input, G_input, B_input, H_1, ..., H_N]
        self.prediction_channels = 3  # RGB prediction channels (0, 1, 2)
        self.input_channels = 3       # RGB input register channels (3, 4, 5)  
        self.hidden_channels = hidden_channels
        self.state_channels = self.prediction_channels + self.input_channels + self.hidden_channels
        self.device = device

        self.update_net = nn.Sequential(
            nn.Conv2d(self.state_channels * 3, 128, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, self.state_channels, kernel_size=1, bias=True)
        )
        
        # Sobel filters for perception of gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Perception filters: gradients and self-identity for each channel
        perception_filters = torch.cat([identity, sobel_x, sobel_y], dim=0)
        self.perception_conv = nn.Conv2d(self.state_channels, self.state_channels * 3, kernel_size=3, padding=1, bias=False, groups=self.state_channels)
        self.perception_conv.weight = nn.Parameter(perception_filters.repeat(self.state_channels, 1, 1, 1), requires_grad=False)

        # Initialize with VERY small weights for numerical stability
        with torch.no_grad():
            # Extremely small weights to prevent explosion
            self.update_net[0].weight.normal_(0, 0.001)
            self.update_net[2].weight.normal_(0, 0.001)
            self.update_net[0].bias.zero_()
            self.update_net[2].bias.zero_()

        self.to(device)

    def forward(self, state_grid: torch.Tensor, steps: int = 1, decay: float = 0.99) -> torch.Tensor:
        """
        Performs the Scotopic NCA update steps following Aman's approach.

        Args:
            state_grid: The current state of the NCA grid, shape (B, C, H, W).
            steps: The number of update iterations to perform.
            decay: The decay factor for hidden states.

        Returns:
            The updated state grid.
        """
        current_state = state_grid
        for _ in range(steps):
            perception_vector = self.perception_conv(current_state)
            delta = self.update_net(perception_vector)
            new_state = current_state + delta

            # Apply state decay to hidden channels only, following Aman's approach
            channels = []
            
            # Prediction channels (0, 1, 2) - clamp to [0, 1] 
            for i in range(self.prediction_channels):
                pred_clamped = torch.clamp(new_state[:, i:i+1, :, :], 0.0, 1.0)
                channels.append(pred_clamped)
            
            # Input register channels (3, 4, 5) - keep as is (will be overwritten anyway)
            for i in range(self.prediction_channels, self.prediction_channels + self.input_channels):
                channels.append(new_state[:, i:i+1, :, :])
            
            # Hidden channels (6+) - apply decay for stability
            for i in range(self.prediction_channels + self.input_channels, self.state_channels):
                decayed_channel = new_state[:, i:i+1, :, :] * decay
                channels.append(decayed_channel)
            
            current_state = torch.cat(channels, dim=1)
            
        return current_state

    def get_prediction_rgb(self, state_grid: torch.Tensor) -> torch.Tensor:
        """
        Extracts the RGB prediction from the state grid (Aman's approach).
        Returns the first 3 channels which are the model's pixel predictions.
        """
        return torch.clamp(state_grid[:, :self.prediction_channels, :, :], 0.0, 1.0) 