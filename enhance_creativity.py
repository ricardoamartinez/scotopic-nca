#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np

class CreativeScotopicNCA(nn.Module):
    """
    Enhanced Scotopic NCA with more creative and diverse behavior.
    Based on advanced NCA techniques for richer pattern formation.
    """
    def __init__(self, hidden_channels: int = 24, device: str = 'cpu'):
        super().__init__()
        self.prediction_channels = 3
        self.input_channels = 3
        self.hidden_channels = hidden_channels
        self.state_channels = self.prediction_channels + self.input_channels + self.hidden_channels
        self.device = device

        # Enhanced perception with more filters for complex pattern detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Add Laplacian for blob detection
        laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Combine all perception filters
        perception_filters = torch.cat([identity, sobel_x, sobel_y, laplacian], dim=0)
        self.perception_conv = nn.Conv2d(
            self.state_channels, self.state_channels * 4, 
            kernel_size=3, padding=1, bias=False, groups=self.state_channels
        )
        self.perception_conv.weight = nn.Parameter(
            perception_filters.repeat(self.state_channels, 1, 1, 1), requires_grad=False
        )

        # Multi-scale update network for richer dynamics
        self.update_net = nn.Sequential(
            nn.Conv2d(self.state_channels * 4, 128, kernel_size=1),
            nn.ReLU(),
            nn.Dropout2d(0.1),  # Add stochasticity for diversity
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, self.state_channels, kernel_size=1)
        )
        
        # Auxiliary creativity loss network to encourage diverse patterns
        self.creativity_net = nn.Sequential(
            nn.Conv2d(self.state_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Initialize with very small weights for stability
        with torch.no_grad():
            for layer in [self.update_net, self.creativity_net]:
                for module in layer.modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        module.weight.normal_(0, 0.001)
                        if module.bias is not None:
                            module.bias.zero_()

        self.to(device)

    def forward(self, state_grid: torch.Tensor, steps: int = 1, decay: float = 0.9995) -> torch.Tensor:
        current_state = state_grid
        for _ in range(steps):
            # Enhanced perception with 4 filters per channel
            perception_vector = self.perception_conv(current_state)
            delta = self.update_net(perception_vector)
            new_state = current_state + delta

            # Apply selective decay and constraints
            channels = []
            
            # Prediction channels (0,1,2) - clamp without noise for stability
            for i in range(self.prediction_channels):
                pred = torch.clamp(new_state[:, i:i+1, :, :], 0.0, 1.0)
                channels.append(pred)
            
            # Input register (3,4,5) - preserve
            for i in range(self.prediction_channels, self.prediction_channels + self.input_channels):
                channels.append(new_state[:, i:i+1, :, :])
            
            # Hidden channels - apply MUCH LESS decay for better temporal memory
            for i in range(self.prediction_channels + self.input_channels, self.state_channels):
                hidden = new_state[:, i:i+1, :, :] * decay  # 0.9995 instead of 0.999
                # Clamp hidden states to prevent explosion
                hidden = torch.clamp(hidden, -3.0, 3.0)
                channels.append(hidden)
            
            current_state = torch.cat(channels, dim=1)
            
        return current_state

    def get_prediction_rgb(self, state_grid: torch.Tensor) -> torch.Tensor:
        return torch.clamp(state_grid[:, :self.prediction_channels, :, :], 0.0, 1.0)
    
    def compute_creativity_loss(self, state_grid: torch.Tensor) -> torch.Tensor:
        """Compute loss that encourages diverse and interesting patterns"""
        creativity_score = self.creativity_net(state_grid)
        # Encourage higher creativity scores (more diverse patterns)
        return F.mse_loss(creativity_score, torch.ones_like(creativity_score))

def create_sparse_mask(frame: torch.Tensor, keep_fraction: float = 0.01) -> torch.Tensor:
    """Simple random sparse mask exactly like Aman's approach"""
    b, c, h, w = frame.shape
    num_pixels = h * w
    num_to_keep = int(num_pixels * keep_fraction)
    
    indices = torch.randperm(num_pixels, device=frame.device)[:num_to_keep]
    mask_flat = torch.zeros(num_pixels, device=frame.device)
    mask_flat[indices] = 1.0
    
    return mask_flat.view(h, w).unsqueeze(0).unsqueeze(0).repeat(b, c, 1, 1)

def run_creative_learning(resolution: int = 128):
    """Run the creative enhanced Scotopic NCA"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Creative Scotopic NCA on device: {device}")
    
    model = CreativeScotopicNCA(device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Higher LR for faster convergence
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize with richer random state
    state_grid = torch.zeros(1, model.state_channels, resolution, resolution, device=device)
    state_grid[:, :model.prediction_channels, :, :] = torch.rand(
        1, model.prediction_channels, resolution, resolution, device=device
    ) * 0.1
    
    print(f"Starting creative learning at {resolution}x{resolution}. Press 'q' to quit.")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame_rgb, (resolution, resolution), interpolation=cv2.INTER_AREA)
        frame_tensor = torch.from_numpy(resized_frame).float().to(device) / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)

        optimizer.zero_grad()
        
        # Enhanced sparse input
        mask = create_sparse_mask(frame_tensor, keep_fraction=0.01)
        sparse_input = frame_tensor * mask

        # Inject into input register
        new_state_grid = state_grid.clone()
        new_state_grid[:, model.prediction_channels:model.prediction_channels+model.input_channels, :, :] = sparse_input

        # Multiple update steps for richer dynamics
        state_grid = model(new_state_grid, steps=3, decay=0.9995)

        # Get prediction
        prediction_rgb = model.get_prediction_rgb(state_grid)
        
        # FOCUSED loss function for better convergence
        sparse_loss = F.mse_loss(prediction_rgb * mask, sparse_input)
        
        # Encourage filling gaps (prevent pure black regions)
        gap_filling_loss = F.mse_loss(prediction_rgb.mean(), torch.tensor(0.3, device=device))
        
        # Add small reconstruction loss on full frame (weighted low)
        full_reconstruction_loss = F.mse_loss(prediction_rgb, frame_tensor)
        
        # Balanced loss focusing on reconstruction
        total_loss = sparse_loss + 0.02 * gap_filling_loss + 0.1 * full_reconstruction_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        
        # State management
        state_grid = state_grid.detach()
        state_grid = torch.clamp(state_grid, -10.0, 10.0)
        
        frame_count += 1
        
        # Progress reporting
        if frame_count % 30 == 0:
            pred_stats = prediction_rgb.flatten()
            pred_mean, pred_std = pred_stats.mean().item(), pred_stats.std().item()
            pred_min, pred_max = pred_stats.min().item(), pred_stats.max().item()
            print(f"Frame {frame_count}: Loss={total_loss.item():.6f}, "
                  f"Sparse={sparse_loss.item():.6f}, Gap={gap_filling_loss.item():.6f}, "
                  f"Pred=[{pred_min:.2f},{pred_mean:.2f},{pred_max:.2f}]")

        # Visualization
        with torch.no_grad():
            pred_mean_vis = prediction_rgb.mean().item()
            
            original_vis = (frame_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            sparse_vis = (sparse_input.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pred_vis = (prediction_rgb.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Ensure contiguous arrays
            original_vis = np.ascontiguousarray(original_vis)
            sparse_vis = np.ascontiguousarray(sparse_vis)
            pred_vis = np.ascontiguousarray(pred_vis)

            # Labels
            cv2.putText(original_vis, 'Original', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(sparse_vis, 'Input (1%)', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(pred_vis, f'Fixed NCA (μ={pred_mean_vis:.2f})', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            combined = np.hstack([original_vis, sparse_vis, pred_vis])
            display = cv2.resize(combined, (1280, 427))
            
            cv2.imshow('Fixed Scotopic NCA - Better Temporal Memory', cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Creative learning session ended.")

if __name__ == '__main__':
    run_creative_learning() 