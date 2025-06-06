#!/usr/bin/env python3

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scotopic_nca.model import RobustScotopicNCA
from live_online_learning import create_sparse_mask

def debug_nca_collapse():
    """Debug version to understand NCA collapse"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Debugging NCA collapse on device: {device}")
    
    # Create simple test pattern
    resolution = 64  # Smaller for easier debugging
    model = RobustScotopicNCA(device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Create a simple test image (white square on black background)
    test_frame = torch.zeros(1, 3, resolution, resolution, device=device)
    test_frame[0, :, 20:40, 20:40] = 1.0  # White square
    
    # Initialize state grid with small random values
    state_grid = torch.zeros(1, model.state_channels, resolution, resolution, device=device)
    state_grid[:, :model.prediction_channels, :, :] = torch.rand(1, model.prediction_channels, resolution, resolution, device=device) * 0.1
    
    print(f"Initial state stats:")
    print(f"  Prediction channels mean: {state_grid[:, :3, :, :].mean().item():.6f}")
    print(f"  Prediction channels std: {state_grid[:, :3, :, :].std().item():.6f}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    for step in range(20):
        optimizer.zero_grad()
        
        # Create sparse input (1% of pixels)
        mask = create_sparse_mask(test_frame, keep_fraction=0.01)
        sparse_input = test_frame * mask
        
        # Store original state for comparison
        original_state = state_grid.clone()
        
        # Inject sparse input into input register
        new_state_grid = state_grid.clone()
        new_state_grid[:, model.prediction_channels:model.prediction_channels+model.input_channels, :, :] = sparse_input
        
        # Run NCA update
        updated_state = model(new_state_grid, steps=1)
        
        # Get prediction
        prediction_rgb = model.get_prediction_rgb(updated_state)
        
        # Calculate losses
        full_frame_loss = F.mse_loss(prediction_rgb, test_frame)
        sparse_pixel_loss = F.mse_loss(prediction_rgb * mask, sparse_input)
        loss = full_frame_loss + 2.0 * sparse_pixel_loss
        
        # Backward pass
        loss.backward()
        
        # Check gradients before clipping
        total_grad_norm = 0
        param_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                param_count += 1
                if step < 3:  # Only print first few steps
                    print(f"  {name}: grad_norm={grad_norm:.6f}")
        
        avg_grad_norm = total_grad_norm / param_count if param_count > 0 else 0
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update state for next iteration
        state_grid = updated_state.detach()
        
        # Debug statistics
        pred_mean = prediction_rgb.mean().item()
        pred_std = prediction_rgb.std().item()
        pred_min = prediction_rgb.min().item()
        pred_max = prediction_rgb.max().item()
        
        state_change = (updated_state - original_state).abs().mean().item()
        
        print(f"Step {step+1:2d}: Loss={loss.item():.6f}, "
              f"Pred=[{pred_min:.3f}, {pred_mean:.3f}, {pred_max:.3f}], "
              f"Std={pred_std:.3f}, StateΔ={state_change:.6f}, "
              f"GradNorm={avg_grad_norm:.6f}")
        
        # Check for collapse
        if pred_max < 0.01:
            print(f"*** COLLAPSE DETECTED at step {step+1} ***")
            print(f"Prediction range: [{pred_min:.6f}, {pred_max:.6f}]")
            print(f"State change: {state_change:.6f}")
            
            # Analyze what went wrong
            print("\nDiagnostics:")
            print(f"  Hidden state mean: {updated_state[:, 6:, :, :].mean().item():.6f}")
            print(f"  Hidden state std: {updated_state[:, 6:, :, :].std().item():.6f}")
            print(f"  Input register mean: {updated_state[:, 3:6, :, :].mean().item():.6f}")
            
            # Check if gradients are flowing
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"  {name} grad: mean={param.grad.mean().item():.6f}, std={param.grad.std().item():.6f}")
            
            break
    
    return model, state_grid

if __name__ == '__main__':
    debug_nca_collapse() 