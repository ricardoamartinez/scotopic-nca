import torch
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np
from scotopic_nca.model import RobustScotopicNCA

def create_sparse_mask(frame: torch.Tensor, keep_fraction: float = 0.01) -> torch.Tensor:
    """Creates a binary mask to randomly drop pixels from an RGB frame."""
    b, c, h, w = frame.shape
    num_pixels = h * w
    num_to_keep = int(num_pixels * keep_fraction)
    
    indices = torch.randperm(num_pixels, device=frame.device)[:num_to_keep]
    mask_flat = torch.zeros(num_pixels, device=frame.device)
    mask_flat[indices] = 1.0
    
    return mask_flat.view(h, w).unsqueeze(0).unsqueeze(0).repeat(b, c, 1, 1)

def run_live_online_learning(
    resolution: int = None,  # Use manageable camera resolution 
    learning_rate: float = 1e-5,  # Much smaller for numerical stability
    updates_per_frame: int = 4  # Fewer updates to prevent explosion
):
    """
    Runs live, online learning with the RobustScotopicNCA on a webcam feed.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running live online learning on device: {device}")

    # 1. Initialize Model with RANDOM weights and Optimizer
    model = RobustScotopicNCA(device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 2. Initialize webcam and get native resolution
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get manageable resolution (native is too large for stability)
    if resolution is None:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        native_res = min(width, height)
        resolution = min(native_res, 128)  # Cap at 128 for stability
        print(f"Native resolution: {native_res}x{native_res}, using: {resolution}x{resolution}")
    
    # 3. Initialize NCA state grid with small random values to prevent collapse
    state_grid = torch.zeros(1, model.state_channels, resolution, resolution, device=device)
    # Initialize prediction channels with small random values (prevent immediate collapse)
    state_grid[:, :model.prediction_channels, :, :] = torch.rand(1, model.prediction_channels, resolution, resolution, device=device) * 0.1
    
    print("Starting live feed. Press 'q' to quit. Reconstruction will improve over time.")
    
    frame_count = 0
    
    while True:
        # --- CAPTURE AND PRE-PROCESS FRAME ---
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame_rgb, (resolution, resolution), interpolation=cv2.INTER_AREA)
        frame_tensor = torch.from_numpy(resized_frame).float().to(device) / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0) # Shape: (1, 3, H, W)

        # --- ONLINE LEARNING STEP ---
        optimizer.zero_grad()
        
        # Create sparse input for this frame
        mask = create_sparse_mask(frame_tensor).to(device)
        sparse_input = frame_tensor * mask

        # Inject sparse input into INPUT REGISTER channels (3,4,5) - Aman's approach
        new_state_grid = state_grid.clone()
        new_state_grid[:, model.prediction_channels:model.prediction_channels+model.input_channels, :, :] = sparse_input

        # Run NCA update steps with much less aggressive decay
        state_grid = model(new_state_grid, steps=updates_per_frame, decay=0.999)

        # Get the RGB predictions from the model (channels 0,1,2)
        prediction_rgb = model.get_prediction_rgb(state_grid)
        
        # DEBUGGING: Check if prediction is collapsing
        pred_mean = prediction_rgb.mean().item()
        pred_max = prediction_rgb.max().item()
        pred_min = prediction_rgb.min().item()
        
        # CRITICAL: Use much gentler loss to prevent collapse
        # Only compare on sparse pixels to start with
        sparse_pixel_loss = F.mse_loss(prediction_rgb * mask, sparse_input)
        
        # Add a small "alive" loss to prevent total collapse
        alive_loss = F.mse_loss(prediction_rgb.mean(), torch.tensor(0.1, device=device))
        
        # Much gentler combined loss
        loss = sparse_pixel_loss + 0.01 * alive_loss
        
        # Debug collapse with emergency break
        if frame_count > 30 and pred_max < 0.001:
            print(f"*** SEVERE COLLAPSE DETECTED at frame {frame_count} ***")
            print(f"  Pred range: [{pred_min:.6f}, {pred_max:.6f}]")
            print(f"  Sparse loss: {sparse_pixel_loss.item():.6f}")
            print(f"  State mean: {state_grid.mean().item():.6f}")
            print(f"  Breaking to prevent further collapse...")
            break
        
        # Backpropagation and weight update on EVERY frame
        loss.backward()
        
        # Aggressive gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        
        optimizer.step()
        
        # Detach state grid for next iteration to avoid gradient accumulation
        state_grid = state_grid.detach()
        
        # CRITICAL: Clamp state values to prevent numerical explosion
        state_grid = torch.clamp(state_grid, -10.0, 10.0)
        
        frame_count += 1
        
        # Print progress every 30 frames (roughly 1 second)
        if frame_count % 30 == 0:
            pred_std = prediction_rgb.std().item()
            sparse_loss_val = sparse_pixel_loss.item()
            alive_loss_val = alive_loss.item()
            print(f"Frame {frame_count}: Total={loss.item():.6f}, Sparse={sparse_loss_val:.6f}, Alive={alive_loss_val:.6f}, Pred=[{pred_min:.3f}, {pred_mean:.3f}, {pred_max:.3f}]")

        # --- VISUALIZATION ---
        with torch.no_grad():
            # Prepare frames for display - ensure proper format for OpenCV
            original_vis = (frame_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            sparse_vis = (sparse_input.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Show the PREDICTION channels (0,1,2) - Aman's approach
            prediction_vis = (prediction_rgb.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Ensure arrays are contiguous and properly shaped for OpenCV
            original_vis = np.ascontiguousarray(original_vis)
            sparse_vis = np.ascontiguousarray(sparse_vis)
            prediction_vis = np.ascontiguousarray(prediction_vis)

            # Add text labels
            cv2.putText(original_vis, 'Original (RGB)', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(sparse_vis, 'Input (1%)', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(prediction_vis, f'NCA Prediction (Loss: {loss.item():.4f})', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Combine frames into a single large display window
            combined_display = np.hstack([original_vis, sparse_vis, prediction_vis])
            
            # Resize for better viewing
            h, w, _ = combined_display.shape
            display_width = 1280
            display_height = int(h * (display_width / w))
            large_display = cv2.resize(combined_display, (display_width, display_height))
            
            cv2.imshow('Scotopic Vision NCA - Aman Bhargava Method', cv2.cvtColor(large_display, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Inference stopped.")

if __name__ == '__main__':
    run_live_online_learning() 