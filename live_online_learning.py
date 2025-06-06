import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
import time
import threading
from queue import Queue

class ScotopicMordvintsevNCA(nn.Module):
    """
    Hybrid approach: Mordvintsev's local perception + Scotopic reconstruction
    """
    def __init__(self, hidden_channels=8):
        super().__init__()
        
        # Channel setup: RGB (3) + sparse input (3) + hidden (8) = 14 total
        self.n_channels = 3 + 3 + hidden_channels  # RGB + sparse input + hidden
        self.hidden_channels = hidden_channels
        
        # Sobel filters for gradient perception (fixed, not learned)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sobel_y = sobel_x.T
        
        # Register as buffers (fixed, won't be updated)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
        # Update rule network (perception -> update)
        # Input: state + grad_x + grad_y = 14 + 14 + 14 = 42 channels
        perception_size = self.n_channels * 3
        
        # Deeper network for better pattern learning
        self.update_net = nn.Sequential(
            nn.Conv2d(perception_size, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, self.n_channels, 1, bias=False)  # No bias in final layer
        )
        
        # Initialize final layer to small random values (not zero)
        with torch.no_grad():
            self.update_net[-1].weight.normal_(0, 0.01)
    
    def perceive(self, state):
        """Perception step: compute gradients using Sobel filters"""
        batch_size, channels, height, width = state.shape
        
        # Compute gradients for each channel
        grad_x_list = []
        grad_y_list = []
        
        for c in range(channels):
            # Apply Sobel filters to each channel
            channel = state[:, c:c+1, :, :]  # (B, 1, H, W)
            
            gx = F.conv2d(channel, self.sobel_x, padding=1)
            gy = F.conv2d(channel, self.sobel_y, padding=1)
            
            grad_x_list.append(gx)
            grad_y_list.append(gy)
        
        grad_x = torch.cat(grad_x_list, dim=1)  # (B, channels, H, W)
        grad_y = torch.cat(grad_y_list, dim=1)  # (B, channels, H, W)
        
        # Concatenate: [state, grad_x, grad_y]
        perception = torch.cat([state, grad_x, grad_y], dim=1)  # (B, 42, H, W)
        return perception
    
    def update_rule(self, perception):
        """Update rule: perception -> state delta"""
        return self.update_net(perception)
    
    def stochastic_update(self, state, ds, update_probability=0.5):
        """Stochastic update masking as in paper"""
        batch_size, channels, height, width = state.shape
        
        # Random mask per cell
        rand_mask = torch.rand(batch_size, 1, height, width, device=state.device) < update_probability
        rand_mask = rand_mask.float()
        
        # Apply stochastic masking
        ds_masked = ds * rand_mask
        return state + ds_masked
    
    def inject_sparse_input(self, state, sparse_input, sparse_mask):
        """Inject sparse input directly into input channels"""
        # Channels 3-5 are for sparse input
        state[:, 3:6, :, :] = sparse_input * sparse_mask
        return state
    
    def forward(self, state, sparse_input, sparse_mask, steps=1, update_probability=0.5):
        """Forward pass: run NCA for multiple steps with sparse input injection"""
        for step in range(steps):
            # Inject sparse input at each step (key for scotopic vision)
            state = self.inject_sparse_input(state, sparse_input, sparse_mask)
            
            # Perception
            perception = self.perceive(state)
            
            # Update rule
            ds = self.update_rule(perception)
            
            # Stochastic update
            state = self.stochastic_update(state, ds, update_probability)
            
            # Clamp to prevent explosion
            state = torch.clamp(state, -3, 3)
        
        return state
    
    def get_rgb(self, state):
        """Extract RGB channels"""
        return torch.sigmoid(state[:, :3, :, :])  # Apply sigmoid to get [0,1] range

def create_sparse_mask(frame: torch.Tensor, keep_fraction: float = 0.01) -> torch.Tensor:
    """Creates a binary mask to randomly sample pixels from frame"""
    b, c, h, w = frame.shape
    num_pixels = h * w
    num_to_keep = int(num_pixels * keep_fraction)
    
    indices = torch.randperm(num_pixels, device=frame.device)[:num_to_keep]
    mask_flat = torch.zeros(num_pixels, device=frame.device)
    mask_flat[indices] = 1.0
    
    return mask_flat.view(h, w).unsqueeze(0).unsqueeze(0).repeat(b, c, 1, 1)

class AsyncFrameCapture:
    """Async camera capture to decouple from model updates"""
    def __init__(self, resolution):
        self.cap = cv2.VideoCapture(0)
        self.resolution = resolution
        self.current_frame = None
        self.running = True
        self.lock = threading.Lock()
        
    def capture_loop(self):
        """Continuous frame capture in separate thread"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized_frame = cv2.resize(frame_rgb, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
                with self.lock:
                    self.current_frame = resized_frame
            time.sleep(1/120)  # 120 FPS capture
            
    def get_current_frame(self):
        """Get the most recent frame (thread-safe)"""
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None
            
    def start(self):
        """Start async capture"""
        self.thread = threading.Thread(target=self.capture_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop capture and cleanup"""
        self.running = False
        self.cap.release()

def create_initial_state(n_channels, height, width, device):
    """Create initial state with random colors"""
    state = torch.zeros(1, n_channels, height, width, device=device)
    
    # Initialize RGB channels with random colors (not gray!)
    state[:, 0, :, :] = torch.rand(1, height, width, device=device)  # Red
    state[:, 1, :, :] = torch.rand(1, height, width, device=device)  # Green
    state[:, 2, :, :] = torch.rand(1, height, width, device=device)  # Blue
    
    # Sparse input channels start at zero
    state[:, 3:6, :, :] = 0.0
    
    # Hidden channels with small random values
    state[:, 6:, :, :] = torch.randn(1, n_channels - 6, height, width, device=device) * 0.1
    
    return state

def run_live_online_learning(
    resolution: int = None,
    learning_rate: float = 2e-3,
    model_fps: int = 120,  # Maximum FPS for model updates
    camera_fps: int = 120  # Maximum FPS for camera capture
):
    """
    Hybrid Scotopic-Mordvintsev approach:
    Use local perception for communication, direct training for reconstruction
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Hybrid Scotopic-Mordvintsev NCA on device: {device}")

    # Initialize webcam and resolution
    temp_cap = cv2.VideoCapture(0)
    if not temp_cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    if resolution is None:
        width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        native_res = min(width, height)
        resolution = native_res  # Full native resolution 1:1
        print(f"Using full resolution: {resolution}x{resolution}")
    temp_cap.release()

    # Initialize hybrid NCA
    model = ScotopicMordvintsevNCA(hidden_channels=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize frame capture
    frame_capture = AsyncFrameCapture(resolution)
    frame_capture.start()
    
    # Wait for camera
    print("Waiting for camera...")
    while frame_capture.get_current_frame() is None:
        time.sleep(0.1)
    
    # Initialize state
    state = create_initial_state(model.n_channels, resolution, resolution, device)
    
    print(f"Starting hybrid learning at max {model_fps} FPS. Press 'q' to quit.")
    
    try:
        start_time = time.time()
        last_display_time = start_time
        model_step = 0
        
        while True:
            loop_start = time.time()
            
            # Get current camera frame
            current_frame_np = frame_capture.get_current_frame()
            if current_frame_np is None:
                time.sleep(0.001)
                continue
                
            frame_tensor = torch.from_numpy(current_frame_np).float().to(device) / 255.0
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)

            # Create sparse input (2% for max FPS)
            sparse_mask = create_sparse_mask(frame_tensor, keep_fraction=0.02)
            sparse_input = frame_tensor

            # Run NCA evolution
            optimizer.zero_grad()
            
            # Minimal evolution steps for max FPS
            steps = torch.randint(1, 3, (1,)).item()  # Ultra-short steps for max FPS
            evolved_state = model(state, sparse_input, sparse_mask, steps=steps, update_probability=0.7)
            
            # Get RGB predictions
            rgb_prediction = model.get_rgb(evolved_state)
            
            # Loss: reconstruction on ALL pixels (not just sparse)
            reconstruction_loss = F.mse_loss(rgb_prediction, frame_tensor)
            
            # Sparse consistency: predictions should match sparse input where available
            sparse_consistency_loss = F.mse_loss(rgb_prediction * sparse_mask, frame_tensor * sparse_mask)
            
            # Smoothness for natural images
            grad_x = torch.abs(rgb_prediction[:, :, :, 1:] - rgb_prediction[:, :, :, :-1])
            grad_y = torch.abs(rgb_prediction[:, :, 1:, :] - rgb_prediction[:, :, :-1, :])
            smoothness_loss = grad_x.mean() + grad_y.mean()
            
            # Hidden state regularization to prevent explosion
            hidden_reg = (evolved_state[:, 6:, :, :] ** 2).mean()
            
            # Total loss with balanced weights
            loss = reconstruction_loss + 2.0 * sparse_consistency_loss + 0.1 * smoothness_loss + 0.01 * hidden_reg
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update state for next iteration
            state = evolved_state.detach()
            
            # Reinitialize less frequently for better FPS
            if model_step % 300 == 0:
                state = create_initial_state(model.n_channels, resolution, resolution, device)
            
            model_step += 1

            # Visualization
            current_time = time.time()
            if current_time - last_display_time >= 1.0/60:  # 60 FPS display (max practical for visual)
                last_display_time = current_time
                
                with torch.no_grad():
                    # Create sparse input for visualization
                    sparse_vis_input = frame_tensor * sparse_mask
                    
                    # Create visualizations
                    original_vis = (frame_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    sparse_vis = (sparse_vis_input.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    prediction_vis = (rgb_prediction.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    
                    # Ensure contiguous arrays for OpenCV
                    original_vis = np.ascontiguousarray(original_vis)
                    sparse_vis = np.ascontiguousarray(sparse_vis)
                    prediction_vis = np.ascontiguousarray(prediction_vis)
                    
                    # Add text overlays
                    model_fps_actual = model_step / (current_time - start_time)
                    cv2.putText(original_vis, 'Original', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(sparse_vis, f'Sparse 2%', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(prediction_vis, f'Hybrid NCA', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    cv2.putText(prediction_vis, f'{model_fps_actual:.0f} FPS', (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                    # Combine and display
                    combined_display = np.hstack([original_vis, sparse_vis, prediction_vis])
                    h, w, _ = combined_display.shape
                    display_width = 1200
                    display_height = int(h * (display_width / w))
                    large_display = cv2.resize(combined_display, (display_width, display_height))
                    
                    cv2.imshow('Hybrid Scotopic-Mordvintsev NCA', cv2.cvtColor(large_display, cv2.COLOR_RGB2BGR))

                # Progress logging (less frequent for better FPS)
                if model_step % 120 == 0:
                    model_fps_actual = model_step / (current_time - start_time)
                    rgb_mean = rgb_prediction.mean().item()
                    rgb_std = rgb_prediction.std().item()
                    rgb_min = rgb_prediction.min().item()
                    rgb_max = rgb_prediction.max().item()
                    print(f"Step {model_step}: Loss={loss.item():.6f} (Recon={reconstruction_loss.item():.6f}, "
                          f"Sparse={sparse_consistency_loss.item():.6f}), FPS={model_fps_actual:.1f}, "
                          f"RGB=[{rgb_min:.3f}, {rgb_max:.3f}], Mean={rgb_mean:.3f}±{rgb_std:.3f}")

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Timing control
            loop_time = time.time() - loop_start
            target_loop_time = 1.0 / model_fps
            if loop_time < target_loop_time:
                time.sleep(target_loop_time - loop_time)
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        frame_capture.stop()
        cv2.destroyAllWindows()
        print(f"Stopped after {model_step} model steps")

if __name__ == '__main__':
    run_live_online_learning() 