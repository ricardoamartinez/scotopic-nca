import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
import time
import threading
from queue import Queue

class SimpleStableNCA(nn.Module):
    """
    Much simpler, more stable NCA for RGB reconstruction
    No living/dead mechanism, just basic RGB + communication channels
    """
    def __init__(self):
        super().__init__()
        
        # Simple architecture: RGB (3) + communication (3) = 6 channels total
        self.n_channels = 6  # Much simpler than Growing NCA
        
        # Sobel filters for perception
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sobel_y = sobel_x.T
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
        # Very simple update network
        perception_size = self.n_channels * 3  # 18 channels: state + grad_x + grad_y
        
        self.update_net = nn.Sequential(
            nn.Conv2d(perception_size, 32, 1),  # Much smaller network
            nn.ReLU(),
            nn.Conv2d(32, self.n_channels, 1, bias=False)
        )
        
        # Initialize with very small weights to prevent explosion
        with torch.no_grad():
            for param in self.parameters():
                param.normal_(0, 0.001)  # Very small initialization
    
    def perceive(self, state):
        """Simple perception using Sobel filters"""
        batch_size, channels, height, width = state.shape
        
        grad_x_list = []
        grad_y_list = []
        
        for c in range(channels):
            channel = state[:, c:c+1, :, :]
            
            gx = F.conv2d(channel, self.sobel_x, padding=1)
            gy = F.conv2d(channel, self.sobel_y, padding=1)
            
            grad_x_list.append(gx)
            grad_y_list.append(gy)
        
        grad_x = torch.cat(grad_x_list, dim=1)
        grad_y = torch.cat(grad_y_list, dim=1)
        
        perception = torch.cat([state, grad_x, grad_y], dim=1)
        return perception
    
    def forward(self, state, sparse_input, sparse_mask, steps=1):
        """Simple forward pass without stochastic updates or living masks"""
        for step in range(steps):
            # Direct sparse injection (avoid in-place operations)
            rgb_channels = torch.where(
                sparse_mask.bool(),
                sparse_input,
                state[:, 0:3, :, :]
            )
            
            # Reconstruct state without in-place operations
            state = torch.cat([rgb_channels, state[:, 3:6, :, :]], dim=1)
            
            # Perception
            perception = self.perceive(state)
            
            # Update rule
            ds = self.update_net(perception)
            
            # Simple update with small step size for stability
            state = state + ds * 0.1  # Very small step size
            
            # Clamp RGB channels to [0,1] and communication channels to [-1,1] (non-inplace)
            rgb_clamped = torch.clamp(state[:, 0:3, :, :], 0, 1)  # RGB stays positive
            comm_clamped = torch.clamp(state[:, 3:6, :, :], -1, 1)  # Communication can be negative
            state = torch.cat([rgb_clamped, comm_clamped], dim=1)
        
        return state
    
    def get_rgb(self, state):
        """Extract RGB channels with full color range"""
        # Don't use sigmoid - just clamp to [0,1] to preserve full color diversity
        return torch.clamp(state[:, :3, :, :], 0, 1)

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
            time.sleep(1/120)
            
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

def create_initial_state(height, width, device):
    """Create simple initial state with diverse RGB initialization"""
    state = torch.zeros(1, 6, height, width, device=device)
    
    # Initialize RGB channels with random colors to encourage diversity
    # Each pixel gets its own random RGB value
    state[:, 0, :, :] = torch.rand(1, height, width, device=device) * 0.6 + 0.2  # Red: [0.2, 0.8]
    state[:, 1, :, :] = torch.rand(1, height, width, device=device) * 0.6 + 0.2  # Green: [0.2, 0.8]
    state[:, 2, :, :] = torch.rand(1, height, width, device=device) * 0.6 + 0.2  # Blue: [0.2, 0.8]
    
    # Communication channels start at zero
    state[:, 3:6, :, :] = 0.0
    
    return state

def run_live_online_learning(
    resolution: int = None,
    learning_rate: float = 1e-4,  # Much lower learning rate
    model_fps: int = 120,
    camera_fps: int = 120
):
    """
    Simple stable NCA for RGB reconstruction - no collapse issues
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Simple Stable NCA on device: {device}")

    # Initialize webcam and resolution
    temp_cap = cv2.VideoCapture(0)
    if not temp_cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    if resolution is None:
        width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        native_res = min(width, height)
        resolution = native_res
        print(f"Using full resolution: {resolution}x{resolution}")
    temp_cap.release()

    # Initialize simple NCA
    model = SimpleStableNCA().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize frame capture
    frame_capture = AsyncFrameCapture(resolution)
    frame_capture.start()
    
    # Wait for camera
    print("Waiting for camera...")
    while frame_capture.get_current_frame() is None:
        time.sleep(0.1)
    
    # Initialize state
    state = create_initial_state(resolution, resolution, device)
    
    print(f"Starting Simple Stable NCA at max {model_fps} FPS. Press 'q' to quit.")
    print("Much more stable - no collapse issues!")
    
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

            # Create sparse input with enough information for color learning
            sparse_mask = create_sparse_mask(frame_tensor, keep_fraction=0.05)  # 5% for better color learning
            sparse_input = frame_tensor

            # Run simple NCA
            optimizer.zero_grad()
            
            # Very few steps for stability
            steps = 1  # Just 1 step per iteration for stability
            evolved_state = model(state, sparse_input, sparse_mask, steps=steps)
            
            # Get RGB predictions
            rgb_prediction = model.get_rgb(evolved_state)
            
            # Main reconstruction loss
            reconstruction_loss = F.mse_loss(rgb_prediction, frame_tensor)
            
            # Color diversity loss - encourage RGB channels to be different from each other
            r_channel = rgb_prediction[:, 0, :, :]
            g_channel = rgb_prediction[:, 1, :, :]
            b_channel = rgb_prediction[:, 2, :, :]
            
            # Penalty for channels being too similar (encourages color diversity)
            rg_similarity = F.mse_loss(r_channel, g_channel)
            rb_similarity = F.mse_loss(r_channel, b_channel)
            gb_similarity = F.mse_loss(g_channel, b_channel)
            
            # We want to minimize similarity (maximize diversity), so subtract it
            color_diversity_loss = -0.1 * (rg_similarity + rb_similarity + gb_similarity)
            
            # Combined loss
            loss = reconstruction_loss + color_diversity_loss
            
            # Very safe training
            if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() > 0:
                loss.backward()
                # Very aggressive gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
            
            # Update state for next iteration (properly detach to avoid gradient accumulation)
            with torch.no_grad():
                state = evolved_state.clone().detach()
            
            # Reset periodically to prevent drift
            if model_step % 1000 == 0:
                with torch.no_grad():
                    state = create_initial_state(resolution, resolution, device)
            
            model_step += 1

            # Visualization
            current_time = time.time()
            if current_time - last_display_time >= 1.0/60:
                last_display_time = current_time
                
                with torch.no_grad():
                    # Create sparse input for visualization
                    sparse_vis_input = frame_tensor * sparse_mask
                    
                    # Create visualizations with NaN safety
                    original_vis = (frame_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    sparse_vis = (sparse_vis_input.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    
                    pred_numpy = rgb_prediction.squeeze().permute(1, 2, 0).cpu().numpy()
                    pred_numpy = np.nan_to_num(pred_numpy, nan=0.5, posinf=1.0, neginf=0.0)
                    pred_numpy = np.clip(pred_numpy, 0.0, 1.0)
                    prediction_vis = (pred_numpy * 255).astype(np.uint8)
                    
                    # Communication channels visualization
                    comm_channels = evolved_state[0, 3:6, :, :].mean(dim=0).cpu().numpy()
                    comm_channels = np.nan_to_num(comm_channels, nan=0.0, posinf=1.0, neginf=0.0)
                    comm_channels = (comm_channels - comm_channels.min()) / (comm_channels.max() - comm_channels.min() + 1e-8)
                    comm_vis = (comm_channels * 255).astype(np.uint8)
                    comm_vis = cv2.cvtColor(comm_vis, cv2.COLOR_GRAY2RGB)
                    
                    # Ensure contiguous arrays for OpenCV
                    original_vis = np.ascontiguousarray(original_vis)
                    sparse_vis = np.ascontiguousarray(sparse_vis)
                    prediction_vis = np.ascontiguousarray(prediction_vis)
                    comm_vis = np.ascontiguousarray(comm_vis)
                    
                    # Add text overlays
                    model_fps_actual = model_step / (current_time - start_time)
                    cv2.putText(original_vis, 'Original', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(sparse_vis, f'Sparse 5%', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.putText(prediction_vis, f'Simple NCA', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    cv2.putText(comm_vis, f'Communication', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(prediction_vis, f'{model_fps_actual:.0f} FPS', (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                    # Combine and display (4 panels)
                    top_row = np.hstack([original_vis, sparse_vis])
                    bottom_row = np.hstack([prediction_vis, comm_vis])
                    combined_display = np.vstack([top_row, bottom_row])
                    
                    h, w, _ = combined_display.shape
                    display_width = 1200
                    display_height = int(h * (display_width / w))
                    large_display = cv2.resize(combined_display, (display_width, display_height))
                    
                    cv2.imshow('Simple Stable NCA', cv2.cvtColor(large_display, cv2.COLOR_RGB2BGR))

                # Progress logging
                if model_step % 120 == 0:
                    model_fps_actual = model_step / (current_time - start_time)
                    
                    # Safe stats
                    def safe_stat(tensor, default=0.0):
                        val = tensor.item() if hasattr(tensor, 'item') else tensor
                        return default if (np.isnan(val) or np.isinf(val)) else val
                    
                    rgb_mean = safe_stat(rgb_prediction.mean())
                    rgb_std = safe_stat(rgb_prediction.std())
                    loss_val = safe_stat(loss)
                    recon_loss_val = safe_stat(reconstruction_loss)
                    
                    # Color channel statistics
                    r_mean = safe_stat(rgb_prediction[:, 0, :, :].mean())
                    g_mean = safe_stat(rgb_prediction[:, 1, :, :].mean())
                    b_mean = safe_stat(rgb_prediction[:, 2, :, :].mean())
                    
                    r_std = safe_stat(rgb_prediction[:, 0, :, :].std())
                    g_std = safe_stat(rgb_prediction[:, 1, :, :].std())
                    b_std = safe_stat(rgb_prediction[:, 2, :, :].std())
                    
                    print(f"Step {model_step}: ReconLoss={recon_loss_val:.6f}, TotalLoss={loss_val:.6f}, FPS={model_fps_actual:.1f}")
                    print(f"  RGB means: R={r_mean:.3f}, G={g_mean:.3f}, B={b_mean:.3f}")
                    print(f"  RGB stds:  R={r_std:.3f}, G={g_std:.3f}, B={b_std:.3f} - COLOR DIVERSITY!")

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