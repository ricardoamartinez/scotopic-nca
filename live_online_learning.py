import torch
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np
import time
import threading
from queue import Queue
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
            time.sleep(1/30)  # 30 FPS capture
            
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

def run_live_online_learning(
    resolution: int = None,  # Full native resolution as requested
    learning_rate: float = 1e-5,  # Conservative for full resolution stability  
    model_fps: int = 60,   # Slower for full resolution + many steps
    camera_fps: int = 30   # Camera capture rate
):
    """
    Async Scotopic NCA with decoupled model/camera rates and local minima escape mechanisms.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running async online learning on device: {device}")
    print(f"Model FPS: {model_fps}, Camera FPS: {camera_fps}")

    # Initialize webcam and resolution
    temp_cap = cv2.VideoCapture(0)
    if not temp_cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    if resolution is None:
        width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution = min(width, height)  # FULL native resolution as requested
        print(f"Using FULL native resolution: {resolution}x{resolution}")
    temp_cap.release()

    # Initialize model with simpler architecture (closer to Aman's approach)
    model = RobustScotopicNCA(hidden_channels=6, device=device)  # Fewer hidden channels  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize async frame capture
    frame_capture = AsyncFrameCapture(resolution)
    frame_capture.start()
    
    # Wait for first frame
    print("Waiting for camera...")
    while frame_capture.get_current_frame() is None:
        time.sleep(0.1)
    
    # Initialize NCA state grid with better starting conditions
    state_grid = torch.zeros(1, model.state_channels, resolution, resolution, device=device)
    # Initialize prediction channels with more diverse colors (prevent black convergence)
    state_grid[:, 0, :, :] = torch.rand(1, resolution, resolution, device=device) * 0.3 + 0.2  # Red: 0.2-0.5
    state_grid[:, 1, :, :] = torch.rand(1, resolution, resolution, device=device) * 0.3 + 0.2  # Green: 0.2-0.5  
    state_grid[:, 2, :, :] = torch.rand(1, resolution, resolution, device=device) * 0.3 + 0.2  # Blue: 0.2-0.5
    # Initialize hidden channels with small random values
    state_grid[:, model.prediction_channels+model.input_channels:, :, :] = torch.randn(1, model.state_channels-model.prediction_channels-model.input_channels, resolution, resolution, device=device) * 0.01
    
    print(f"Starting async learning at {model_fps} FPS. Press 'q' to quit.")
    
    model_step = 0
    
    try:
        start_time = time.time()
        last_display_time = start_time
        
        while True:
            loop_start = time.time()
            
            # Get current frame from async capture
            current_frame_np = frame_capture.get_current_frame()
            if current_frame_np is None:
                time.sleep(0.001)
                continue
                
            # Convert to tensor - same frame might be used multiple times with different sampling
            frame_tensor = torch.from_numpy(current_frame_np).float().to(device) / 255.0
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)

            # --- ONLINE LEARNING STEP ---
            optimizer.zero_grad()
            
            # Create NEW sparse mask each step (different sampling of same frame)
            mask = create_sparse_mask(frame_tensor).to(device)
            sparse_input = frame_tensor * mask

            # Inject sparse input into INPUT REGISTER channels (3,4,5)
            new_state_grid = state_grid.clone()
            new_state_grid[:, model.prediction_channels:model.prediction_channels+model.input_channels, :, :] = sparse_input

            # No exploration noise - keep it simple like Aman's approach

            # Run MANY update steps - Aman's key insight: "information travels 1 pixel at a time"
            # Need 50-100+ steps for information to propagate across the image
            steps = max(50, resolution // 8)  # Scale steps with resolution 
            state_grid = model(new_state_grid, steps=steps, decay=0.999)

            # Get RGB predictions
            prediction_rgb = model.get_prediction_rgb(state_grid)
            
            # AMAN'S SIMPLE APPROACH: Just L2 loss between prediction and sparse input where data exists
            # "We will compare the cell's estimate value to the incoming value to compute loss. We will use L2 loss."
            loss = F.mse_loss(prediction_rgb * mask, sparse_input)
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            
            # Detach and clamp state
            state_grid = state_grid.detach()
            state_grid = torch.clamp(state_grid, -10.0, 10.0)
            
            # Simple monitoring - no complex local minima detection
            
            model_step += 1

            # --- VISUALIZATION (less frequent than model updates) ---
            current_time = time.time()
            if current_time - last_display_time >= 1.0/30:  # 30 FPS display
                last_display_time = current_time
                
                with torch.no_grad():
                    # Stats for monitoring
                    pred_mean = prediction_rgb.mean().item()
                    pred_max = prediction_rgb.max().item()
                    pred_min = prediction_rgb.min().item()
                    pred_std = prediction_rgb.std().item()
                    
                    # Prepare display frames
                    original_vis = (frame_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    sparse_vis = (sparse_input.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    prediction_vis = (prediction_rgb.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    
                    # Ensure contiguous arrays
                    original_vis = np.ascontiguousarray(original_vis)
                    sparse_vis = np.ascontiguousarray(sparse_vis)
                    prediction_vis = np.ascontiguousarray(prediction_vis)

                    # Add enhanced status text
                    model_fps_actual = model_step / (current_time - start_time)
                    cv2.putText(original_vis, 'Original (RGB)', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(sparse_vis, 'Sparse (1%)', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(prediction_vis, f'Aman NCA ({model_fps_actual:.0f}fps)', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.putText(prediction_vis, f'Steps: {steps}', (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    cv2.putText(prediction_vis, f'Range: [{pred_min:.2f}, {pred_max:.2f}]', (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                    # Combine and display
                    combined_display = np.hstack([original_vis, sparse_vis, prediction_vis])
                    h, w, _ = combined_display.shape
                    display_width = 1200
                    display_height = int(h * (display_width / w))
                    large_display = cv2.resize(combined_display, (display_width, display_height))
                    
                    cv2.imshow('Aman Bhargava Scotopic NCA - Exact Implementation', cv2.cvtColor(large_display, cv2.COLOR_RGB2BGR))

                # Print progress every 120 model steps (~1 second at 120fps)  
                if model_step % 120 == 0:
                    print(f"Step {model_step}: Loss={loss.item():.6f}, Steps={steps}, "
                          f"ModelFPS={model_fps_actual:.1f}, Pred=[{pred_min:.3f}, {pred_max:.3f}], Std={pred_std:.3f}")

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Timing control for target model FPS
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