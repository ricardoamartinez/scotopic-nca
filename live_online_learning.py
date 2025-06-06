import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
import time
import threading
from queue import Queue

class AdaptiveScotopicNCA(nn.Module):
    """
    Enhanced NCA with error prediction, attention, and multi-scale processing
    """
    def __init__(self):
        super().__init__()
        
        # Enhanced architecture: RGB (3) + Error Prediction (1) + Attention (1) + Communication (6) = 11 channels
        self.rgb_channels = 3
        self.error_channel = 1      # Predicts reconstruction error
        self.attention_channel = 1  # Attention weights
        self.comm_channels = 6      # Communication/memory
        self.n_channels = self.rgb_channels + self.error_channel + self.attention_channel + self.comm_channels
        
        # Multi-scale Sobel filters for better perception
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sobel_y = sobel_x.T
        
        # Large scale gradients for global structure
        sobel_x_large = torch.tensor([
            [-1, -2, 0, 2, 1],
            [-2, -4, 0, 4, 2], 
            [-1, -2, 0, 2, 1]
        ], dtype=torch.float32) / 16.0
        sobel_y_large = torch.tensor([
            [-1, -2, -1],
            [-2, -4, -2],
            [0, 0, 0],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=torch.float32) / 16.0
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        self.register_buffer('sobel_x_large', sobel_x_large.view(1, 1, 3, 5))
        self.register_buffer('sobel_y_large', sobel_y_large.view(1, 1, 5, 3))
        
        # Multi-scale perception network
        perception_size = self.n_channels * 5  # state + grad_x + grad_y + grad_x_large + grad_y_large
        
        # Main update network with attention
        self.update_net = nn.Sequential(
            nn.Conv2d(perception_size, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, self.n_channels, 1, bias=False)
        )
        
        # Error prediction network - predicts where errors will be high
        self.error_predictor = nn.Sequential(
            nn.Conv2d(perception_size, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 1),
            nn.ReLU(), 
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()  # Error prediction in [0,1]
        )
        
        # Attention network - learns where to focus processing
        self.attention_net = nn.Sequential(
            nn.Conv2d(perception_size, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 1), 
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()  # Attention weights in [0,1]
        )
        
        # Initialize with very small weights but allow error and attention networks to be more expressive
        with torch.no_grad():
            for param in self.update_net.parameters():
                param.normal_(0, 0.001)
            for param in self.error_predictor.parameters():
                param.normal_(0, 0.01)  # Slightly larger for error prediction
            for param in self.attention_net.parameters():
                param.normal_(0, 0.01)  # Slightly larger for attention
    
    def perceive(self, state):
        """Multi-scale perception using multiple Sobel filters"""
        batch_size, channels, height, width = state.shape
        
        grad_x_list = []
        grad_y_list = []
        grad_x_large_list = []
        grad_y_large_list = []
        
        for c in range(channels):
            channel = state[:, c:c+1, :, :]
            
            # Fine scale gradients
            gx = F.conv2d(channel, self.sobel_x, padding=1)
            gy = F.conv2d(channel, self.sobel_y, padding=1)
            
            # Large scale gradients (with padding to maintain size)
            gx_large = F.conv2d(channel, self.sobel_x_large, padding=(1, 2))  # 3x5 filter
            gy_large = F.conv2d(channel, self.sobel_y_large, padding=(2, 1))  # 5x3 filter
            
            grad_x_list.append(gx)
            grad_y_list.append(gy)
            grad_x_large_list.append(gx_large)
            grad_y_large_list.append(gy_large)
        
        grad_x = torch.cat(grad_x_list, dim=1)
        grad_y = torch.cat(grad_y_list, dim=1)
        grad_x_large = torch.cat(grad_x_large_list, dim=1)
        grad_y_large = torch.cat(grad_y_large_list, dim=1)
        
        # Multi-scale perception
        perception = torch.cat([state, grad_x, grad_y, grad_x_large, grad_y_large], dim=1)
        return perception
    
    def forward(self, state, sparse_input, sparse_mask, target_frame=None, steps=1):
        """Enhanced forward pass with error prediction and attention"""
        predicted_errors = []
        attention_maps = []
        
        for step in range(steps):
            # Inject sparse input into RGB channels
            rgb_channels = torch.where(
                sparse_mask.bool(),
                sparse_input,
                state[:, :self.rgb_channels, :, :]
            )
            
            # Update error prediction based on sparse input confidence
            input_confidence = sparse_mask.float().mean(dim=1, keepdim=True)  # How much info we have
            uncertainty = 1.0 - input_confidence  # Higher uncertainty with less info
            
            # Update state with current uncertainty estimate
            state = torch.cat([
                rgb_channels,
                uncertainty,  # Error channel based on input sparsity
                state[:, self.rgb_channels + self.error_channel:, :, :]
            ], dim=1)
            
            # Multi-scale perception
            perception = self.perceive(state)
            
            # Predict future error
            predicted_error = self.error_predictor(perception)
            predicted_errors.append(predicted_error)
            
            # Generate attention map
            attention = self.attention_net(perception)
            attention_maps.append(attention)
            
            # Main update with attention weighting
            ds = self.update_net(perception)
            
            # Apply attention to focus updates on important regions
            attention_expanded = attention.expand_as(ds)
            ds = ds * (0.5 + attention_expanded)  # Attention modulates update strength
            
            # Update with adaptive step size based on predicted error
            error_adaptive_step = 0.05 + 0.15 * predicted_error  # Higher step where error predicted
            ds = ds * error_adaptive_step
            
            state = state + ds
            
            # Advanced clamping with channel-specific ranges
            channels = []
            
            # RGB channels [0,1]
            rgb_clamped = torch.clamp(state[:, :self.rgb_channels, :, :], 0, 1)
            channels.append(rgb_clamped)
            
            # Error prediction channel [0,1]
            error_clamped = torch.clamp(state[:, self.rgb_channels:self.rgb_channels+self.error_channel, :, :], 0, 1)
            channels.append(error_clamped)
            
            # Attention channel [0,1]
            attention_clamped = torch.clamp(state[:, self.rgb_channels+self.error_channel:self.rgb_channels+self.error_channel+self.attention_channel, :, :], 0, 1)
            channels.append(attention_clamped)
            
            # Communication channels [-1,1]
            comm_clamped = torch.clamp(state[:, self.rgb_channels+self.error_channel+self.attention_channel:, :, :], -1, 1)
            channels.append(comm_clamped)
            
            state = torch.cat(channels, dim=1)
            
        return state, predicted_errors, attention_maps
    
    def get_rgb(self, state):
        """Extract RGB channels"""
        return torch.clamp(state[:, :self.rgb_channels, :, :], 0, 1)
    
    def get_error_prediction(self, state):
        """Extract error prediction channel"""
        return state[:, self.rgb_channels:self.rgb_channels+self.error_channel, :, :]
    
    def get_attention(self, state):
        """Extract attention channel"""
        return state[:, self.rgb_channels+self.error_channel:self.rgb_channels+self.error_channel+self.attention_channel, :, :]

def rgb_to_lab(rgb_tensor):
    """Convert RGB to LAB color space for perceptual color loss"""
    # Simple approximation of RGB to LAB conversion
    r, g, b = rgb_tensor[:, 0:1, :, :], rgb_tensor[:, 1:2, :, :], rgb_tensor[:, 2:3, :, :]
    
    # Approximate LAB conversion (simplified)
    l = 0.299 * r + 0.587 * g + 0.114 * b  # Luminance
    a = 0.5 * (r - g) + 0.5  # Red-Green
    b_lab = 0.5 * (0.5 * (r + g) - b) + 0.5  # Blue-Yellow
    
    return torch.cat([l, a, b_lab], dim=1)

def perceptual_color_loss(pred_rgb, target_rgb):
    """Perceptual color loss in LAB space"""
    pred_lab = rgb_to_lab(pred_rgb)
    target_lab = rgb_to_lab(target_rgb)
    
    # Weight luminance more heavily
    l_loss = F.mse_loss(pred_lab[:, 0:1, :, :], target_lab[:, 0:1, :, :]) * 2.0
    a_loss = F.mse_loss(pred_lab[:, 1:2, :, :], target_lab[:, 1:2, :, :])
    b_loss = F.mse_loss(pred_lab[:, 2:3, :, :], target_lab[:, 2:3, :, :])
    
    return l_loss + a_loss + b_loss

def create_sparse_mask(frame: torch.Tensor, keep_fraction: float = 0.01) -> torch.Tensor:
    """Creates a binary mask to randomly sample pixels from frame"""
    b, c, h, w = frame.shape
    num_pixels = h * w
    num_to_keep = int(num_pixels * keep_fraction)
    
    indices = torch.randperm(num_pixels, device=frame.device)[:num_to_keep]
    mask_flat = torch.zeros(num_pixels, device=frame.device)
    mask_flat[indices] = 1.0
    
    return mask_flat.view(h, w).unsqueeze(0).unsqueeze(0).repeat(b, c, 1, 1)

def apply_low_light(frame: torch.Tensor, light_level: float = 1.0) -> torch.Tensor:
    """Apply low light simulation with non-linear response"""
    # Simulate sensor response curve in low light
    darkened = frame * light_level
    
    # Non-linear response - shadows get crushed more than highlights
    darkened = torch.pow(darkened, 1.0 + (1.0 - light_level) * 0.8)
    
    # Reduce contrast in low light more aggressively
    contrast_reduction = 0.2 + 0.8 * light_level
    darkened = darkened * contrast_reduction + (1 - contrast_reduction) * 0.4
    
    return torch.clamp(darkened, 0, 1)

def add_noise(frame: torch.Tensor, noise_level: float = 0.0, light_level: float = 1.0) -> torch.Tensor:
    """Add noise that increases in low light conditions"""
    if noise_level > 0:
        # Noise increases in low light (realistic sensor behavior)
        effective_noise = noise_level * (1.0 + 2.0 * (1.0 - light_level))
        
        # Add both Gaussian and salt-and-pepper noise
        gaussian_noise = torch.randn_like(frame) * effective_noise
        
        # Salt and pepper noise (more pronounced in low light)
        if light_level < 0.5:
            salt_pepper = torch.rand_like(frame)
            salt_mask = salt_pepper < 0.01 * effective_noise
            pepper_mask = salt_pepper > (1.0 - 0.01 * effective_noise)
            
            noisy_frame = frame + gaussian_noise
            noisy_frame = torch.where(salt_mask, torch.ones_like(frame), noisy_frame)
            noisy_frame = torch.where(pepper_mask, torch.zeros_like(frame), noisy_frame)
        else:
            noisy_frame = frame + gaussian_noise
            
        return torch.clamp(noisy_frame, 0, 1)
    return frame

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

def create_initial_state(height, width, device, n_channels):
    """Create initial state with enhanced channel initialization"""
    state = torch.zeros(1, n_channels, height, width, device=device)
    
    # Initialize RGB channels with diverse colors
    state[:, 0, :, :] = torch.rand(1, height, width, device=device) * 0.6 + 0.2  # Red
    state[:, 1, :, :] = torch.rand(1, height, width, device=device) * 0.6 + 0.2  # Green  
    state[:, 2, :, :] = torch.rand(1, height, width, device=device) * 0.6 + 0.2  # Blue
    
    # Initialize error prediction to moderate values
    state[:, 3, :, :] = 0.5
    
    # Initialize attention to uniform distribution
    state[:, 4, :, :] = 0.5
    
    # Communication channels start at zero
    state[:, 5:, :, :] = 0.0
    
    return state

def create_control_panel(controls):
    """Create a visual control panel showing current settings"""
    panel = np.zeros((250, 600, 3), dtype=np.uint8)
    
    # Title
    cv2.putText(panel, 'ADAPTIVE SCOTOPIC NCA', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Control values
    y_pos = 70
    cv2.putText(panel, f"Sparsity: {controls['sparsity']:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    y_pos += 35
    cv2.putText(panel, f"Light Level: {controls['light']:.2f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    y_pos += 35
    cv2.putText(panel, f"Noise Level: {controls['noise']:.3f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
    y_pos += 35
    cv2.putText(panel, f"Learning Rate: {controls['lr']:.4f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 1)
    
    # Features
    y_pos += 50
    cv2.putText(panel, 'NO CHEATING: Model only sees sparse input!', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y_pos += 25
    cv2.putText(panel, 'Features: Uncertainty + Attention + Multi-scale', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Instructions
    y_pos += 30
    cv2.putText(panel, 'Use trackbars to adjust - Press Q to quit', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    return panel

def run_live_online_learning(
    resolution: int = 320,  # Good balance of detail and performance
    base_learning_rate: float = 1e-4,  # Conservative learning rate for stability
    model_fps: int = 60,
    initial_sparsity: float = 1.5,
    initial_light: float = 0.5,
    initial_noise: float = 0.02
):
    """
    Adaptive Scotopic NCA with error prediction and attention mechanisms
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Adaptive Scotopic NCA on device: {device}")
    print(f"Resolution: {resolution}x{resolution}")
    print("Features: Error Prediction, Attention, Multi-scale, Perceptual Loss")

    # Initialize webcam
    temp_cap = cv2.VideoCapture(0)
    if not temp_cap.isOpened():
        print("Error: Could not open webcam.")
        return
    temp_cap.release()

    # Control parameters
    controls = {
        'sparsity': initial_sparsity,
        'light': initial_light,  
        'noise': initial_noise,
        'lr': base_learning_rate
    }
    
    # Trackbar callbacks
    def on_sparsity_change(val):
        controls['sparsity'] = 0.01 + (val / 1000.0) * 19.99  # 0.01% to 20%
    
    def on_light_change(val):
        controls['light'] = val / 100.0  # 0.0 to 1.0
    
    def on_noise_change(val):
        controls['noise'] = val / 1000.0  # 0.0 to 0.1
    
    def on_lr_change(val):
        controls['lr'] = (val / 1000.0) * 0.01  # 0.0 to 0.01
    
    # Create control window
    cv2.namedWindow('Adaptive Scotopic NCA Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Adaptive Scotopic NCA Controls', 600, 250)
    
    # Create trackbars
    sparsity_val = int((initial_sparsity - 0.01) * 1000 / 19.99)
    light_val = int(initial_light * 100)
    noise_val = int(initial_noise * 1000)
    lr_val = int(base_learning_rate * 1000 / 0.01)
    
    cv2.createTrackbar('Sparsity %', 'Adaptive Scotopic NCA Controls', sparsity_val, 1000, on_sparsity_change)
    cv2.createTrackbar('Light Level', 'Adaptive Scotopic NCA Controls', light_val, 100, on_light_change)
    cv2.createTrackbar('Noise Level', 'Adaptive Scotopic NCA Controls', noise_val, 100, on_noise_change)
    cv2.createTrackbar('Learning Rate', 'Adaptive Scotopic NCA Controls', lr_val, 1000, on_lr_change)

    # Initialize enhanced NCA
    model = AdaptiveScotopicNCA().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=base_learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.8, verbose=True)
    
    # Initialize frame capture
    frame_capture = AsyncFrameCapture(resolution)
    frame_capture.start()
    
    # Wait for camera
    print("Waiting for camera...")
    while frame_capture.get_current_frame() is None:
        time.sleep(0.1)
    
    # Initialize state
    state = create_initial_state(resolution, resolution, device, model.n_channels)
    
    # Create main display window
    cv2.namedWindow('Adaptive Scotopic NCA - Enhanced Vision', cv2.WINDOW_NORMAL)
    
    print(f"Starting Adaptive Scotopic NCA at {model_fps} FPS")
    print("Enhanced with: Error Prediction, Attention, Multi-scale, Adaptive Learning")
    print("Press 'q' to quit")
    
    try:
        start_time = time.time()
        last_display_time = start_time
        model_step = 0
        loss_history = []
        
        while True:
            loop_start = time.time()
            
            # Get current camera frame
            current_frame_np = frame_capture.get_current_frame()
            if current_frame_np is None:
                time.sleep(0.001)
                continue
                
            frame_tensor = torch.from_numpy(current_frame_np).float().to(device) / 255.0
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)

            # Apply ALL degradations to create the actual input the model sees
            degraded_frame = frame_tensor.clone()
            degraded_frame = apply_low_light(degraded_frame, controls['light'])
            degraded_frame = add_noise(degraded_frame, controls['noise'], controls['light'])
            
            # Create sparse mask - this is the ONLY input the model gets
            sparse_mask = create_sparse_mask(degraded_frame, keep_fraction=controls['sparsity'] / 100.0)
            sparse_input = degraded_frame * sparse_mask  # Only sparse pixels available!

            # Adaptive learning rate based on conditions
            difficulty = (1.0 - controls['light']) + controls['noise'] + (1.0 - controls['sparsity']/100.0)
            adaptive_lr = controls['lr'] * (1.0 + difficulty)
            for param_group in optimizer.param_groups:
                param_group['lr'] = adaptive_lr

            # Run enhanced NCA - model only sees sparse degraded input!
            optimizer.zero_grad()
            
            evolved_state, predicted_errors, attention_maps = model(
                state, sparse_input, sparse_mask, target_frame=None, steps=1
            )
            rgb_prediction = model.get_rgb(evolved_state)
            
            # Multi-component loss
            # 1. Basic reconstruction loss
            reconstruction_loss = F.mse_loss(rgb_prediction, frame_tensor)
            
            # 2. Perceptual color loss
            perceptual_loss = perceptual_color_loss(rgb_prediction, frame_tensor)
            
            # 3. Error prediction loss (model predicts its own uncertainty)
            if len(predicted_errors) > 0:
                # Use prediction confidence - high error prediction where sparse input is missing
                prediction_variance = torch.var(rgb_prediction, dim=1, keepdim=True)
                error_prediction_loss = F.mse_loss(predicted_errors[-1], prediction_variance.detach())
            else:
                error_prediction_loss = 0.0
            
            # 4. Attention regularization (encourage focused attention)
            if len(attention_maps) > 0:
                attention_entropy = -torch.mean(attention_maps[-1] * torch.log(attention_maps[-1] + 1e-8))
                attention_loss = -0.1 * attention_entropy  # Encourage focused attention
            else:
                attention_loss = 0.0
            
            # 5. Sparse input consistency loss (known pixels should be preserved)
            sparse_consistency_loss = F.mse_loss(rgb_prediction * sparse_mask, sparse_input) * 3.0
            
            # Combined loss with adaptive weighting
            total_loss = (reconstruction_loss + 
                         0.5 * perceptual_loss + 
                         0.3 * error_prediction_loss + 
                         0.1 * attention_loss + 
                         0.2 * sparse_consistency_loss)
            
            # Training step
            if not torch.isnan(total_loss) and not torch.isinf(total_loss) and total_loss.item() > 0:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Update learning rate scheduler
                loss_history.append(total_loss.item())
                if len(loss_history) > 50:
                    scheduler.step(np.mean(loss_history[-50:]))
            
            # Update state
            with torch.no_grad():
                state = evolved_state.clone().detach()
            
            # Reset periodically but less frequently for better adaptation
            if model_step % 2000 == 0:
                with torch.no_grad():
                    state = create_initial_state(resolution, resolution, device, model.n_channels)
                print(f"Reset state at step {model_step}")
            
            model_step += 1

            # Enhanced visualization
            current_time = time.time()
            if current_time - last_display_time >= 1.0/30:  # 30fps display
                last_display_time = current_time
                
                with torch.no_grad():
                    # Create all visualizations
                    original_vis = (frame_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    degraded_vis = (degraded_frame.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    
                    # Sparse input visualization - this is ALL the model sees!
                    sparse_vis = (sparse_input.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    
                    # NCA prediction
                    pred_numpy = rgb_prediction.squeeze().permute(1, 2, 0).cpu().numpy()
                    pred_numpy = np.nan_to_num(pred_numpy, nan=0.5, posinf=1.0, neginf=0.0)
                    pred_numpy = np.clip(pred_numpy, 0.0, 1.0)
                    prediction_vis = (pred_numpy * 255).astype(np.uint8)
                    
                    # Error prediction visualization
                    if len(predicted_errors) > 0:
                        error_pred = predicted_errors[-1].squeeze().cpu().numpy()
                        error_pred = np.clip(error_pred, 0, 1)
                        error_vis = (error_pred * 255).astype(np.uint8)
                        error_vis = cv2.applyColorMap(error_vis, cv2.COLORMAP_HOT)
                        error_vis = cv2.cvtColor(error_vis, cv2.COLOR_BGR2RGB)
                    else:
                        error_vis = np.zeros_like(original_vis)
                    
                    # Attention visualization
                    if len(attention_maps) > 0:
                        attention = attention_maps[-1].squeeze().cpu().numpy()
                        attention = np.clip(attention, 0, 1)
                        attention_vis = (attention * 255).astype(np.uint8)
                        attention_vis = cv2.applyColorMap(attention_vis, cv2.COLORMAP_VIRIDIS)
                        attention_vis = cv2.cvtColor(attention_vis, cv2.COLOR_BGR2RGB)
                    else:
                        attention_vis = np.zeros_like(original_vis)
                    
                    # Ensure contiguous arrays
                    frames = [original_vis, degraded_vis, sparse_vis, prediction_vis, error_vis, attention_vis]
                    frames = [np.ascontiguousarray(frame) for frame in frames]
                    
                    # Add labels
                    model_fps_actual = model_step / (current_time - start_time)
                    labels = [
                        'Original (Unknown to Model)',
                        f'Degraded Full Frame',
                        f'Model Input ({controls["sparsity"]:.1f}%)',
                        f'NCA Reconstruction',
                        'Uncertainty Prediction',
                        'Attention Map'
                    ]
                    
                    font_scale = 0.5
                    font_thickness = 1
                    for i, (frame, label) in enumerate(zip(frames, labels)):
                        cv2.putText(frame, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
                        if i == 3:  # Prediction frame gets additional info
                            cv2.putText(frame, f'{model_fps_actual:.0f} FPS', (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                            cv2.putText(frame, f'LR: {adaptive_lr:.4f}', (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                    
                    # Create large display - 3x2 grid
                    display_size = 280
                    large_frames = []
                    for frame in frames:
                        large_frame = cv2.resize(frame, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
                        large_frames.append(large_frame)
                    
                    # Arrange in 3x2 grid
                    top_row = np.hstack([large_frames[0], large_frames[1], large_frames[2]])
                    bottom_row = np.hstack([large_frames[3], large_frames[4], large_frames[5]])
                    combined_display = np.vstack([top_row, bottom_row])
                    
                    # Show displays
                    cv2.imshow('Adaptive Scotopic NCA - Enhanced Vision', cv2.cvtColor(combined_display, cv2.COLOR_RGB2BGR))
                    
                    # Update control panel
                    control_panel = create_control_panel(controls)
                    cv2.imshow('Adaptive Scotopic NCA Controls', control_panel)

                # Progress logging
                if model_step % 60 == 0:
                    model_fps_actual = model_step / (current_time - start_time)
                    
                    recon_val = reconstruction_loss.item() if not torch.isnan(reconstruction_loss) else 0.0
                    perc_val = perceptual_loss.item() if not torch.isnan(perceptual_loss) else 0.0
                    total_val = total_loss.item() if not torch.isnan(total_loss) else 0.0
                    
                    # Calculate how much information is actually available
                    total_pixels = sparse_input.numel() / 3  # Total pixels (divide by 3 for RGB)
                    known_pixels = (sparse_mask.sum().item() / 3)  # Pixels with known values
                    info_ratio = known_pixels / total_pixels
                    
                    print(f"Step {model_step}: Info={info_ratio*100:.2f}% ({known_pixels:.0f}/{total_pixels:.0f} pixels)")
                    print(f"  Conditions: Light={controls['light']:.2f}, Noise={controls['noise']:.3f}")
                    print(f"  Losses: Recon={recon_val:.6f}, Perc={perc_val:.6f}, Total={total_val:.6f}")
                    print(f"  AdaptiveLR={adaptive_lr:.6f}, FPS={model_fps_actual:.1f}")

                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
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
        print(f"Adaptive Scotopic NCA stopped after {model_step} steps")

if __name__ == '__main__':
    run_live_online_learning() 