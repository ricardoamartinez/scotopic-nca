import torch
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np
from scotopic_nca.model import ScotopicNCA
from scotopic_nca.data import create_sparse_mask

def run_live_online_learning(
    frame_size: int = 64,
    updates_per_frame: int = 1,
    learning_rate: float = 1e-4,
    load_pretrained: bool = True,
    model_path: str = 'scotopic_nca_baseline.pth'
):
    """
    Runs live online learning using webcam feed.
    The model continuously adapts to the live camera input.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running live online learning on device: {device}")

    # Initialize model
    model = ScotopicNCA(device=device)
    
    # Optionally load pretrained weights as starting point
    if load_pretrained:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded pretrained model from {model_path}")
        except FileNotFoundError:
            print(f"No pretrained model found at {model_path}, starting from random weights")
    
    model.train()  # Set to training mode for online learning
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize NCA state grid (persists across frames)
    state_grid = torch.zeros(1, model.state_channels, frame_size, frame_size, device=device)
    
    frame_count = 0
    print("Starting live online learning. Press 'q' to quit.")
    print("The model will continuously adapt to your camera feed!")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
            frame_tensor = torch.from_numpy(resized_frame).float().to(device) / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

            # Create sparse input (only 1% of pixels)
            mask = create_sparse_mask(frame_tensor, keep_fraction=0.01).to(device)
            sparse_input = frame_tensor * mask

            # ONLINE LEARNING: Update model weights on this frame
            optimizer.zero_grad()
            
            # Inject sparse input and run NCA
            new_state = state_grid.clone()
            new_state[:, model.input_channel:model.input_channel+1, :, :] = sparse_input
            state_grid = model(new_state, steps=updates_per_frame)
            
            # Get prediction
            prediction = state_grid[:, model.pred_channel:model.pred_channel+1, :, :]
            
            # Compute loss - try to reconstruct the full frame from sparse input
            reconstruction_loss = F.mse_loss(prediction, frame_tensor)
            sparse_loss = F.mse_loss(prediction * mask, sparse_input)
            total_loss = reconstruction_loss + sparse_loss
            
            # Backpropagation and weight update
            total_loss.backward()
            optimizer.step()

            # Visualization
            original_vis = (frame_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
            sparse_vis = (sparse_input.squeeze().cpu().numpy() * 255).astype(np.uint8)
            reconstructed_vis = (prediction.detach().squeeze().cpu().numpy() * 255).astype(np.uint8)

            # Add text labels with loss information
            cv2.putText(original_vis, 'Original', (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255), 1)
            cv2.putText(sparse_vis, 'Input (1%)', (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255), 1)
            cv2.putText(reconstructed_vis, f'NCA Learning (Loss: {total_loss.item():.4f})', (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255), 1)
            cv2.putText(reconstructed_vis, f'Frame: {frame_count}', (2, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255), 1)

            # Combine all three views
            combined_display = np.hstack([original_vis, sparse_vis, reconstructed_vis])
            cv2.imshow('Live Scotopic NCA - Online Learning', combined_display)

            frame_count += 1
            
            # Print progress
            if frame_count % 30 == 0:  # Every ~1 second at 30fps
                print(f"Frame {frame_count}, Loss: {total_loss.item():.6f}")

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Save the adapted model
        save_path = 'scotopic_nca_online_adapted.pth'
        torch.save(model.state_dict(), save_path)
        print(f"Online adapted model saved to {save_path}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Online learning stopped.")

if __name__ == '__main__':
    run_live_online_learning() 