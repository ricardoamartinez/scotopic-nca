import torch
import cv2
import numpy as np
from scotopic_nca.model import ScotopicNCA
from scotopic_nca.data import create_sparse_mask

def run_live_inference(
    model_path: str = 'scotopic_nca_baseline.pth',
    frame_size: int = 64,
    updates_per_frame: int = 1
):
    """
    Runs live inference using a pre-trained ScotopicNCA model on a webcam feed.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running live inference on device: {device}")

    # 1. Load the pre-trained model
    model = ScotopicNCA(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode

    # 2. Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # 3. Initialize NCA state grid (this persists across frames)
    state_grid = torch.zeros(1, model.state_channels, frame_size, frame_size, device=device)

    print("Starting live feed. Press 'q' to quit.")
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 4. Pre-process the camera frame
            # Convert to grayscale, resize, normalize, and add batch/channel dims
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (frame_size, frame_size))
            frame_tensor = torch.from_numpy(resized_frame).float().to(device) / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0) # Shape: (1, 1, H, W)

            # 5. Create sparse input
            mask = create_sparse_mask(frame_tensor).to(device)
            sparse_input = frame_tensor * mask

            # 6. Inject input and run NCA update
            new_state = torch.zeros_like(state_grid)
            new_state[:, :, :, :] = state_grid.detach()[:, :, :, :]
            new_state[:, model.input_channel:model.input_channel+1, :, :] = sparse_input
            state_grid = model(new_state, steps=updates_per_frame)

            # 7. Get the reconstructed frame
            reconstructed_frame = state_grid[:, model.pred_channel:model.pred_channel+1, :, :]

            # 8. Prepare frames for visualization
            original_vis = (frame_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
            sparse_vis = (sparse_input.squeeze().cpu().numpy() * 255).astype(np.uint8)
            reconstructed_vis = (reconstructed_frame.squeeze().cpu().numpy() * 255).astype(np.uint8)

            # Add text labels
            cv2.putText(original_vis, 'Original', (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255), 1)
            cv2.putText(sparse_vis, 'Input (1%)', (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255), 1)
            cv2.putText(reconstructed_vis, 'NCA Reconstruction', (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255), 1)

            # Combine frames into a single display window
            combined_display = np.hstack([original_vis, sparse_vis, reconstructed_vis])
            cv2.imshow('Live Scotopic NCA Reconstruction', combined_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 9. Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Inference stopped.")

if __name__ == '__main__':
    run_live_inference() 