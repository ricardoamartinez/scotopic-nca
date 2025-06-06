# Scotopic Neural Cellular Automata - Live Online Learning Demo

This project implements a Neural Cellular Automata (NCA) that performs **online learning** on a live camera feed, reconstructing full frames from extremely sparse pixel data (scotopic vision) in real-time.

## Key Features

- **Online Learning**: The model continuously adapts its weights as new webcam frames arrive
- **Real-time Reconstruction**: Reconstructs full 64x64 images from only 1% of randomly sampled pixels
- **Live Visualization**: Shows original, sparse input, and reconstructed frames side-by-side with live loss updates

## Project Structure

- `scotopic_nca/model.py`: Defines the `ScotopicNCA` PyTorch model
- `scotopic_nca/data.py`: Utilities for generating synthetic video data and sparse masks
- `scotopic_nca/training.py`: Optional offline training on synthetic data (for pretraining)
- `live_inference.py`: **Main script** - Live online learning with webcam feed
- `tests/`: Contains `pytest` tests for verification

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Live Online Learning:**
    ```bash
    python live_inference.py
    ```
    This opens your webcam and starts the online learning process. The model will:
    - Capture frames from your camera
    - Show only 1% of pixels as sparse input
    - Continuously learn to reconstruct the full frame
    - Display real-time loss and frame count
    - Save the adapted model when you quit (press 'q')

3.  **Optional - Pretrain on Synthetic Data:**
    ```bash
    python -m scotopic_nca.training
    ```
    This creates a `scotopic_nca_baseline.pth` file that provides better initial weights for online learning.

4.  **Run Tests:**
    ```bash
    pytest
    ```

## How It Works

**Online Learning Process:**
1. **Frame Capture**: Live webcam frames are converted to 64x64 grayscale
2. **Sparse Sampling**: Only 1% of pixels are randomly selected as input
3. **NCA Processing**: The Neural Cellular Automaton attempts to reconstruct the full frame
4. **Loss Computation**: Reconstruction loss between predicted and actual frames
5. **Weight Update**: Backpropagation updates the model weights for the next frame
6. **Repeat**: Process continues in real-time, constantly adapting to your camera feed

**Visualization:**
- **Left pane**: Original camera frame (downsampled to 64x64)
- **Middle pane**: Sparse input showing only 1% of pixels
- **Right pane**: NCA reconstruction with live loss display

The system demonstrates how Neural Cellular Automata can perform real-time learning and reconstruction from extremely limited visual information, simulating scotopic (low-light) vision conditions where only sparse photons are available. 