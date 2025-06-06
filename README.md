# Scotopic NCA - Standalone Version

**Live Neural Cellular Automata with Real-time Online Learning**

This is a complete standalone implementation of a Neural Cellular Automata that performs real-time online learning on webcam input, reconstructing full RGB frames from extremely sparse pixel data (scotopic vision simulation).

## Features

- **Single file implementation** - Everything needed is in `live_online_learning.py`
- **Real-time webcam processing** - Live camera feed with 480x480 resolution
- **Online learning** - Model continuously adapts to your camera input
- **Adjustable sparsity** - Control panel to adjust sparse pixel percentage (0.01% - 10%)
- **Stable architecture** - Simple RGB + communication channels, no collapse issues
- **GPU acceleration** - Automatic CUDA support if available

## Requirements

Only standard packages needed:
- `torch` (PyTorch)
- `opencv-python` (cv2)
- `numpy`

## Quick Start

```bash
# Install dependencies
pip install torch opencv-python numpy

# Run the standalone script
python live_online_learning.py
```

## How It Works

1. **Camera Input**: Captures live webcam frames at 480x480 resolution
2. **Sparse Sampling**: Randomly samples a small percentage of pixels (adjustable 0.01%-10%)
3. **NCA Processing**: Neural Cellular Automata reconstructs full RGB frame from sparse input
4. **Online Learning**: Model weights update continuously based on reconstruction error
5. **Real-time Visualization**: Shows original, sparse input, prediction, and communication channels

## Controls

- **Sparsity Control**: Use the "Controls" window trackbar to adjust sparse pixel percentage
- **Quit**: Press 'q' in any window to exit
- **Reset**: Model automatically resets every 1000 steps to prevent drift

## Display Windows

- **Main Window**: 4-panel view (Original | Sparse Input | NCA Prediction | Communication)
- **Controls Window**: Sparsity adjustment trackbar

The system demonstrates how Neural Cellular Automata can perform real-time learning and reconstruction from extremely limited visual information, simulating scotopic (low-light) vision conditions. 