# Scotopic Neural Cellular Automata - Live Demo

This project implements a Neural Cellular Automata (NCA) for reconstructing a live camera feed from extremely sparse pixel data (scotopic vision), based on the demo by Aman Bhargava.

The system first trains a model on synthetic data and then uses the trained model to perform real-time inference on a webcam feed.

## How to Run

### Step 1: Train the Model

First, you need to train the baseline NCA model. This will generate a `scotopic_nca_baseline.pth` file containing the model weights.

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python -m scotopic_nca.training
```

### Step 2: Run Live Inference

Once the model is trained, run the live inference script. This will open your webcam and display the real-time reconstruction.

```bash
python live_inference.py
```
Press 'q' in the display window to quit.

### Running Tests

To verify the core components, you can run the test suite:
```bash
pytest
``` 