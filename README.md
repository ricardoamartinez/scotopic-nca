# Scotopic Neural Cellular Automata

This project implements a Neural Cellular Automata (NCA) for reconstructing video frames from extremely sparse pixel data (scotopic vision), based on the demo by Aman Bhargava.

## Project Structure

- `scotopic_nca/model.py`: Defines the `ScotopicNCA` PyTorch model.
- `scotopic_nca/data.py`: Utilities for generating synthetic video data and sparse masks.
- `scotopic_nca/training.py`: The main training script for the baseline model.
- `tests/`: Contains `pytest` tests for verification.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Training:**
    ```bash
    python -m scotopic_nca.training
    ```

3.  **Run Tests:**
    ```bash
    pytest
    ``` 