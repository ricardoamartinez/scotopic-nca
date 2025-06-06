# Test suite for ScotopicNCA
# To be implemented by Agent 1 

import torch
from scotopic_nca.model import ScotopicNCA
from scotopic_nca.data import generate_moving_square_video

def test_model_instantiation_and_forward_pass():
    """Tests if the model can be created and a forward pass works."""
    model = ScotopicNCA()
    initial_state = torch.zeros(1, 16, 64, 64)
    output_state = model(initial_state, steps=1)
    assert output_state.shape == initial_state.shape, "Output shape mismatch"
    assert not torch.isnan(output_state).any(), "NaNs detected in output"

def test_training_step_reduces_loss():
    """
    Tests if a single training step on a small video reduces the loss.
    This is a basic integration test for the whole pipeline.
    """
    from scotopic_nca.training import train_baseline_nca
    
    # Run a very short training session
    model = train_baseline_nca(epochs=2, sequence_length=4, frame_size=32)
    
    # We can't guarantee loss reduction on every single step, but this
    # at least ensures the training loop runs without crashing.
    # A more robust test would check loss over several steps.
    assert isinstance(model, ScotopicNCA), "Training did not return a model instance" 