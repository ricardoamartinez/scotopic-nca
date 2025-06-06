"""
Scotopic Neural Cellular Automata package.

This package implements a neural cellular automaton for reconstructing
video frames from extremely sparse pixel data.
"""

from .model import ScotopicNCA
from .data import generate_moving_square_video, create_sparse_mask

__all__ = ['ScotopicNCA', 'generate_moving_square_video', 'create_sparse_mask'] 