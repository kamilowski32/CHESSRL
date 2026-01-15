import numpy as np
from env.action_encoding import move_to_index
from config import ACTION_SPACE

def legal_moves_mask(board) -> np.ndarray:
    """
    Zwraca maskę bool o długości ACTION_SPACE
    """
    mask = np.zeros(ACTION_SPACE, dtype=np.bool_)

    for move in board.legal_moves:
        idx = move_to_index(move)
        mask[idx] = True

    return mask
