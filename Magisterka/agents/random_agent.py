import random

class RandomAgent:
    def __init__(self, name="Random"):
        self.name = name
        self.training = False  # dla kompatybilności

    def select_move(self, board):
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves)

    def reset_episode(self):
        pass

    def set_eval_mode(self):
        pass
