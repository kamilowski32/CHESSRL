import chess
from env.encoding import board_to_tensor
from config import (
    STEP_REWARD,
    WIN_REWARD,
    LOSS_REWARD,
    DRAW_REWARD,
    TIMEOUT_REWARD,
    MAX_MOVES,
)


class EnvPoolLeague:
    """
    Environment for league / pool-based self-play.

    - środowisko NIE zna agentów
    - obsługuje dwóch graczy (white / black)
    - reward liczony z perspektywy gracza,
      który wykonał OSTATNI ruch
    """

    def __init__(self):
        self.board = chess.Board()
        self.move_count = 0

    @property
    def turn(self):
        """0 = WHITE, 1 = BLACK"""
        return 0 if self.board.turn == chess.WHITE else 1

    def reset(self):
        self.board.reset()
        self.move_count = 0
        return board_to_tensor(self.board)

    def step(self, move):
        """
        Wykonuje ruch przekazany z zewnątrz.

        Args:
            move (chess.Move): legalny ruch

        Returns:
            next_state
            reward
            done
            info (dict)
        """

        self.board.push(move)
        self.move_count += 1

        done = self.board.is_game_over() or self.move_count >= MAX_MOVES

        reward = STEP_REWARD

        if done:
            if self.board.is_game_over():
                result = self.board.result()

                # board.turn jest już po zmianie koloru
                if result == "1-0":
                    reward = WIN_REWARD if self.board.turn == chess.BLACK else LOSS_REWARD
                elif result == "0-1":
                    reward = WIN_REWARD if self.board.turn == chess.WHITE else LOSS_REWARD
                else:
                    reward = DRAW_REWARD
            else:
                reward = TIMEOUT_REWARD

        next_state = board_to_tensor(self.board)
        info = {
            "result": self.board.result() if self.board.is_game_over() else None,
            "moves": self.move_count,
        }

        return next_state, reward, done, info
