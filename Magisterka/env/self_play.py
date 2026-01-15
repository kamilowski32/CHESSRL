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


class SelfPlayEnv:
    """
    Single-agent self-play environment (OPCJA A).

    Agent gra sam ze sobą, na zmianę kolorami.
    Reward zawsze liczony z perspektywy gracza,
    który wykonał OSTATNI ruch.
    """

    def __init__(self, agent):
        self.agent = agent
        self.board = chess.Board()
        self.move_count = 0

    def reset(self):
        self.board.reset()
        self.move_count = 0
        return board_to_tensor(self.board)

    def step(self):
        """
        Wykonuje jeden ruch (jedną decyzję agenta).

        Zwraca:
        - next_state
        - reward
        - done
        """
        state = board_to_tensor(self.board)

        # Agent wybiera legalny ruch
        move = self.agent.select_move(self.board)
        self.board.push(move)
        self.move_count += 1

        done = self.board.is_game_over() or self.move_count >= MAX_MOVES

        # ===== Reward =====
        reward = STEP_REWARD

        if done:
            if self.board.is_game_over():
                result = self.board.result()

                # UWAGA: board.turn zmienił się PO ruchu
                # więc zwycięzca to PRZECIWNY kolor
                if result == "1-0":
                    reward = WIN_REWARD if self.board.turn == chess.BLACK else LOSS_REWARD
                elif result == "0-1":
                    reward = WIN_REWARD if self.board.turn == chess.WHITE else LOSS_REWARD
                else:
                    reward = DRAW_REWARD
            else:
                reward = TIMEOUT_REWARD

        next_state = board_to_tensor(self.board)
        return next_state, reward, done
