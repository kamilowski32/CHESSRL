import chess
import math

class MiniMaxAgent:
    def __init__(self, depth=1, name="MiniMax"):
        self.depth = depth
        self.name = f"{name}_depth{depth}"

    def select_move(self, state):
        best_score = -math.inf
        best_move = None

        for move in state.legal_moves:
            state.push(move)
            score = -self.minimax(state, self.depth - 1)
            state.pop()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def minimax(self, board, depth):
        if depth == 0 or board.is_game_over():
            return self.evaluate(board)

        best = -math.inf
        for move in board.legal_moves:
            board.push(move)
            score = -self.minimax(board, depth - 1)
            board.pop()
            best = max(best, score)
        return best

    def evaluate(self, board):
        if board.is_checkmate():
            return -1000
        if board.is_stalemate():
            return 0

        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }

        score = 0
        for piece, value in piece_values.items():
            score += len(board.pieces(piece, chess.WHITE)) * value
            score -= len(board.pieces(piece, chess.BLACK)) * value

        return score
