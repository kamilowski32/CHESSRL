import numpy as np
import chess
from config import NUM_PLANES

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Zwraca tensor (18, 8, 8) typu float32
    """

    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    # ===== FIGURY =====
    for square, piece in board.piece_map().items():
        row = 7 - square // 8
        col = square % 8
        offset = 0 if piece.color == chess.WHITE else 6
        planes[offset + piece_map[piece.piece_type], row, col] = 1.0

    # ===== STRONA NA RUCHU =====
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # ===== ROSZADY =====
    planes[13, :, :] = board.has_kingside_castling_rights(chess.WHITE)
    planes[14, :, :] = board.has_queenside_castling_rights(chess.WHITE)
    planes[15, :, :] = board.has_kingside_castling_rights(chess.BLACK)
    planes[16, :, :] = board.has_queenside_castling_rights(chess.BLACK)

    # ===== EN PASSANT =====
    if board.ep_square is not None:
        r = 7 - board.ep_square // 8
        c = board.ep_square % 8
        planes[17, r, c] = 1.0

    return planes
