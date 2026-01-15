import chess
from config import PROMOTION_OPTIONS

PROMOTIONS = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

def move_to_index(move: chess.Move) -> int:
    """
    Zamienia ruch szachowy na indeks akcji
    """
    from_sq = move.from_square
    to_sq = move.to_square
    promo = PROMOTIONS.index(move.promotion)
    return from_sq * 64 * PROMOTION_OPTIONS + to_sq * PROMOTION_OPTIONS + promo

def index_to_move(index: int) -> chess.Move:
    """
    Zamienia indeks akcji na ruch szachowy
    """
    from_sq = index // (64 * PROMOTION_OPTIONS)
    rest = index % (64 * PROMOTION_OPTIONS)
    to_sq = rest // PROMOTION_OPTIONS
    promo = PROMOTIONS[rest % PROMOTION_OPTIONS]
    return chess.Move(from_sq, to_sq, promotion=promo)
