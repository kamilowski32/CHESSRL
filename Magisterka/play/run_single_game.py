"""
TU podłączasz swój istniejący kod rozgrywki.
Turniej NICZEGO więcej nie potrzebuje.
"""

import chess
import chess.pgn


def run_single_game(white_agent, black_agent):
    """
    Zwraca:
        result: str  ("1-0", "0-1", "1/2-1/2")
        game: chess.pgn.Game
    """

    board = chess.Board()
    game = chess.pgn.Game()
    node = game

    agents = {
        chess.WHITE: white_agent,
        chess.BLACK: black_agent
    }

    while not board.is_game_over():
        agent = agents[board.turn]
        move = agent.select_move(board)   # <- TO JUŻ MASZ
        board.push(move)
        node = node.add_variation(move)

    game.headers["Result"] = board.result()
    return board.result(), game
