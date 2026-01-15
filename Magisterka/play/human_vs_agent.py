import chess
from env.encoding import board_to_tensor


def print_board(board):
    print(board)
    print()


def get_human_move(board):
    while True:
        try:
            move_uci = input("Your move (UCI, e.g. e2e4): ").strip()
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                return move
            else:
                print("Illegal move, try again.")
        except Exception:
            print("Invalid format, try again.")


def play_human_vs_agent(agent, human_color):
    """
    Human vs agent game.
    Agent plays deterministically (eval mode).
    """

    board = chess.Board()
    agent.set_eval_mode()

    print("Game start!")
    print_board(board)

    while not board.is_game_over():

        if board.turn == human_color:
            move = get_human_move(board)
        else:
            move = agent.select_move(board)
            print(f"Agent move: {move.uci()}")

        board.push(move)
        print_board(board)

    print("Game over:", board.result())
