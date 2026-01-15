import chess.pgn

def evaluate(agent, opponents, env, n_games, return_games=False):


    wins = draws = losses = 0
    results = []
    games = []

    agent.set_eval_mode()

    for _ in range(n_games):
        env.reset()
        done = False

        opponent = opponents.sample()
        opponent.set_eval_mode() if hasattr(opponent, "set_eval_mode") else None

        game = chess.pgn.Game()
        game.setup(env.board)
        node = game

        while not done:
            actor = agent if env.turn == 0 else opponent
            move = actor.select_move(env.board)
            _, _, done, _ = env.step(move)
            node = node.add_variation(move)

        result = env.board.result()

        if result == "1-0":
            wins += 1
            results.append(1.0)
        elif result == "0-1":
            losses += 1
            results.append(0.0)
        else:
            draws += 1
            results.append(0.5)

        if return_games:
            game.headers["Result"] = result
            games.append(game)

    score = (wins + 0.5 * draws) / n_games

    agent.set_train_mode()

    if return_games:
        return score, wins, draws, losses, results, games
    else:
        return score, wins, draws, losses, results