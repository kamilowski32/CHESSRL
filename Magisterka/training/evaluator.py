import chess
from env.self_play import SelfPlayEnv


class Evaluator:
    """
    Ewaluacja: current agent vs snapshot agent.
    Snapshot jest zamrożony (eval mode).
    """

    def __init__(self, agent, snapshot, games=20):
        self.agent = agent
        self.snapshot = snapshot
        self.games = games

    def evaluate(self):
        wins = 0
        losses = 0
        draws = 0

        for i in range(self.games):
            env = SelfPlayEnv(
                agent=self.agent if i % 2 == 0 else self.snapshot,
                opponent=self.snapshot if i % 2 == 0 else self.agent
            )

            env.reset()
            done = False

            while not done:
                _, _, done = env.step()

            if env.board.is_game_over():
                result = env.board.result()
                if result == "1-0":
                    wins += 1 if i % 2 == 0 else losses + 1
                elif result == "0-1":
                    losses += 1 if i % 2 == 0 else wins + 1
                else:
                    draws += 1
            else:
                draws += 1

        total = self.games
        winrate = wins / total

        return {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "winrate": winrate
        }
