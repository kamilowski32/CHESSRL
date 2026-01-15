from env.self_play import SelfPlayEnv
from collections import defaultdict

class Tournament:
    def __init__(self, agents, games_per_pair=100):
        self.agents = agents
        self.games_per_pair = games_per_pair
        self.results = defaultdict(lambda: {"win": 0, "draw": 0, "loss": 0})

    def play_game(self, white, black):
        env = SelfPlayEnv(white, black)
        env.reset()
        done = False

        while not done:
            _, _, done = env.step()

        return env.board.result()

    def run(self):
        names = list(self.agents.keys())

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]

                for g in range(self.games_per_pair):
                    if g < self.games_per_pair // 2:
                        white, black = self.agents[a], self.agents[b]
                        w_name, b_name = a, b
                    else:
                        white, black = self.agents[b], self.agents[a]
                        w_name, b_name = b, a

                    result = self.play_game(white, black)

                    if result == "1-0":
                        self.results[w_name]["win"] += 1
                        self.results[b_name]["loss"] += 1
                    elif result == "0-1":
                        self.results[b_name]["win"] += 1
                        self.results[w_name]["loss"] += 1
                    else:
                        self.results[w_name]["draw"] += 1
                        self.results[b_name]["draw"] += 1

        return self.results
