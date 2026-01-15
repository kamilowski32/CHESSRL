from env.pool_league_env import EnvPoolLeague
from training.train_with_pool import train_with_pool

class LeagueTrainerWithPool:
    def __init__(self, agent, logger, max_episodes, eval_interval, eval_games, verbose, seed):
        self.agent = agent
        self.logger = logger
        self.max_episodes = max_episodes
        self.eval_interval = eval_interval
        self.eval_games = eval_games
        self.verbose = verbose
        self.seed = seed

        # ===== SELF-PLAY ENV =====
        self.env = EnvPoolLeague()
        
    def train(self):
        train_with_pool(
            learner=self.agent,
            logger=self.logger,
            env=self.env,
            seed = self.seed
        )
