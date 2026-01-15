import os
import csv
import math
import random
from collections import deque
from typing import Optional

import chess
import torch

from env.self_play import SelfPlayEnv
from utils.checkpoints import save_model
from config import CHECKPOINT_INTERVAL, MAX_MOVES


class LeagueTrainer:
    """
    AlphaZero-lite League Trainer z:
    - pulą historycznych przeciwników
    - ważonym losowaniem (softmax)
    - brakiem rollbacków
    - selekcją najlepszych snapshotów

    Champion:
    - aktualnie trenowany agent (ciągły learning)

    Opponents:
    - zamrożone snapshoty wag (pool)

    League:
    - tylko ocenia i zapisuje najlepsze wersje
    """

    def __init__(
        self,
        agent,
        logger,
        max_episodes: int,
        eval_interval: int = 500,
        eval_games: int = 20,
        pool_size: int = 5,
        temperature: float = 0.5,
        log_dir: Optional[str] = None,
        verbose: bool = True
    ):
        self.agent = agent
        self.logger = logger
        self.max_episodes = max_episodes

        self.eval_interval = eval_interval
        self.eval_games = eval_games
        self.pool_size = pool_size
        self.temperature = temperature

        self.verbose = verbose

        # ===== SELF-PLAY ENV =====
        self.env = SelfPlayEnv(self.agent)

        # ===== OPPONENT POOL =====
        self.opponent_pool = deque(maxlen=self.pool_size)
        self.best_score = 0.0

        # seed pool – losowy agent (epizod 0)
        self.opponent_pool.append({
            "weights": self._clone_weights(self.agent),
            "score": 0.0,
            "episode": 0
        })

        # ===== LOGI =====
        agent_name = type(self.agent).__name__
        base_dir = log_dir or os.environ.get("EXPERIMENT_DIR", "runs")

        self.log_dir = os.path.join(base_dir, f"POOL_{agent_name}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.csv_path = os.path.join(self.log_dir, f"POOL_{agent_name}_league.csv")
        self._init_csv()

    # ------------------------------------------------------------------
    # WEIGHTS
    # ------------------------------------------------------------------

    def _clone_weights(self, agent):
        return {
            k: v.detach().cpu().clone()
            for k, v in agent.model.state_dict().items()
        }

    def _load_weights(self, weights):
        self.agent.model.load_state_dict(weights)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def _init_csv(self):
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "episode",
            "win_rate",
            "draw_rate",
            "score",
            "pool_size",
            "accepted"
        ])
        self.csv_file.flush()

    # ------------------------------------------------------------------
    # OPPONENT SAMPLING (WEIGHTED)
    # ------------------------------------------------------------------

    def _sample_opponent(self):
        scores = [o["score"] for o in self.opponent_pool]

        min_score = min(scores)
        shifted = [s - min_score + 1e-3 for s in scores]

        exp_scores = [
            math.exp(s / self.temperature) for s in shifted
        ]
        total = sum(exp_scores)
        probs = [s / total for s in exp_scores]

        return random.choices(self.opponent_pool, weights=probs, k=1)[0]

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------

    def train(self):
        os.makedirs("checkpoints", exist_ok=True)

        for episode in range(1, self.max_episodes + 1):
            # ===== TRAINING =====
            self.agent.set_train_mode()

            self._run_training_episode()
            loss = self.agent.update()
            self.agent.reset_episode()

            if loss is not None:
                self.logger.log_scalar("Loss", loss, episode)

            # ===== LEAGUE EVALUATION =====
            if episode % self.eval_interval == 0:
                win_rate, draw_rate = self._evaluate_league()
                self.agent.reset_episode()
                
                score = win_rate + 0.5 * draw_rate
                accepted = score > self.best_score + 0.01

                if accepted:
                    self.best_score = score

                    snapshot = {
                        "weights": self._clone_weights(self.agent),
                        "score": score,
                        "episode": episode
                    }
                    self.opponent_pool.append(snapshot)

                    save_model(
                        self.agent,
                        f"checkpoints/POOL_{type(self.agent).__name__}_ep{episode}_score{score:.2f}.pth",
                        episode
                    )

                self._log_league_metrics(
                    episode, win_rate, draw_rate, score, accepted
                )

            # ===== REGULAR CHECKPOINT =====
            if episode % CHECKPOINT_INTERVAL == 0:
                save_model(
                    self.agent,
                    f"checkpoints/POOL_{type(self.agent).__name__}_ep{episode}.pth",
                    episode
                )

        self.csv_file.close()

    # ------------------------------------------------------------------
    # TRAINING EPISODE
    # ------------------------------------------------------------------

    def _run_training_episode(self):
        self.env.reset()
        done = False

        while not done:
            _, reward, done = self.env.step()
            self.agent.store_reward(reward)

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------

    def _evaluate_league(self):
        wins = 0
        draws = 0
        total_games = self.eval_games * 2

        self.agent.set_eval_mode()

        with torch.no_grad():
            for _ in range(self.eval_games):
                opponent = self._sample_opponent()

                w, d = self._play_eval_game(True, opponent["weights"], verbose = self.verbose)
                wins += w
                draws += d

                w, d = self._play_eval_game(False, opponent["weights"], verbose=self.verbose)
                wins += w
                draws += d

        return wins / total_games, draws / total_games

    def _play_eval_game(
        self,
        champion_white: bool,
        opponent_weights,
        verbose: bool = True
    ):
        if verbose:
            print("STARTING LEAGUE GAME")
        board = chess.Board()
        move_count = 0

        while not board.is_game_over() and move_count < MAX_MOVES:
            if board.turn == chess.WHITE:
                if champion_white:
                    move = self.agent.select_move(board)
                else:
                    move = self._select_move_opponent(board, opponent_weights)
            else:
                if champion_white:
                    move = self._select_move_opponent(board, opponent_weights)
                else:
                    move = self.agent.select_move(board)

            if verbose:
                move_san = board.san(move)
                print(f"Move {move_count + 1}: {move_san}")

            board.push(move)
            move_count += 1

        result = board.result()

        if verbose:
            champ_color = "WHITE" if champion_white else "BLACK"
            print(
                f"[LEAGUE GAME] "
                f"Champion {champ_color} | "
                f"Result {result} | "
                f"Moves {move_count}"
            )

        if result == "1-0":
            return (1, 0) if champion_white else (0, 0)
        elif result == "0-1":
            return (0, 0) if champion_white else (1, 0)
        else:
            return (0, 1)


    def _select_move_opponent(self, board, opponent_weights):
        current_weights = self._clone_weights(self.agent)

        self.agent.reset_episode()
        self._load_weights(opponent_weights)
        self.agent.set_eval_mode()

        move = self.agent.select_move(board)

        self._load_weights(current_weights)
        self.agent.reset_episode()

        return move

    # ------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------

    def _log_league_metrics(
        self,
        episode: int,
        win_rate: float,
        draw_rate: float,
        score: float,
        accepted: bool
    ):
        self.logger.log_scalar("League/win_rate", win_rate, episode)
        self.logger.log_scalar("League/draw_rate", draw_rate, episode)
        self.logger.log_scalar("League/score", score, episode)

        self.csv_writer.writerow([
            episode,
            win_rate,
            draw_rate,
            score,
            len(self.opponent_pool),
            accepted
        ])
        self.csv_file.flush()

        status = "ACCEPTED" if accepted else "REJECTED"
        print(
            f"[LEAGUE] Ep {episode:5d} | "
            f"Win {win_rate:.2f} | "
            f"Draw {draw_rate:.2f} | "
            f"Score {score:.2f} | "
            f"Pool {len(self.opponent_pool)} | "
            f"{status}"
        )
