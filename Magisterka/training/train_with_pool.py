from datetime import datetime
from pathlib import Path
from agents.minimax_agent import MiniMaxAgent
import chess.pgn

from agents.agent_pool import AgentPool
from agents.random_agent import RandomAgent
from evaluation.evaluator import evaluate
from evaluation.elo import EloRating
from utils.checkpointing import save_checkpoint
from utils.csv_logger import CSVLogger
from utils.load_pool import load_pool_from_checkpoints

from config import (
    MAX_EPISODES,
    POOL_SIZE,
    SNAPSHOT_INTERVAL,
    EVAL_INTERVAL,
    EVAL_GAMES,
    BEST_SCORE_THRESHOLD,
    LOAD_EXISTING_POOL,
    STEP_REWARD,
    WIN_REWARD,
    LOSS_REWARD,
    DRAW_REWARD,
    TIMEOUT_REWARD,
    PGN_INTERVAL,
)

def minimax_depth_schedule(episode):
    if episode < 3000:
        return 1
    elif episode < 7000:
        return 2
    elif episode < 12000:
        return 3
    elif episode < 30000:
        return 4
    elif episode < 50000:
        return 5
    else:
        return 6


def train_with_pool(learner, logger, env, seed):
    """
    FINAL pool-based league training.

    Reward:
    - terminal-only
    - learner ALWAYS gets reward for game outcome
    - timeout penalty preserved
    """

    agent_name = learner.name

    # ===== DIRECTORIES =====
    checkpoint_dir = Path(f"checkpoints/agent_{seed}/{agent_name}/pool")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_dir = checkpoint_dir / "best"
    best_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    experiment_dir = Path(f"experiments/pooltraining/{agent_name}")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    pgn_base = Path(f"PGNs/{agent_name}_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    pgn_dirs = {
        "wins": pgn_base / "wins",
        "losses": pgn_base / "losses",
        "draws": pgn_base / "draws",
        "eval": pgn_base / "eval",
    }
    for d in pgn_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ===== CSV LOGGERS =====
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_logger = CSVLogger(
        experiment_dir / f"train_{agent_name}_{seed}_{run_id}.csv",
        header = [
            "episode",
            "reward_terminal",
            "reward_total",
            "moves",
            "result",          # win/draw/loss/timeout
            "color",            # White/Black
            "opponent"         # opponent name
        ]

    )

    eval_logger = CSVLogger(
        experiment_dir / f"eval_{agent_name}_{seed}_{run_id}.csv",
        header=["episode", "wins", "draws", "losses", "score", "elo"]
    )

    league_logger = CSVLogger(
        experiment_dir / f"league_{agent_name}_{seed}_{run_id}.csv",
        header=[
            "episode",
            "win_rate",
            "draw_rate",
            "score",
            "pool_size",
            "accepted",
        ]
    )

    entropy_logger = CSVLogger(
        experiment_dir / f"entropy_{agent_name}_{seed}_{run_id}.csv",
        header=["episode", "entropy"]
    )


    # ===== ELO =====
    elo = EloRating(rating=1200)
    opponent_elo = EloRating(rating=1200)

    pool = AgentPool(max_size=POOL_SIZE)

    pool.add_anchor(RandomAgent())
    pool.add_anchor(MiniMaxAgent(depth=1))  # placeholder



    # ===== LOAD SNAPSHOTS =====
    if LOAD_EXISTING_POOL:
        pool.extend(load_pool_from_checkpoints(learner, checkpoint_dir))

    # ===== MAIN LOOP =====
    for episode in range(1, MAX_EPISODES + 1):

        print(f"\n=== Episode {episode} ===")

        new_depth = minimax_depth_schedule(episode)
        current_mm = pool.anchors[1]

        learner.name = f"{agent_name}_ep{episode}"

        if current_mm.depth != new_depth:
            pool.anchors[1] = MiniMaxAgent(depth=new_depth)
            print(f"[ANCHOR] MiniMax depth -> {new_depth}")

        opponent = pool.sample() or RandomAgent()

        if episode % 2 == 0:
            white, black = learner, opponent
        else:
            white, black = opponent, learner

        opponent_name = opponent.name if hasattr(opponent, "name") else "Opponent"

        env.reset()
        learner.reset_episode()

        # ===== PGN GAME (TRAIN) =====
        game = chess.pgn.Game()
        game.setup(env.board)
        node = game

        done = False
        move_count = 0

        while not done:
            actor = white if env.turn == 0 else black
            action = actor.select_move(env.board)

            _, _, done, _ = env.step(action)
            node = node.add_variation(action)
            move_count += 1
            if actor is learner:
                learner.store_reward(STEP_REWARD)

        # ===== TERMINAL REWARD (LEARNER POV) =====
        if env.board.is_game_over():
            result = env.board.result()
            if result == "1-0":
                terminal_reward = WIN_REWARD if learner is white else LOSS_REWARD
                result = "win" if learner is white else "loss"
            elif result == "0-1":
                terminal_reward = WIN_REWARD if learner is black else LOSS_REWARD
                result = "win" if learner is black else "loss"
            else:
                terminal_reward = DRAW_REWARD
                result = "draw"
        else:
            terminal_reward = TIMEOUT_REWARD
            result = "*"

        color = "White" if learner is white else "Black"

        if len(learner.values) > 0:
            learner.rewards[-1] = terminal_reward
            loss = learner.update()
        else:
            learner.reset_episode()
            loss = None

        if hasattr(learner, "last_entropy") and learner.last_entropy is not None:
            entropy_logger.log([episode, learner.last_entropy])


        if logger is not None:
            logger.log_scalar("Episode/length", move_count, episode)
            logger.log_scalar("Reward/episode", terminal_reward, episode)

        total_reward = sum(learner.rewards)

        train_logger.log([
            episode,
            terminal_reward,
            total_reward,
            move_count,
            result,
            color,
            opponent_name
        ])

        # ===== SAVE TRAIN PGN =====
        if episode % PGN_INTERVAL == 0:
            game.headers.update({
                "Event": "Pool Training",
                "Date": datetime.now().strftime("%Y.%m.%d"),
                "Round": str(episode),
                "White": learner.name if learner is white else opponent_name,
                "Black": learner.name if learner is black else opponent_name,
                "Agent": agent_name,
                "seed": str(seed),
                "Result": result,
            })

            folder = (
                "wins" if terminal_reward == WIN_REWARD else
                "losses" if terminal_reward == LOSS_REWARD else
                "draws"
            )

            with open(
                pgn_dirs[folder] / f"{agent_name}_ep{episode}.pgn",
                "w",
                encoding="utf-8"
            ) as f:
                print(game, file=f)

        # ===== SNAPSHOT =====
        if episode % SNAPSHOT_INTERVAL == 0:
            pool.add(learner)
            save_checkpoint(learner, checkpoint_dir, episode)

        # ===== EVALUATION =====
        if episode % EVAL_INTERVAL == 0:

            print("\n--- EVALUATION ---")
            score, w, d, l, results, games = evaluate(
                learner,
                pool,
                env,
                EVAL_GAMES,
                return_games=True
            )

            for r in results:
                elo.update(opponent_elo.rating, r)

            # save ALL eval PGNs
            for i, g in enumerate(games):
                g.headers.update({
                    "Event": "Evaluation",
                    "Agent": agent_name,
                    "seed": str(seed),
                    "Episode": str(episode),
                })
                with open(
                    pgn_dirs["eval"] / f"{agent_name}_eval_ep{episode}_g{i}.pgn",
                    "w",
                    encoding="utf-8"
                ) as f:
                    print(g, file=f)

            accepted = score >= BEST_SCORE_THRESHOLD
            if accepted:
                best_score = score
                save_checkpoint(
                    learner,
                    best_checkpoint_dir,
                    episode,
                    metrics={"score": score, "elo": elo.rating}
                )

            eval_logger.log([
                episode,
                w,
                d,
                l,
                score,
                round(elo.rating, 2)
            ])

            league_logger.log([
                episode,
                w / EVAL_GAMES,
                d / EVAL_GAMES,
                score,
                len(pool),
                accepted
            ])

            # ===== TENSORBOARD: LEAGUE METRICS =====
            if logger is not None:
                logger.log_scalar("League/score", score, episode)
                logger.log_scalar("League/win_rate", w / EVAL_GAMES, episode)
                logger.log_scalar("League/draw_rate", d / EVAL_GAMES, episode)
                logger.log_scalar("League/pool_size", len(pool), episode)


    train_logger.close()
    eval_logger.close()
    league_logger.close()
    entropy_logger.close()
