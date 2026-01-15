import argparse
import torch
import chess

from config import CHECKPOINT_INTERVAL, DEVICE, MAX_EPISODES, SEED
from utils.seed import set_seed
from utils.logger import Logger
from utils.checkpoints import load_agent_from_checkpoint, load_model

from models.networks import ChessResNet
from agents.pg_agent import PGAgent
from agents.a2c_agent import A2CAgent
from agents.ppo_agent import PPOAgent

from training.league_trainer_with_pool import LeagueTrainerWithPool
from training.trainer import Trainer
from training.league_trainer import LeagueTrainer
from play.human_vs_agent import play_human_vs_agent


def parse_args():
    parser = argparse.ArgumentParser("Chess RL")

    parser.add_argument(
        "--agent",
        required=True,
        choices=["pg", "a2c", "ppo"]
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "league", "human", "league_pool"]
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None
    )

    parser.add_argument(
        "--color",
        default="white",
        choices=["white", "black"]
    )
    
    # --- NOWOŚĆ: Dodajemy argument seed ---
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED, # Domyślnie bierze z config.py
        help="Random seed for reproducibility"
    )

    return parser.parse_args()


def main():

    args = parse_args()
    set_seed(args.seed)

    device = torch.device(DEVICE)
    model = ChessResNet().to(device)

    if args.agent == "pg":
        agent = PGAgent(model, device)
    elif args.agent == "a2c":
        agent = A2CAgent(model, device)
    else:
        agent = PPOAgent(model, device)

    if args.checkpoint is not None:
        from utils.checkpoints import load_agent_from_checkpoint
        agent = load_agent_from_checkpoint(
            agent,
            args.checkpoint,
            device
        )


    # ===== HUMAN =====
    if args.mode == "human":
        agent.set_eval_mode()
        human_color = chess.WHITE if args.color == "white" else chess.BLACK
        play_human_vs_agent(agent, human_color)
        return

    logger = Logger(f"runs/{args.agent}_{args.mode}")

    # ===== BASELINE SELF-PLAY =====
    if args.mode == "train":
        trainer = Trainer(
            agent=agent,
            logger=logger,
            max_episodes=MAX_EPISODES
        )
        trainer.train()
        return

    # ===== LEAGUE =====
    if args.mode == "league":
        league_trainer = LeagueTrainer(
            agent=agent,
            logger=logger,
            max_episodes=MAX_EPISODES,  # np. 5k
            eval_interval=CHECKPOINT_INTERVAL,
            eval_games=20,
            verbose = False
        )
        league_trainer.train()
        return
    
    # ===== LEAGUE WITH POOL =====
    if args.mode == "league_pool":
        league_trainer = LeagueTrainerWithPool(
            agent=agent,
            logger=logger,
            max_episodes=MAX_EPISODES,  # np. 5k
            eval_interval=CHECKPOINT_INTERVAL,
            eval_games=20,
            verbose = False,
            seed = args.seed
        )
        league_trainer.train()
        return


if __name__ == "__main__":
    main()
