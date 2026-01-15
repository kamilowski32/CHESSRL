import itertools
import csv
import random
import torch
from pathlib import Path

from agents.snapshot_agent import SnapshotAgent
from utils.seed import set_seed as set_global_seed
from play.run_single_game import run_single_game as play_game
from utils.pgn_writer import save_pgn
from models.networks import ChessResNet


# =========================
# KONFIGURACJA
# =========================
ALGORITHMS = ["A2C", "PG", "PPO"]
SEEDS = [42, 404, 67]
GAMES_PER_PAIR = 50

OUTPUT_BASE = Path("tournaments")


# =========================
# ŁADOWANIE AGENTÓW
# =========================
def load_agents(episodes):
    """
    episodes: list[int]  np. [5000] albo [5000, 10000]
    """
    agents = {}

    for episode in episodes:
        for algo in ALGORITHMS:
            for seed in SEEDS:
                # Używamy bieżącego katalogu roboczego
                base_path = Path("checkpoints") # lub Path("experiments") zależnie gdzie zapisujesz
                model_path = base_path / f"agent_{seed}" / algo / "pool" / f"ep_{episode}.pth"

                if not model_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {model_path}")

                checkpoint = torch.load(model_path, map_location="cpu")

                if "model_state" not in checkpoint:
                    raise ValueError(f"Invalid checkpoint format: {model_path}")

                model = ChessResNet()
                model.load_state_dict(checkpoint["model_state"])
                model.eval()

                agent = SnapshotAgent(model, device="cpu")

                name = f"{algo}_{seed}_ep{episode}"
                agents[name] = agent

                print(f"[OK] Loaded {name}")

    return agents


# =========================
# TURNIEJ
# =========================
def run_tournament(episodes, tournament_name):
    agents = load_agents(episodes)

    out_dir = OUTPUT_BASE / tournament_name
    pgn_dir = out_dir / "pgn"

    pgn_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for (name_a, agent_a), (name_b, agent_b) in itertools.combinations(
        agents.items(), 2
    ):
        matchup_dir = pgn_dir / f"{name_a}_vs_{name_b}"
        matchup_dir.mkdir(exist_ok=True)

        for game_idx in range(GAMES_PER_PAIR):

            # ===== LOSOWY SEED PER GRA =====
            game_seed = random.getrandbits(32)
            set_global_seed(game_seed)

            # ===== ZMIANA KOLORÓW =====
            if game_idx < GAMES_PER_PAIR // 2:
                white, black = agent_a, agent_b
                white_name, black_name = name_a, name_b
            else:
                white, black = agent_b, agent_a
                white_name, black_name = name_b, name_a

            # ===== ROZGRYWKA =====
            result, game = play_game(
                white,
                black,
            )

            # ===== ZAPIS PGN =====
            pgn_path = matchup_dir / f"game_{game_idx:02d}.pgn"

            save_pgn(
                game=game,
                path=pgn_path,
                white_name=white_name,
                black_name=black_name,
                extra_headers={
                    "Seed": str(game_seed),
                    "Tournament": tournament_name
                }
            )

            # ===== CSV =====
            results.append({
                "tournament": tournament_name,
                "white": white_name,
                "black": black_name,
                "result": result,
                "seed": game_seed
            })

    # ===== ZAPIS CSV =====
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tournament",
                "white",
                "black",
                "result",
                "seed"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"[OK] Tournament '{tournament_name}' finished → {csv_path}")


# =========================
# MAIN — 3 TURNIEJE
# =========================
if __name__ == "__main__":

    # 1️⃣ tylko 5000
    run_tournament(
        episodes=[50000],
        tournament_name="pool_league_ep50000"
    )

    # 2️⃣ tylko 10000
    run_tournament(
        episodes=[100000],
        tournament_name="pool_league_ep100000"
    )

    # 3️⃣ mieszany 5000 vs 10000
    run_tournament(
        episodes=[50000, 100000],
        tournament_name="pool_league_mixed_50000_vs_100000"
    )
