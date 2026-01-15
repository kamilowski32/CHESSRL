import os
import subprocess
import time
from datetime import datetime
from multiprocessing import Process

from config import MAX_EPISODES as ME

# Konfiguracja
AGENTS = ["a2c", "pg", "ppo"]
SEEDS = [42, 67, 404]
EPISODES = ME

BASE_DIR = "experiments" # lub "checkpoints" zależnie jak wolisz

def run_league_pool(agent: str, seed: int):
    # Unikalna nazwa procesu dla logów
    process_name = f"{agent.upper()}_{seed}"
    print(f"\n=== START LEAGUE_POOL | {process_name} ===")

    # Struktura katalogów: experiments/agent_42/ppo/league_pool
    # Dzięki temu tournament.py łatwo to znajdzie
    log_dir = os.path.join(BASE_DIR, f"agent_{seed}", agent, "league_pool")
    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        "python", "main.py",
        "--agent", agent,
        "--mode", "league_pool",
        "--seed", str(seed)  # Przekazujemy seed do main.py
    ]

    # Zmienne środowiskowe
    env = os.environ.copy()
    env["EXPERIMENT_DIR"] = log_dir
    env["MAX_EP_OVERRIDE"] = str(EPISODES)
    
    # CRITICAL: Ograniczenie CPU dla pojedynczego procesu
    # Przy 9 procesach (3 agenty * 3 seedy) to konieczne, by system nie zamarł
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"

    # Przekierowanie wyjścia do plików w folderze eksperymentu
    stdout_path = os.path.join(log_dir, "stdout.log")
    stderr_path = os.path.join(log_dir, "stderr.log")

    try:
        with open(stdout_path, "w") as f_out, open(stderr_path, "w") as f_err:
            subprocess.run(cmd, env=env, check=True, stdout=f_out, stderr=f_err)
        print(f"=== FINISHED | {process_name} ===")
    except subprocess.CalledProcessError:
        print(f"!!! ERROR in {process_name} !!! Sprawdź logi w: {stderr_path}")


def main():
    print("===== STARTING PARALLEL LEAGUE_POOL EXPERIMENTS (3 SEEDS) =====")
    print(f"Timestamp: {datetime.now()}")
    print(f"Agents: {AGENTS}")
    print(f"Seeds:  {SEEDS}")
    print(f"Total processes: {len(AGENTS) * len(SEEDS)}")

    processes = []

    # Uruchamiamy wszystko równolegle
    # UWAGA: Jeśli masz słabe CPU/mało RAM, 9 procesów może "zabić" maszynę.
    # Na Google DL VM (np. n1-standard-8 lub lepszym) powinno być OK.
    
    for agent in AGENTS:
        for seed in SEEDS:
            p = Process(
                target=run_league_pool,
                args=(agent, seed),
                daemon=False
            )
            processes.append(p)
            p.start()
            
            # Małe opóźnienie między startami, żeby nie obciążyć dysku/CPU nagłym skokiem
            time.sleep(2)

    # Czekamy na zakończenie wszystkich
    for p in processes:
        p.join()

    print("\n===== ALL EXPERIMENTS FINISHED =====")


if __name__ == "__main__":
    main()