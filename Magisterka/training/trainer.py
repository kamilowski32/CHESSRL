import os
import csv
from typing import Optional

from env.self_play import SelfPlayEnv
from utils.checkpoints import save_model
from config import CHECKPOINT_INTERVAL


class Trainer:
    """
    Single-agent self-play Trainer (OPCJA A).

    Agent gra sam ze sobą, na zmianę kolorami.
    Reward zawsze liczony z perspektywy gracza,
    który wykonał decyzję (ruch).

    Odpowiedzialności (LOGICZNIE ROZDZIELONE):
    1. Pętla treningowa (epizody)
    2. Egzekucja epizodu
    3. Aktualizacja agenta
    4. Logowanie (TensorBoard + CSV)
    5. Checkpointing

    Trainer NIE:
    - nie liczy gradientów
    - nie zna szczegółów algorytmu (PG/A2C/PPO)
    """

    def __init__(
        self,
        agent,
        logger,
        max_episodes: int,
        log_dir: Optional[str] = None
    ):
        self.agent = agent
        self.logger = logger
        self.max_episodes = max_episodes

        # Single-agent self-play environment
        self.env = SelfPlayEnv(agent)

        # ===== LOGI EKSPERYMENTU =====
        self.log_dir = log_dir or os.environ.get(
            "EXPERIMENT_DIR", "runs/default"
        )
        os.makedirs(self.log_dir, exist_ok=True)

        self.csv_path = os.path.join(self.log_dir, "episode_logs.csv")
        self._init_csv()

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------

    def _init_csv(self):
        """
        Inicjalizacja pliku CSV.

        CSV służy do:
        - analizy statystycznej
        - wykresów w pracy
        - porównań PG / A2C / PPO
        """
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            "episode",
            "result",           # white / black / draw / timeout
            "episode_reward",
            "episode_length"
        ])
        self.csv_file.flush()

    # ------------------------------------------------------------------
    # MAIN TRAINING LOOP
    # ------------------------------------------------------------------

    def train(self):
        """
        Główna pętla treningowa.

        Każdy epizod:
        - reset środowiska
        - sekwencja decyzji agenta
        - update modelu PO epizodzie (on-policy)
        """
        os.makedirs("checkpoints", exist_ok=True)

        for episode in range(1, self.max_episodes + 1):
            self.agent.set_train_mode()

            episode_reward, episode_length, result = \
                self._run_single_episode()

            # Aktualizacja modelu (PG / A2C / PPO)
            loss = self.agent.update()

            # WAŻNE: reset rolloutów
            self.agent.reset_episode()

            self._log_metrics(
                episode,
                episode_reward,
                episode_length,
                result,
                loss
            )

            self._maybe_checkpoint(episode)

        self.csv_file.close()

    # ------------------------------------------------------------------
    # EPISODE EXECUTION
    # ------------------------------------------------------------------

    def _run_single_episode(self):
        """
        Wykonuje jeden pełny epizod gry.

        Reward:
        - zawsze z perspektywy gracza,
          który wykonał OSTATNI ruch
        """
        self.env.reset()
        done = False

        episode_reward = 0.0
        move_count = 0

        while not done:
            _, reward, done = self.env.step()

            # Agent zapisuje reward TYLKO za swoje decyzje
            self.agent.store_reward(reward)

            episode_reward += reward
            move_count += 1

        result = self._determine_result()
        return episode_reward, move_count, result

    # ------------------------------------------------------------------
    # RESULT & LOGGING
    # ------------------------------------------------------------------

    def _determine_result(self) -> str:
        """
        Określa wynik partii w sposób analitycznie czytelny.
        """
        board = self.env.board

        if board.is_game_over():
            outcome = board.result()
            if outcome == "1-0":
                return "white"
            elif outcome == "0-1":
                return "black"
            else:
                return "draw"

        return "timeout"

    def _log_metrics(
        self,
        episode: int,
        reward: float,
        length: int,
        result: str,
        loss: Optional[float]
    ):
        """
        Centralne miejsce logowania metryk.
        """
        # ===== TensorBoard =====
        self.logger.log_scalar("Reward/episode", reward, episode)
        self.logger.log_scalar("Episode/length", length, episode)

        if loss is not None:
            self.logger.log_scalar("Loss", loss, episode)

        # ===== CSV =====
        self.csv_writer.writerow([
            episode,
            result,
            reward,
            length
        ])
        self.csv_file.flush()

        # ===== Console =====
        print(
            f"Ep {episode:5d} | "
            f"R {reward:7.2f} | "
            f"L {length:3d} | "
            f"{result}"
        )

    # ------------------------------------------------------------------
    # CHECKPOINTING
    # ------------------------------------------------------------------

    def _maybe_checkpoint(self, episode: int):
        """
        Okresowy zapis modelu.

        Checkpointy wykorzystywane są:
        - w LeagueTrainer
        - w analizie stabilności algorytmów
        """
        if episode % CHECKPOINT_INTERVAL == 0:
            save_model(
                self.agent,
                f"checkpoints/{type(self.agent).__name__}_ep{episode}.pth",
                episode
            )
            print(f"[Checkpoint] Saved at episode {episode}")
