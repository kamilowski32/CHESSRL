from abc import ABC, abstractmethod
import copy
import torch
from env.encoding import board_to_tensor


class BaseAgent(ABC):
    """
    Formalny kontrakt agenta RL (Opcja A).

    Agent:
    - zna chess.Board
    - wybiera legalny ruch
    - zapisuje reward za WŁASNE decyzje
    """

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.training = True

    @abstractmethod
    def select_move(self, board):
        """Zwraca chess.Move"""
        pass

    @abstractmethod
    def store_reward(self, reward: float):
        pass

    @abstractmethod
    def update(self):
        pass

    def reset_episode(self):
        """Czyści rollout (domyślnie no-op, nadpisywane w agentach)"""
        pass

    def set_train_mode(self):
        self.training = True
        self.model.train()

    def set_eval_mode(self):
        self.training = False
        self.model.eval()

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model.load(path)



    def _state_tensor(self, board):
        """
        Zamienia chess.Board na tensor wejściowy sieci.

        Centralizacja tej logiki:
        - eliminuje duplikację kodu
        - gwarantuje spójne wejście dla PG / A2C / PPO
        """
        state_np = board_to_tensor(board)
        state = torch.tensor(
            state_np,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)
        return state
    
    def clone(self, freeze=True):
        """
        Create a frozen copy of the agent (for pool / league).
        """
        new_agent = copy.deepcopy(self)

        new_agent.model.eval()
        new_agent.training = False

        if freeze:
            for p in new_agent.model.parameters():
                p.requires_grad = False

        return new_agent