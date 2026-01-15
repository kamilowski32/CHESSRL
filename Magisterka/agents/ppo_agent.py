import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.base_agent import BaseAgent
from config import (
    GAMMA,
    PPO_GAE_LAMBDA,
    PPO_LR,
    PPO_CLIP_EPS,
    PPO_EPOCHS,
    PPO_VALUE_COEF,
    PPO_ENTROPY_BETA,
)
from env.action_encoding import index_to_move
from env.action_mask import legal_moves_mask


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO-Clip)
    On-policy, actor-critic with action masking.
    """

    def __init__(self, model, device):
        super().__init__(model, device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=PPO_LR)
        self.name = "PPO"
        self.reset_buffer()
        self.last_entropy = None


    # ------------------------------------------------------------------
    # Buffer
    # ------------------------------------------------------------------

    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.masks = []   # ← KLUCZOWE

    def reset_episode(self):
        self.reset_buffer()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_move(self, board):
        state = self._state_tensor(board)  # [1, obs_dim]

        logits, value = self.model(state)

        # --- action mask ---
        mask = legal_moves_mask(board)
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device)

        logits = logits.clone()
        logits[:, ~mask_tensor] = -1e9

        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if self.training:
            action = dist.sample()

            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(dist.log_prob(action))
            self.values.append(value.squeeze(0))
            self.masks.append(mask_tensor)
        else:
            action = torch.argmax(probs, dim=-1)

        return index_to_move(action.item())

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def store_reward(self, reward):
        self.rewards.append(reward)

    # ------------------------------------------------------------------
    # PPO Update
    # ------------------------------------------------------------------

    def update(self):
        if len(self.rewards) == 0:
            self.reset_buffer()
            return None

        # 1. Przygotowanie danych (konwersja na tensory)
        values = torch.stack(self.values).squeeze(-1).detach()  # V(s)
        # Zakładamy, że update jest na koniec epizodu, więc next_value dla ostatniego kroku to 0
        # Jeśli trenujesz n-krokowo (nie całe epizody), musiałbyś tu przekazać next_value z sieci
        next_value = 0.0 
        
        # 2. Obliczanie GAE
        gae = 0
        returns = []
        advantages = []

        # Iterujemy od tyłu
        for step in reversed(range(len(self.rewards))):
            # V(s_{t+1})
            next_val = values[step + 1] if step + 1 < len(values) else next_value
            
            # Delta (błąd TD): r + gamma * V(s') - V(s)
            delta = self.rewards[step] + GAMMA * next_val - values[step]
            
            # GAE: delta + gamma * lambda * gae_prev
            gae = delta + GAMMA * PPO_GAE_LAMBDA * gae
            
            # Zbieramy wyniki (insert(0) bo idziemy od tyłu)
            advantages.insert(0, gae)
            
            # Return = Advantage + Value (to jest cel dla Value Head)
            returns.insert(0, gae + values[step])

        # Konwersja na tensory
        advantages = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)

        # Normalizacja Advantages (bardzo ważne dla stabilności PPO!)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- batch tensors ---
        states = torch.cat(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs).detach()
        masks = torch.stack(self.masks)

        total_loss = 0.0

        for _ in range(PPO_EPOCHS):
            logits, new_values = self.model(states)

            # 🔥 Maskowanie (tak jak w oryginale)
            logits = logits.clone()
            logits[~masks] = -1e9

            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)

            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            clipped_ratio = torch.clamp(
                ratio,
                1.0 - PPO_CLIP_EPS,
                1.0 + PPO_CLIP_EPS
            )

            # Policy Loss używa Advantages z GAE
            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()

            # Value Loss celuje w Returns (V + GAE)
            value_loss = F.mse_loss(
                new_values.squeeze(-1),
                returns
            )

            entropy_loss = dist.entropy().mean()
            self.last_entropy = entropy_loss.item()

            loss = (
                policy_loss
                + PPO_VALUE_COEF * value_loss
                - PPO_ENTROPY_BETA * entropy_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        self.reset_buffer()
        return total_loss / PPO_EPOCHS
