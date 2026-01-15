import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.base_agent import BaseAgent
from env.action_mask import legal_moves_mask
from env.action_encoding import index_to_move
from config import GAMMA, PG_LR


class PGAgent(BaseAgent):
    """
    Policy Gradient (REINFORCE with baseline)
    - correct action masking
    - on-policy
    """

    def __init__(self, model, device):
        super().__init__(model, device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=PG_LR)
        self.name = "PG"
        self.reset_buffer()
        self.last_entropy = None


    # ------------------------------------------------------------------
    # Buffer
    # ------------------------------------------------------------------

    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.masks = []

    def reset_episode(self):
        self.reset_buffer()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_move(self, board):
        state = self._state_tensor(board)

        logits, value = self.model(state)

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
    # PG Update
    # ------------------------------------------------------------------

    def update(self):
        if len(self.rewards) == 0:
            self.reset_buffer()
            return None

        # ----- returns -----
        returns = []
        R = 0.0
        for r in reversed(self.rewards):
            R = r + GAMMA * R
            returns.insert(0, R)

        returns = torch.tensor(returns, device=self.device)
        values = torch.stack(self.values).squeeze(-1)
        advantages = returns - values.detach()

        states = torch.cat(self.states)
        actions = torch.stack(self.actions)
        masks = torch.stack(self.masks)

        # 🔥 RECOMPUTE POLICY (CRITICAL)
        logits, new_values = self.model(states)

        logits = logits.clone()
        logits[~masks] = -1e9

        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)

        entropy = dist.entropy().mean()
        self.last_entropy = entropy.item()

        # ----- losses -----
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(new_values.squeeze(-1), returns)

        loss = policy_loss + 0.5 * value_loss

        # ----- optimize -----
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.reset_buffer()
        return loss.item()
