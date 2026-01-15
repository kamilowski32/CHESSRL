import torch
import chess

from agents.base_agent import BaseAgent
from env.action_encoding import move_to_index   # ← TO JEST KLUCZ


class SnapshotAgent(BaseAgent):
    """
    Agent ewaluacyjny (snapshot):
    - dziedziczy po BaseAgent
    - używa wytrenowanej sieci (ChessResNet)
    - realizuje stochastyczną politykę π(a|s)
    - brak uczenia, brak eksploracji
    """

    def __init__(self, model, device="cpu"):
        super().__init__(model, device)
        self.set_eval_mode()   # model.eval(), training=False

    @torch.no_grad()
    def select_move(self, board: chess.Board) -> chess.Move:
        """
        Wybór ruchu zgodnie z policy head sieci.
        W PEŁNI zgodny z treningiem.
        """

        # ===== 1. Stan → tensor (TA SAMA funkcja co w treningu) =====
        state = self._state_tensor(board)   # (1, C, H, W)

        # ===== 2. Forward pass =====
        policy_logits, _ = self.model(state)
        policy_logits = policy_logits.squeeze(0)

        # ===== 3. Softmax → rozkład prawdopodobieństwa =====
        probs = torch.softmax(policy_logits, dim=0)

        # ===== 4. Maskowanie nielegalnych ruchów =====
        legal_moves = list(board.legal_moves)
        legal_indices = [move_to_index(m) for m in legal_moves]

        legal_probs = probs[legal_indices]

        # zabezpieczenie numeryczne (rzadkie, ale poprawne)
        if legal_probs.sum().item() == 0:
            return legal_moves[0]

        legal_probs = legal_probs / legal_probs.sum()

        # ===== 5. Sampling zgodnie z polityką =====
        move_idx = torch.multinomial(legal_probs, 1).item()
        return legal_moves[move_idx]

    # ==========================================================
    # NO-OP (SnapshotAgent się nie uczy)
    # ==========================================================

    def store_reward(self, reward: float):
        pass

    def update(self):
        pass
