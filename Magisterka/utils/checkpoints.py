import torch


def save_model(agent, path, episode=None):
    """
    Save model and optimizer state.

    Parameters:
    - agent: RL agent (A2C, PG, DQN)
    - path: checkpoint path
    - episode: optional training episode number
    """
    checkpoint = {
        "model_state_dict": agent.model.state_dict()
    }

    # Optimizer is optional (e.g. frozen agents)
    if hasattr(agent, "optimizer") and agent.optimizer is not None:
        checkpoint["optimizer_state_dict"] = agent.optimizer.state_dict()

    if episode is not None:
        checkpoint["episode"] = episode

    torch.save(checkpoint, path)



def load_agent_from_checkpoint(agent, path, device):
    """
    Uniwersalny loader checkpointów:
    - train
    - league
    - league_pool
    - human

    Obsługiwane formaty:
    - {"agent": agent}
    - {"model_state_dict": ...}
    - {"model_state": ...}
    - raw state_dict
    """

    checkpoint = torch.load(path, map_location=device)

    # ===== CASE 1: zapisany cały agent =====
    if isinstance(checkpoint, dict) and "agent" in checkpoint:
        agent = checkpoint["agent"]
        agent.model.to(device)
        agent.set_eval_mode()
        return agent

    # ===== CASE 2: standardowy PyTorch =====
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        agent.model.to(device)
        agent.set_eval_mode()
        return agent

    # ===== CASE 3: TWÓJ FORMAT (pool / best) =====
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        agent.model.load_state_dict(checkpoint["model_state"])
        agent.model.to(device)
        agent.set_eval_mode()
        return agent

    # ===== CASE 4: zapisany sam state_dict =====
    if isinstance(checkpoint, dict):
        try:
            agent.model.load_state_dict(checkpoint)
            agent.model.to(device)
            agent.set_eval_mode()
            return agent
        except Exception:
            pass

    raise ValueError(
        f"Nieznany format checkpointa: {type(checkpoint)} | keys={getattr(checkpoint, 'keys', lambda: 'N/A')()}"
    )


def load_model(agent, path, device, load_optimizer=True):
    """
    Load model (and optionally optimizer) state.

    Parameters:
    - agent: RL agent instance
    - path: checkpoint path
    - device: torch device
    - load_optimizer: whether to restore optimizer state

    Returns:
    - agent
    - episode (or None)
    """
    checkpoint = torch.load(path, map_location=device)

    agent.model.load_state_dict(checkpoint["model_state_dict"])

    if (
        load_optimizer
        and "optimizer_state_dict" in checkpoint
        and hasattr(agent, "optimizer")
        and agent.optimizer is not None
    ):
        agent.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    episode = checkpoint.get("episode", None)

    agent.set_eval_mode()
    return agent, episode
