import torch
from pathlib import Path

def save_checkpoint(agent, base_dir, episode, metrics=None):
    agent_dir = Path(base_dir)
    agent_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "episode": episode,
            "model_state": agent.model.state_dict(),
            "metrics": metrics,
        },
        agent_dir / f"ep_{episode}.pth"
    )
