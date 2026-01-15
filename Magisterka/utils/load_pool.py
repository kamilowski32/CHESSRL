import torch
from pathlib import Path

def load_pool_from_checkpoints(agent_class, config, checkpoint_dir):
    agents = []

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return agents

    for ckpt in sorted(checkpoint_dir.glob("*.pth")):
        agent = agent_class(config)
        data = torch.load(ckpt, map_location="cpu")
        agent.model.load_state_dict(data["model_state"])
        agent.model.eval()
        for p in agent.model.parameters():
            p.requires_grad = False
        agents.append(agent)

    return agents
