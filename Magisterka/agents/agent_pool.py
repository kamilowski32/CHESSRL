import random

from agents.base_agent import BaseAgent

class AgentPool:
    def __init__(self, max_size=10, anchor_prob=0.35):
        self.anchors = []
        self.dynamic = []
        self.max_size = max_size
        self.anchor_prob = anchor_prob

    # --------------------------------------------------
    # Anchors
    # --------------------------------------------------

    def add_anchor(self, agent):
        self.anchors.append(agent)

    # --------------------------------------------------
    # Dynamic snapshots
    # --------------------------------------------------

    def add(self, agent):
        if isinstance(agent, BaseAgent):
            snapshot = agent.clone(freeze=True)
            self.dynamic.append(snapshot)
        else:
            self.dynamic.append(agent)

        if len(self.dynamic) > self.max_size:
            self.dynamic.pop(0)

    def extend(self, agents):
        for a in agents:
            self.dynamic.append(a)
        self.dynamic = self.dynamic[-self.max_size:]

    # --------------------------------------------------
    # Sampling
    # --------------------------------------------------

    def sample(self):
        if self.anchors and self.dynamic:
            if random.random() < self.anchor_prob:
                return random.choice(self.anchors)
            else:
                return random.choice(self.dynamic)

        # fallbacki
        if self.anchors:
            return random.choice(self.anchors)
        if self.dynamic:
            return random.choice(self.dynamic)

        return None

    def __len__(self):
        return len(self.anchors) + len(self.dynamic)