class FrozenAgentWrapper:
    """
    Wrapper dla zamrożonego przeciwnika (np. snapshot A2C).
    Agent NIE uczy się i zawsze gra deterministycznie.
    """
    def __init__(self, agent):
        self.agent = agent
        self.agent.set_eval_mode()

    def select_move(self, board):
        return self.agent.select_move(board)
