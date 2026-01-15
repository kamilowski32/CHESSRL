def dqn_reward_shaping(raw_reward, done):
    """
    Reward shaping dla DQN:
    - brak rewardu krokowego
    - silny sygnał terminalny
    """
    if not done:
        return 0.0

    if raw_reward > 0:
        return 2.0     # wygrana
    elif raw_reward < 0:
        return -2.0    # przegrana
    else:
        return -0.5    # remis
