# config.py

# =========================
# OGÓLNE
# =========================

import torch


SEED = 67
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# TRENING
# =========================

MAX_EPISODES = 100000
CHECKPOINT_INTERVAL = 500
GAMMA = 0.99

MAX_MOVES = 500

# =========================
# NAGRODY (ZBALANSOWANE)
# =========================

# Mała kara za istnienie - wymusza szukanie najkrótszej drogi,
# ale nie przesłania nagrody za wygraną.
STEP_REWARD = -0.0005

# Wartości nagród
WIN_REWARD     = +1.0
DRAW_REWARD    = -0.1
LOSS_REWARD    = -1.0
TIMEOUT_REWARD = -1.5


# =========================
# PRZESTRZEŃ AKCJI
# =========================

BOARD_SIZE = 8
NUM_PIECE_PLANES = 12
NUM_EXTRA_PLANES = 6
NUM_PLANES = NUM_PIECE_PLANES + NUM_EXTRA_PLANES

PROMOTION_OPTIONS = 5
ACTION_SPACE = 64 * 64 * PROMOTION_OPTIONS

# =========================
# CNN
# =========================

CNN_CHANNELS = [64, 64, 128]
FC_SIZE = 512

# =========================
# A2C
# =========================

A2C_LR = 3e-4
A2C_ENTROPY_BETA = 0.01
A2C_GAE_LAMBDA = 0.95

# =========================
# POLICY GRADIENT
# =========================

PG_LR = 3e-4

# =========================
# PPO
# =========================

PPO_LR = 3e-4
PPO_CLIP_EPS = 0.2
PPO_EPOCHS = 10
PPO_VALUE_COEF = 0.5
PPO_ENTROPY_BETA = 0.01
PPO_GAE_LAMBDA = 0.95


POOL_SIZE = 8
SNAPSHOT_INTERVAL = 500

EVAL_INTERVAL = 1000
EVAL_GAMES = 20
BEST_SCORE_THRESHOLD = 0.55

LOAD_EXISTING_POOL = False

# =========================
PGN_INTERVAL = 250