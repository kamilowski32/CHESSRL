# Chess RL – Self-Play Reinforcement Learning

Projekt pracy magisterskiej:
- PPO
- A2C
- Policy Gradient (REINFORCE)

## Cechy
- Bitboard 18x8x8 (AlphaZero-style)
- CNN backbone
- Maskowanie nielegalnych ruchów
- Self-play
- TensorBoard
- Turniej każdy-z-każdym

## Uruchomienie
```bash
python main.py
tensorboard --logdir runs
