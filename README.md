# SAC Trading Agent

A Soft Actor-Critic (SAC) reinforcement learning agent for algorithmic trading using CNN-LSTM networks.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Place your data files in `data/` directory:**
   - `trainalt_data.csv` (market data)
   - `deflection_data.csv` (entry points)

3. **Train the agent:**
   ```bash
   python scripts/train.py
   ```

## Project Structure

```
sac-trading-agent/
├── src/                    # Source code
│   ├── agent/             # SAC agent implementation
│   ├── environment/       # Trading environment
│   ├── memory/           # Replay buffer
│   └── utils/            # Utilities
├── scripts/              # Training/testing scripts
├── config/              # Configuration files
├── data/               # Data files (CSV)
├── models/             # Saved model weights
└── logs/              # Training logs
```

## Configuration

Modify `config/training_config.yaml` to adjust hyperparameters.

## Requirements

- Python 3.8+
- PyTorch
- pandas, numpy
- PyYAML

See `requirements.txt` for full dependencies.
