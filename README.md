# Doom IDM Curiosity-Driven RL

Intrinsic Curiosity Module (ICM) implementation for Doom using PPO, based on:
- [Burda et al. 2018 - Large-Scale Study of Curiosity-Driven Learning](https://arxiv.org/abs/1808.04355)
- [Pathak et al. 2017 - Curiosity-driven Exploration](https://arxiv.org/abs/1705.05363)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/pythonlearner1025/BoredDoomGuy.git
cd BoredDoomGuy
```

### 2. Obtain Doom1.WAD

You need the original Doom 1 WAD file. Place it as `Doom1.WAD` in the repository root.

**Option 1: Purchase the game**
- [Steam](https://store.steampowered.com/app/2280/Ultimate_Doom/)
- [GOG](https://www.gog.com/game/the_ultimate_doom)

**Option 2: Use shareware version**
```bash
wget https://distro.ibiblio.org/slitaz/sources/packages/d/doom1.wad
mv doom1.wad Doom1.WAD
```

### 3. Run setup script

```bash
bash scripts/setup.sh
```

This will:
- Install Python 3.12 (if not present)
- Create virtual environment
- Install system dependencies (Boost, SDL2, OpenAL, etc.)
- Install PyTorch with CUDA support
- Build ViZDoom from source

### 4. Train the agent

```bash
source env3.12/bin/activate
python idm.py
```

Monitor training at the wandb URL printed to console.

## Project Structure

```
.
├── idm.py              # Main training script (ICM + PPO)
├── scripts/
│   ├── setup.sh        # Environment setup script
│   └── *.sh            # Other utility scripts
├── test/               # Test scripts
├── Arnold/             # Arnold DQN framework (legacy)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Configuration

Key hyperparameters in `idm.py`:

```python
INTRINSIC_ONLY = True        # Pure curiosity (no extrinsic reward)
USE_INVERSE_DYNAMICS = True  # ICM vs Random Features
IGNORE_DONES = True          # "Death is not the end"
EPOCHS = 15                  # PPO epochs per iteration
FWD_HIDDEN = 1024           # Forward model capacity
```

## Troubleshooting

**"Doom1.WAD not found"**
- Make sure `Doom1.WAD` is in the repo root directory
- Check the filename is exactly `Doom1.WAD` (case-sensitive on Linux)

**"No module named 'vizdoom'"**
- ViZDoom must be built from source for Python 3.12
- Re-run `scripts/setup.sh` to rebuild

**CUDA out of memory**
- Reduce `MINIBATCH_SIZE` in `idm.py`
- Reduce `MAX_ROLLOUT_FRAMES`
- Use fewer worker threads

## References

```bibtex
@article{burda2018largescale,
  title={Large-Scale Study of Curiosity-Driven Learning},
  author={Burda, Yuri and Edwards, Harri and Storkey, Amos and Klimov, Oleg},
  journal={arXiv preprint arXiv:1808.04355},
  year={2018}
}

@inproceedings{pathak2017curiosity,
  title={Curiosity-driven Exploration by Self-supervised Prediction},
  author={Pathak, Deepak and Agrawal, Pulkit and Efros, Alexei A and Darrell, Trevor},
  booktitle={ICML},
  year={2017}
}
```
