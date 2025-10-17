# Doom Curiosity Playground (notes-in-progress)

> ⚠️ This README is intentionally incomplete. I will come back and fill in the conceptual and algorithmic explanations after I’ve internalised both papers.

## 1. Install & Run (tedious bits handled here)

### 1.1 Clone & assets
```bash
git clone https://github.com/pythonlearner1025/BoredDoomGuy.git
cd BoredDoomGuy
```

Download/locate `Doom1.WAD` (Ultimate Doom). Place it at the repo root next to `idm.py`.

- Steam/GOG purchasers: copy the original WAD and rename to `Doom1.WAD`
- Shareware fallback:
  ```bash
  wget https://distro.ibiblio.org/slitaz/sources/packages/d/doom1.wad
  mv doom1.wad Doom1.WAD
  ```

### 1.2 Python virtualenv
```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch + ViZDoom wheel selections are pinned inside `requirements.txt`. If wheels fail for your platform, rebuild ViZDoom from source (see `scripts/`).

### 1.3 Baseline run (ICM)
```bash
source env/bin/activate
python idm.py
```

### 1.4 LPM run
```bash
source env/bin/activate
python idm_lpm.py  # accepts env overrides like LPM_ITERS, LPM_DRY_RUN
```

WandB logging is enabled by default; set `WANDB_MODE=disabled` if running in restricted environments.

## 2. Environment snapshot

- **Level**: Doom 1, map `E1M1` (Hangar)
- **API**: ViZDoom player mode, 60×80 grayscale stacks (FRAME_SKIP=4, FRAME_STACK=4)
- **Action space**: macro-actions defined in `MACRO_ACTIONS` inside each script

TODO: Describe relevant extrinsic rewards, termination conditions, and known quirks (e.g., teleports, enemies, doors).

## 3. Papers & origin

| Implementation | Reference | Link |
| -------------- | --------- | ---- |
| `idm.py`       | Pathak et al., Intrinsic Curiosity Module | https://arxiv.org/pdf/1705.05363 |
| `idm_lpm.py`   | *Beyond Noisy TVs: Noise-Robust Exploration via Learning Progress Monitoring* | https://arxiv.org/pdf/2509.25438v1 |

I will summarise the contributions from each once I finish re-reading them carefully.

## 4. `idm.py` – Intrinsic Curiosity Module (outline for me to fill)

### 4.1 Core idea (placeholder)
- TODO: explain inverse dynamics vs forward model loss trade-off.
- TODO: discuss intrinsic reward scaling and running stats.

### 4.2 Implementation pointers
```python
# Encoder & policy/value heads
agent = Agent(n_actions).to(device)               # idm.py:120-ish

# Curiosity models
phi_enc = TrainableEncoder(...); fwd_model = ForwardDynamicsModel(...)
idm_model = InverseDynamicsModel(...)
```

### 4.3 What I still need to reason about
- [ ] How PPO minibatching interacts with feature learning.
- [ ] Effect of `IGNORE_DONES=True` in Doom (think about lava pits / death resets).
- [ ] Calibration of intrinsic vs extrinsic weighting in this specific level.

## 5. `idm_lpm.py` – Learning Progress Monitoring (outline for me to fill)

### 5.1 Core idea (placeholder)
- TODO: articulate the dual-network setup and why expected past error matters.
- TODO: capture how log-MSE stabilises the intrinsic signal.

### 5.2 Code landmarks
```python
dynamics_model = LPMDynamicsModel(n_actions).to(device)  # idm_lpm.py:781-784
error_model = LPMErrorModel(n_actions).to(device)

# Intrinsic reward inside rollout:
intrinsic_reward = (expected_error - epsilon) if error_ready else 0.0
```

### 5.3 Handling the noisy TV claim
- TODO: Summarise the noisy-TV argument from the paper (expected error vs single-sample).
- TODO: Contrast with raw prediction error approaches.
- TODO: Note any implementation gotchas (buffers, warm-up, log-space).

### 5.4 My empirical notes (to be completed after experiments)

- **Does LPM actually suppress noisy-TV artefacts in Doom E1M1?**  
  > _Leave this section blank until I run controlled tests; note down observations, plots, and failure cases here._

## 6. Shared config knobs (for quick lookup)

- `FRAME_SKIP`, `FRAME_STACK`, `MAX_ROLLOUT_FRAMES`, `MAX_FRAMES_PER_EPISODE`
- Threading: defaults to `cpu_count() - 2`, override with `LPM_THREADS` (LPM script) or edit constants in `idm.py`
- Checkpoint cadence: every 25 iterations (ICM) and mirrored in LPM script
- Debug frame dumps land in `debug/` (ignored by git)

## 7. TODOs for future revision

- [ ] Flesh out conceptual sections above in my own words.
- [ ] Add diagrams / reward curves once experiments stabilise.
- [ ] Document best-known hyperparameter tweaks for Doom E1M1.
- [ ] Summarise differences between `idm.py` and `idm_lpm.py` once confident.
