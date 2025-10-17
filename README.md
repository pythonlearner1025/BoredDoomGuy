# Doom Curiosity Playground [WIP] 

## 1. Install & Run 

### 1.1 Clone & assets
```bash
git clone https://github.com/pythonlearner1025/BoredDoomGuy.git
cd BoredDoomGuy
```

Download/locate `Doom1.WAD` (Ultimate Doom). Place it at the repo root 

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
python icm.py
```

### 1.4 LPM run
```bash
source env/bin/activate
python icm_lpm.py  # accepts env overrides like LPM_ITERS, LPM_DRY_RUN
```

WandB logging is enabled by default; set `WANDB_MODE=disabled` if running in restricted environments.

## 2. Environment snapshot

- **Level**: Doom 1, map `E1M1` (Hangar)
- **API**: ViZDoom player mode, 60×80 grayscale stacks (FRAME_SKIP=4, FRAME_STACK=4)
- **Action space**: macro-actions defined in `MACRO_ACTIONS` inside each script

In both icm.py and icm_lpm.py, the environment runs in infinite horizon mode - the episode ending (death) does not terminate the observe/act loop in run_episode. Only reaching the timeout or number of global collected frames exceeding global max_frames triggers an exit from the observe/act loop.

Following idea from Burda et al, death -> respawn is just another state transition. This infinite horizon formulation is aligned with how human players play, and prevents reward hacking behavior like suicide when episode gets too boring. This is possible because the spawn area would be so well explored that the agent won't find it interesting - incentivizing it to explore the beyond. 

## 3. Papers & origin

| Implementation | Reference | Link |
| -------------- | --------- | ---- |
| `icm.py`       |  Large Scale Curiosity Driven Learning | https://arxiv.org/pdf/1808.04355 |
| `icm_lpm.py`   | *Beyond Noisy TVs: Noise-Robust Exploration via Learning Progress Monitoring* | https://arxiv.org/pdf/2509.25438v1 |

## 4. `icm.py` – Intrinsic Curiosity Module (outline for me to fill)

It's Pathak's original Intrinsic Curiosity Module (ICM) formulation but with PPO over normalized advantages via Generalized Advantage Estimation.

The core idea is to define curiosity as error prediction the next state. A forward dynamics model is trained to predict an encoded representation of the next state phi(s_{t+1}) given the current state phi(s_{t}) and the action a_{t}. The encoder phi is aligned to represent features of states that are only influeancable by the agent's actions (since there are too many features in a complex environment). This is done by defining an inverse dynamics model IDM that takes as input phi(s_{t}, s_{t+1}) and outputs prediction of the action at time t, a_{hat}_{t}. The loss mse(a_{hat}_{t}, a_{t}) then backprops to the encoder model phi. 

### 4.1 Implementation pointers
TODO CODEX: rewrite the above paragraph in section 4 using real code in icm.py. 

## 5. `icm_lpm.py` – Learning Progress Monitoring 

Learning progress monitoring differes from Pathak's ICM by defining rwd_i = (expected_error_t - error_t), where error_t is the mse(phi_obs_next_t, dynamics_model.forward(phi_obs_t, act_t)). expected_error_t is predicted by an error prediction model where expected_error_t = error_model.forward(phi_obs_t, act_t). The error model is trained over a buffer of pairs of (phi_obs_t, act_t, error_t) from outer loop iteration T, and evaluated in rollout at iteration T+1. To understand what the error difference means, consdier a state obs_t which yields high entropy from both models due to pure noise. In this case expected_error_t ~= error_t, because both the error_model or dynamics_model can't compress noise. Thus the rwd_i is low and the model doesn't get stuck in the Noisy TV problem. Contrarily if obs_t yields high entropy but this is due to epistemic uncertainty, reward will be positive as long as forward dynamics model leads in learning the epistemic state transition knowledge, until both error model and forward dynamics model reach a similar understanding and reward is again ~= zero.     

### 5.2 Code landmarks
```python
dynamics_model = LPMDynamicsModel(n_actions).to(device)  # icm_lpm.py:781-784
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
- Threading: defaults to `cpu_count() - 2`, override with `LPM_THREADS` (LPM script) or edit constants in `icm.py`
- Checkpoint cadence: every 25 iterations (ICM) and mirrored in LPM script
- Debug frame dumps land in `debug/` (ignored by git)

## 7. TODOs for future revision

- [ ] Flesh out conceptual sections above in my own words.
- [ ] Add diagrams / reward curves once experiments stabilise.
- [ ] Document best-known hyperparameter tweaks for Doom E1M1.
- [ ] Summarise differences between `icm.py` and `icm_lpm.py` once confident.
