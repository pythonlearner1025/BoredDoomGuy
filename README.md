# Doom Curiosity Playground [WIP] 

## 1. Install & Run 

### 1.1 Clone & assets
```bash
git clone https://github.com/pythonlearner1025/BoredDoomGuy.git
cd BoredDoomGuy
```

Download/locate `Doom1.WAD` (Ultimate Doom). Place it at the repo root.

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

WandB logging is enabled by default. Set `WANDB_MODE=disabled` when running in restricted environments.

## 2. Environment snapshot

- **Level**: Doom 1, map `E1M1` (Hangar)
- **API**: ViZDoom player mode, 60×80 grayscale stacks (FRAME_SKIP=4, FRAME_STACK=4)
- **Action space**: macro-actions defined in `MACRO_ACTIONS` inside each script

In both `icm.py` and `icm_lpm.py`, the environment runs in infinite-horizon mode. Death does not terminate the observe/act loop in `run_episode`, and only hitting the timeout or `max_frames` stops the rollout.

Following Burda et al., the death-to-respawn transition is treated as just another state change. This mirrors how people play Doom and discourages reward hacking via intentional suicide, because the spawn room becomes boring once the agent has fully explored it.

## 3. Papers & origin

| Implementation | Reference | Link |
| -------------- | --------- | ---- |
| `icm.py`       |  Large Scale Curiosity Driven Learning | https://arxiv.org/pdf/1808.04355 |
| `icm_lpm.py`   | *Beyond Noisy TVs: Noise-Robust Exploration via Learning Progress Monitoring* | https://arxiv.org/pdf/2509.25438v1 |

## 4. `icm.py` – Intrinsic Curiosity Module

Pathak's Intrinsic Curiosity Module trains a forward dynamics predictor alongside PPO with GAE-normalised advantages. Curiosity is the prediction error between encoded next-state features and the forward model's output, so we freeze the policy rollout to collect frame stacks and actions.

During each PPO minibatch we encode the stacked frames, predict the next embedding, and measure the curiosity reward as the mean-squared error. The encoder then feeds an inverse dynamics model that learns to recover the action, keeping features focused on agent-controllable factors.

```python
phi_s = phi_enc(mb_obs_stack)
phi_s_next = phi_enc(mb_obs_next_stack)
phi_pred = fwd_model(phi_s.detach(), act_onehot)

per_sample_fwd = F.mse_loss(phi_pred, phi_s_next.detach(), reduction='none').mean(dim=1)
fwd_loss_val = per_sample_fwd.mean()

phi_s_idm = phi_enc(mb_obs_stack)
phi_s_next_idm = phi_enc(mb_obs_next_stack)

action_logits = idm_model(phi_s_idm, phi_s_next_idm)
phi_pred_enc = fwd_model(phi_s_idm, act_onehot)

idm_loss_val = F.cross_entropy(action_logits, mb_acts, reduction='none').mean()
fwd_loss_enc = F.mse_loss(phi_pred_enc, phi_s_next_idm.detach(), reduction='none').mean(dim=1).mean()
joint_loss_val = (1 - beta) * idm_loss_val + beta * fwd_loss_enc
```

We optimise the joint loss to update the encoder and IDM while training the forward model separately so gradients stay stable. This mirrors the original ICM recipe but plugs directly into the Doom macro-action loop with infinite horizon.

## 5. `icm_lpm.py` – Learning Progress Monitoring 

Learning Progress Monitoring (LPM) reframes curiosity as the gap between predicted dynamics error and realised error. We encode the frame stack, roll it through the dynamics model, and subtract the observed MSE so the intrinsic reward only stays high where learning progress is unfolding.

When the agent encounters pure noise, both models plateau at the same reconstruction error, and the reward approaches zero. Epistemically uncertain but learnable transitions keep producing positive gaps until the dynamics predictor catches up, discouraging the classic Noisy TV trap.

```python
with torch.no_grad():
    phi_t = encoder_model(obs_t)
    pred_next = dynamics_model(phi_t, act_onehot)
    expected_error = math.exp(error_model(phi_t, act_onehot).item())

phi_next_t = encoder_model(obs_next_t)
epsilon = F.mse_loss(pred_next, phi_next_t, reduction='none').view(1, -1).mean().item()
intrinsic_reward = (expected_error - epsilon) if error_ready else 0.0

pred_next = dynamics_model(phi_obs, act_onehot)
dyn_loss = F.mse_loss(pred_next, phi_next)
eps_pred = error_model(phi_obs, act_onehot)

err_loss = F.mse_loss(eps_pred, torch.log(eps_b + EPSILON_CLAMP))
```

The outer loop serially performs PPO updates, then replay-driven training for dynamics and error models. This keeps the expected-error predictor calibrated against the replay buffers gathered in the previous iteration.

### My empirical notes (to be completed after experiments)

- **Does LPM actually suppress noisy-TV artefacts in Doom E1M1?**  
  > _Leave this section blank until I run controlled tests; note down observations, plots, and failure cases here._

## 7. TODOs for future revision

- [ ] Add diagrams / reward curves once experiments stabilise.
- [ ] Document best-known hyperparameter tweaks for Doom E1M1.
