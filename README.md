# Can Curiosity Driven Agents Play Doom? 

My effort to get a curiosity driven agent to complete Doom E1M1 level "The Hangar" 

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
pip install -r requirements.txt
```

PyTorch + ViZDoom wheel selections are pinned inside `requirements.txt`. If wheels fail for your platform, rebuild ViZDoom from source (see `scripts/`).

### 1.3 Baseline run (ICM)
```bash
source env/bin/activate
python icm.py
```

### 1.4 Learning Progress Monitor (LPM) run
```bash
source env/bin/activate
python icm_lpm.py  
```

## 2. Environment snapshot

- **Level**: Doom 1, map `E1M1` 
- **API**: ViZDoom player mode, 60×80 grayscale stacks 
- **Action space**: defined in `MACRO_ACTIONS` inside each script

In both `icm.py` and `icm_lpm.py`, the environment supports infinite-horizon mode. In this mode, death does not terminate the observe/act loop in `run_episode`, and only hitting the timeout or `max_frames` stops the rollout.

Following Burda et al., the death-to-respawn transition is treated as just another state change. This mirrors how people play Doom and discourages reward hacking via intentional suicide, because the spawn room becomes boring once the agent has fully explored it.

## 3. Papers & origin

| Implementation | Reference | Link |
| -------------- | --------- | ---- |
| `icm.py`       |  Large Scale Curiosity Driven Learning | https://arxiv.org/pdf/1808.04355 |
| `icm_lpm.py`   | Beyond Noisy TVs: Noise-Robust Exploration via Learning Progress Monitoring | https://arxiv.org/pdf/2509.25438v1 |

## 4. `icm.py` – Intrinsic Curiosity Module

Pathak's Intrinsic Curiosity Module trains a forward dynamics predictor alongside PPO with GAE-normalised advantages. Curiosity is the prediction error between encoded next-state features and the forward dynamics model's output, so we freeze the policy rollout to collect frame stacks and actions.

During each PPO minibatch we encode the stacked frames, predict the next embedding, and measure the curiosity reward as the mean-squared error. The encoder then feeds an inverse dynamics model that learns to recover the action, keeping features focused on agent-controllable factors.

## 4.1 Core training code

```python
fwd_optimizer = torch.optim.Adam(phi_enc.parameters())
feat_idm_optimizer = torch.option.Adam(list(phi_enc.parameters()) + list(idm_model.parameters()))

phi_s = phi_enc(mb_obs_stack)
# mb_obs_next_stack is collected during rollout
phi_s_next = phi_enc(mb_obs_next_stack)
phi_pred = fwd_model(phi_s.detach(), act_onehot)

per_sample_fwd = F.mse_loss(phi_pred, phi_s_next.detach(), reduction='none').mean(dim=1)

# train just the fwd dynamics model (phi_enc detached)
fwd_loss = per_sample_fwd.mean()
fwd_loss.backward()
fwd_optimizer.step()

phi_idm_t = phi_enc(mb_obs_stack)
phi_idm_t_next = phi_enc(mb_obs_next_stack)

# train both phi_enc (feature encoder) and idm_model (inverse dynamics model)
action_logits = idm_model(phi_idm_t, phi_idm_t_next)
# mb_acts is collected during rollout
idm_loss = F.cross_entropy(action_logits, mb_acts, reduction='none').mean()

phi_pred_enc = fwd_model(phi_idm_t, act_onehot)
fwd_loss_enc = F.mse_loss(phi_pred_enc, phi_idm_t_next.detach(), reduction='none').mean(dim=1).mean()
joint_loss = (1 - beta) * idm_loss + beta * fwd_loss_enc
joint_loss.backward()
feat_idm_optimizer.step()
```

We optimise the joint loss to update the encoder and IDM while training the forward model separately so gradients stay stable, mirroring the original ICM recipe.

## 5. `icm_lpm.py` – Learning Progress Monitoring 

Learning Progress Monitoring (LPM)'s key idea is to set reward signal as prediction improvement over training iterations, where the current iteration is $`\tau`$:
 
$$\text{rwd}_i = \epsilon^{\tau-1} - \epsilon^{\tau}$$

$`o_{t}`$ is the observed state of the game in episode step t, and

$$\hat{o}_{t} = f_{\text{dynamics}}^{\tau}(o_{t-1}, a_{t-1})$$

$$\epsilon^{\tau} = \log(\text{MSE}(\hat{o}_{t}, o_{t}))$$

$`\epsilon^{\tau-1}`$ can be naively calculated by using $`f_{\text{dynamics}}^{\tau-1}`$, but we instead train a neural network to predict it directly given the observation and action. 

The formulation rewards agents for improved prediction of observation transition dynamics. Let's consier two cases: transitions that are learnable and transitions that are not. 

In the first case, an agent is uncertain about a transition at first but over iterations the forward dynamics model learns, and $`\epsilon^{\tau}`$ approaches zero, and the agent receives reward proportional to its past uncertainty, $`\epsilon^{\tau-1}`$. But since the error prediction model is trained too, $`\epsilon^{\tau-1}`$ approaches zero too, so over many iterations rwd_i for this particular transition approaches zero.  

In the second case, the transition is purely random and both forward dynamics model and error prediction model's output logits collapse to a uniform distribution and are  equivalent. Therefore, $`\textrwd_i`$ is zero. 

The above two cases illustrates how, in theory, a LPM model would act out of curiosity (seeks learnable novel dynamics of the world), while avoiding the Noisy TV problem (derives no reward from incompressible/unlearnable dynamics, i.e. random events)

## 5.1 Core Training Code

```python
with torch.no_grad():
    phi_t = phi_enc(obs_t)
    pred_phi_t_next = fwd_model(phi_t, act_onehot)
    expected_error = math.exp(error_model(phi_t, act_onehot).item())

phi_t_next = phi_enc(obs_next_t)
epsilon = F.mse_loss(pred_phi_t_next, phi_t_next, reduction='none').view(1, -1).mean().item()
intrinsic_reward = (expected_error - epsilon) if error_ready else 0.0

pred_phi_t_next = fwd_model(phi_obs, act_onehot)
dyn_loss = F.mse_loss(pred_phi_t_next, phi_next)
eps_pred = error_model(phi_obs, act_onehot)

err_loss = F.mse_loss(eps_pred, torch.log(eps_b + EPSILON_CLAMP))
```

### Empirical Observations 

- [e0ee62b](https://github.com/pythonlearner1025/BoredDoomGuy/tree/e0ee62bbf3801535a81fb67de552775668b8a634): 
    - Total mean rwd dips down from 1 and and plateus around 1e-3 (iter 30). Intrinsic rwd identical to total mean rwd graph, per-iter mean extrinsic more noisy but oscillates around 1e-3 with around 0.018 std, while the running average shows the same story.  
    - Forwad dynamics error is extremely small (starts 4e-4 and approaches 1e-5 by iter 30), suggesting RandomEncoder destroys visual differences in obs during encoding   
- [ad61d92](https://github.com/pythonlearner1025/BoredDoomGuy/tree/ad61d92db335c3f47582ac28a72a787239281aa9): 
    - This is an implementation of OpenAI's [Random Network Distillation](https://openai.com/index/reinforcement-learning-with-prediction-based-rewards/) work
    - Empirically, with INTRINSIC_COEF=EXTRINSIC_COEF=0.5, it was the only implementation out of ICM (Pathak et al, 2017) and LPM (Hou et al, 2025) that didn't asymptotically converge to zero reward. 
    - However, intrinsic reward collapses to zero early on and the policy is updated only by extrinsic rewards - i.e. it behaves no differently than a PPO agent with reward scaled by half. 
    - I am setting EXTRINSIC_COEF=1.0 to investigate if purely curiosity driven agent can gain external rewards + exhibit complex behavior (kills enemies, finds secret room, completes level) 

## 7. TODOs for future revision

- [X] Implement Learning Progress Monitoring, a 2025 paper solution to Noisy Tv Problem. Does it work? No, converges to no action
- [X] Implement Random Network Distillation, OpenAI's solution. Does it work? No, extrinsic reward dominates policy.
- [ ] Document best-known hyperparameter tweaks for Doom E1M1.
