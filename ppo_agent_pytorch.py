"""
ppo_agent_pytorch.py

A self-contained PyTorch reference implementation that follows the core logic
of the TensorFlow PPO agent in OpenAI's random-network-distillation repo.

Focus:
- Clear, explicit core logic (rollout buffers, RND intrinsic reward, normalization,
  GAE for intrinsic & extrinsic returns, PPO clipped objective, separate value
  heads, combined advantages).
- Readable, well-commented code for learning/porting/experimentation.
- NOT focused on engineering details (fancy logging, MPI sync, vectorized env
  efficiency). Hooks/notes included where you'd add distributed sync.

Usage:
- Provide a vectorized environment or a list of environments that behave like
  Gym VectorEnv (reset returns np.ndarray of shape (nenvs, *obs_shape); step
  accepts array of actions shape (nenvs,) and returns obs, rews, dones, infos).
- Policy is included (MLP) and contains two critic heads and an RND module.
  You can swap in your own policy network as long as it follows the API.

Author: adapted for explanation to pythonlearner1025
Date: 2025-10-19
"""

from typing import Optional, Tuple
import time
import math
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ---------------------------
# Utilities: Running stats, RFF, etc.
# ---------------------------

class RunningMeanStd:
    """Running mean & variance (like OpenAI baselines). Works on numpy arrays.
    Not optimized. Useful for observation normalization and reward normalization.
    """
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x: np.ndarray):
        # x should be shape (n, *shape)
        x = np.asarray(x, dtype='float64')
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        # Numerically robust online update
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * (self.count * batch_count / tot_count)
        new_var = M2 / (tot_count)

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class RewardForwardFilter:
    """Reward forward filter: rff[t] = gamma * rff[t-1] + reward[t].
    Used to smooth intrinsic rewards over time before applying running std.
    """
    def __init__(self, gamma):
        self.gamma = gamma
        self.rewems = None

    def update(self, rews):
        # rews: shape (nenvs,) or (nenvs, ) per timestep
        if self.rewems is None:
            self.rewems = rews.copy()
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


# ---------------------------
# Simple MLP Policy with two value heads and RND predictor
# ---------------------------

def mlp(input_dim, hidden_sizes=(256, 256), activation=nn.ReLU):
    layers = []
    last = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, h))
        layers.append(activation())
        last = h
    return nn.Sequential(*layers)


class RNDModule(nn.Module):
    """
    Random Network Distillation stub:
    - target: fixed randomly initialized network (not trained)
    - predictor: trainable network
    intrinsic reward = ||predictor(obs) - target(obs)||^2 (per-sample MSE)
    """
    def __init__(self, in_dim, feat_dim=128, hidden=(128, 128)):
        super().__init__()
        # target network (frozen)
        self.target = mlp(in_dim, hidden_sizes=hidden)
        # predictor network (trained)
        self.predictor = mlp(in_dim, hidden_sizes=hidden)
        self.target_last = nn.Linear(hidden[-1] if hidden else in_dim, feat_dim)
        self.predictor_last = nn.Linear(hidden[-1] if hidden else in_dim, feat_dim)

        # freeze target parameters
        for p in self.target.parameters():
            p.requires_grad = False
        for p in self.target_last.parameters():
            p.requires_grad = False

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # obs shape (..., in_dim)
        t = self.target(obs)
        p = self.predictor(obs)
        tfeat = self.target_last(t)
        pfeat = self.predictor_last(p)
        return pfeat, tfeat

    def intrinsic_reward(self, obs: np.ndarray, device='cpu') -> np.ndarray:
        """
        Compute intrinsic reward for a batch of observations.
        Input obs: numpy array shape (batch, obs_dim)
        Returns: numpy array shape (batch,)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            pfeat, tfeat = self.forward(obs_t)
            # squared error per sample (sum over feature dims)
            mse = torch.mean((pfeat - tfeat) ** 2, dim=-1)
        return mse.cpu().numpy()

    def predictor_loss(self, obs: torch.Tensor) -> torch.Tensor:
        """Loss to optimize predictor network (MSE to target)."""
        pfeat, tfeat = self.forward(obs)
        loss = torch.mean((pfeat - tfeat) ** 2)
        return loss


class Policy(nn.Module):
    """
    Actor-critic policy:
    - shared body (MLP)
    - action head (categorical for discrete action spaces)
    - two value heads (intrinsic & extrinsic)
    - integrated RND module for intrinsic reward
    """
    def __init__(self, obs_dim, n_actions, hidden=(256,256), rnd_feat_dim=128):
        super().__init__()
        self.backbone = mlp(obs_dim, hidden_sizes=hidden)
        last_hidden = hidden[-1] if len(hidden) else obs_dim
        self.action_head = nn.Linear(last_hidden, n_actions)
        self.v_int_head = nn.Linear(last_hidden, 1)
        self.v_ext_head = nn.Linear(last_hidden, 1)

        # For entropy computation we will use Categorical from logits
        # RND
        self.rnd = RNDModule(obs_dim, feat_dim=rnd_feat_dim, hidden=(128,))

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        logits = self.action_head(x)
        v_int = self.v_int_head(x).squeeze(-1)
        v_ext = self.v_ext_head(x).squeeze(-1)
        return logits, v_int, v_ext

    def step(self, obs: np.ndarray, device='cpu'):
        """
        Single-step inference used during rollouts.
        obs: numpy array shape (nenvs, obs_dim)
        Returns:
            actions (np.int32, shape (nenvs,))
            logp (np.float32, shape (nenvs,))
            v_int (np.float32, shape (nenvs,))
            v_ext (np.float32, shape (nenvs,))
            entropy (np.float32, shape (nenvs,))
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        logits, v_int, v_ext = self.forward(obs_t)
        dist = Categorical(logits=logits)
        actions_t = dist.sample()
        logp_t = dist.log_prob(actions_t)
        entropy_t = dist.entropy()
        return (actions_t.cpu().numpy(),
                -logp_t.detach().cpu().numpy(),  # negative log prob to match TF code naming
                v_int.detach().cpu().numpy(),
                v_ext.detach().cpu().numpy(),
                entropy_t.detach().cpu().numpy())


# ---------------------------
# PPO agent (PyTorch ref implementation)
# ---------------------------

class PPOAgent:
    """
    A PyTorch reference implementation of the PPO logic in the TF code.
    Not optimized for speed. Clear, explicit steps and comments.

    Key differences from TF code:
    - No MPI built-in; places where you'd add distributed sync are commented.
    - Buffering is explicit numpy arrays. Conversions to torch happen at update time.
    """
    def __init__(self,
                 policy: Policy,
                 envs,  # vectorized env or list of envs with API described at top
                 nsteps: int = 128,
                 gamma: float = 0.99,
                 gamma_ext: float = 0.99,
                 lam: float = 0.95,
                 nepochs: int = 4,
                 nminibatches: int = 4,
                 cliprange: float = 0.2,
                 ent_coef: float = 0.0,
                 vf_coef: float = 1.0,
                 max_grad_norm: float = 0.5,
                 lr: float = 3e-4,
                 int_coeff: float = 1.0,
                 ext_coeff: float = 2.0,
                 device: Optional[str] = None,
                 use_news_for_intrinsic: bool = False,
                 update_obs_stats_every_step: bool = True):
        self.policy = policy
        self.envs = envs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)

        # hyperparams
        self.nsteps = nsteps
        self.gamma = gamma
        self.gamma_ext = gamma_ext
        self.lam = lam
        self.nepochs = nepochs
        self.nminibatches = nminibatches
        self.cliprange = cliprange
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.int_coeff = int_coeff
        self.ext_coeff = ext_coeff
        self.use_news_for_intrinsic = use_news_for_intrinsic
        self.update_obs_stats_every_step = update_obs_stats_every_step

        # env metadata
        # Expect envs.reset() -> obs shape (nenvs, *obs_shape)
        # Expect envs.step(actions) -> obs, rews, dones, infos (all arrays shape (nenvs, ...))
        first_obs = self.envs.reset()
        assert isinstance(first_obs, np.ndarray)
        self.nenvs = first_obs.shape[0]
        self.obs_shape = first_obs.shape[1:]
        # we need action space info: for simplicity assume discrete and equal across envs
        # user must pass Policy compatible with action space

        # Running stats for observations (normalization) and intrinsic reward RMS
        self.obs_rms = RunningMeanStd(shape=self.obs_shape)
        self.rff_int = RewardForwardFilter(self.gamma)
        self.rff_rms_int = RunningMeanStd(shape=())  # scalar running var for filtered intrinsic reward

        # optimizer (will optimize policy parameters + RND predictor)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        # Buffers
        self._init_buffers()

        # bookkeeping
        self.step_count = 0
        self.t0 = time.time()

    def _init_buffers(self):
        nenvs, nsteps = self.nenvs, self.nsteps
        obs_shape = (nenvs, nsteps + 1) + self.obs_shape  # +1 to store last next_obs for bootstrapping
        self.buf_obs = np.zeros(obs_shape, dtype=np.float32)
        self.buf_acs = np.zeros((nenvs, nsteps), dtype=np.int32)
        self.buf_nlps = np.zeros((nenvs, nsteps), dtype=np.float32)  # old neglogp
        self.buf_vpreds_int = np.zeros((nenvs, nsteps), dtype=np.float32)
        self.buf_vpreds_ext = np.zeros((nenvs, nsteps), dtype=np.float32)
        self.buf_rews_ext = np.zeros((nenvs, nsteps), dtype=np.float32)
        self.buf_rews_int = np.zeros((nenvs, nsteps), dtype=np.float32)  # filled after rollout
        self.buf_news = np.zeros((nenvs, nsteps), dtype=np.float32)  # done flags (1 if done after step)
        self.buf_ent = np.zeros((nenvs, nsteps), dtype=np.float32)

        # last vpreds for bootstrap
        self.buf_vpred_int_last = np.zeros((nenvs,), dtype=np.float32)
        self.buf_vpred_ext_last = np.zeros((nenvs,), dtype=np.float32)
        self.buf_new_last = np.zeros((nenvs,), dtype=np.float32)

    def collect_rollout(self):
        """
        Collect nsteps of data from envs using current policy.
        Fills buffers in-place.
        """
        nenvs, nsteps = self.nenvs, self.nsteps

        # initial observation: will be used as buf_obs[:,0]
        obs = self.envs.reset()
        self.buf_obs[:, 0] = obs

        mem_state = None  # placeholder for recurrent state if needed
        for t in range(nsteps):
            # Policy inference
            actions, oldnlp, vpreds_int, vpreds_ext, entropy = self.policy.step(obs, device=self.device)
            # Step envs
            next_obs, rews, dones, infos = self.envs.step(actions)

            # store
            self.buf_acs[:, t] = actions
            self.buf_nlps[:, t] = oldnlp
            self.buf_vpreds_int[:, t] = vpreds_int
            self.buf_vpreds_ext[:, t] = vpreds_ext
            self.buf_ent[:, t] = entropy
            self.buf_news[:, t] = dones.astype(np.float32)

            # Important: align extrinsic rewards so that reward for transition (obs[t], action) lands at t
            # In the TF code: buf_rews_ext[sli, t-1] = prevrews. Here we will directly store rews at t.
            self.buf_rews_ext[:, t] = rews

            # store next obs into buf_obs[:, t+1]
            self.buf_obs[:, t + 1] = next_obs

            # optionally update observation running stats per step (matches TF option)
            if self.update_obs_stats_every_step:
                # this example uses only the last channel logic from original code if needed.
                # Here we update with whole observation flattened by envs.
                self.obs_rms.update(next_obs.reshape((nenvs, -1)))

            obs = next_obs
            self.step_count += 1

        # After loop, we have buf_obs with shape (nenvs, nsteps+1, *obs_shape)
        # Need last vpreds (bootstrap) from last next_obs
        # Do a final forward pass to get buf_vpred_*_last from buf_obs[:, -1]
        last_obs = self.buf_obs[:, -1]
        with torch.no_grad():
            obs_t = torch.as_tensor(last_obs, dtype=torch.float32, device=self.device)
            logits, v_int_last, v_ext_last = self.policy.forward(obs_t)
            self.buf_vpred_int_last = v_int_last.cpu().numpy()
            self.buf_vpred_ext_last = v_ext_last.cpu().numpy()
            # if not updating obs stats every step, we can update now from the whole rollout
            if not self.update_obs_stats_every_step:
                all_obs = self.buf_obs[:, :-1].reshape((-1,) + self.obs_shape)  # only the nsteps observations
                self.obs_rms.update(all_obs)

        # compute intrinsic rewards using RND
        # the original TF code computed int rewards by feeding (nsteps+1)-long obs sequences into a TF op.
        # Here we compute intrinsic reward per transition using the observation at step t (or next_obs).
        # We'll compute intrinsic reward per transition = RND(obs_t) using predictor-target MSE.
        # We use the observations at t (buf_obs[:, t]) -- this choice is consistent with many RND implementations.
        ob_for_rnd = self.buf_obs[:, :-1].reshape((-1,) + self.obs_shape)  # shape (nenvs*nsteps, obs_dim)
        rews_int_flat = self.policy.rnd.intrinsic_reward(ob_for_rnd, device=self.device)  # shape (nenvs*nsteps,)
        self.buf_rews_int = rews_int_flat.reshape((nenvs, nsteps))

    def normalize_intrinsic_rewards(self):
        """
        The TF implementation:
         - applies RewardForwardFilter to each timestep across envs
         - updates a global RunningMeanStd on the filtered values
         - divides raw intrinsic rewards by sqrt(rff_rms.var)
        We'll mimic that behaviour here.
        """
        nenvs, nsteps = self.nenvs, self.nsteps
        # compute rff per timestep across envs: rff_t = gamma * rff_{t-1} + rews_int[:, t]
        # The code in TF did self.I.rff_int.update(rew) for each timestep column (i.e. per t across all envs)
        # We'll produce a sequence of filtered values rffs_int shape (nenvs, nsteps)
        rffs_int = np.zeros_like(self.buf_rews_int)
        # We'll loop over timesteps to apply filter across time (per-env vectorized)
        # Start internal rff state per env:
        rff_state = np.zeros((nenvs,), dtype=np.float32)  # initial 0
        for t in range(nsteps):
            rff_state = rff_state * self.gamma + self.buf_rews_int[:, t]
            rffs_int[:, t] = rff_state

        # update running mean/std from rffs_int flattened
        self.rff_rms_int.update(rffs_int.reshape(-1, ))

        # normalize intrinsic rewards
        std = math.sqrt(self.rff_rms_int.var + 1e-8)
        rews_int_norm = self.buf_rews_int / std
        return rews_int_norm

    @staticmethod
    def compute_gae(rewards: np.ndarray,
                    values: np.ndarray,
                    last_value: np.ndarray,
                    dones: np.ndarray,
                    gamma: float,
                    lam: float,
                    use_dones: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generic GAE computation (vectorized over nenvs).
        rewards: (nenvs, nsteps)
        values:  (nenvs, nsteps)
        last_value: (nenvs,)
        dones: (nenvs, nsteps)  with 1.0 where done happened after step
        Returns:
            advs: (nenvs, nsteps)
            returns: (nenvs, nsteps)
        """
        nenvs, nsteps = rewards.shape
        advs = np.zeros_like(rewards, dtype=np.float32)
        lastgaelam = np.zeros(nenvs, dtype=np.float32)
        for t in reversed(range(nsteps)):
            if t == nsteps - 1:
                nextnonterminal = 1.0 - (dones[:, t] if use_dones else 0.0)
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - (dones[:, t+1] if use_dones else 0.0)
                nextvalues = values[:, t+1]
            delta = rewards[:, t] + gamma * nextvalues * nextnonterminal - values[:, t]
            lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
            advs[:, t] = lastgaelam
        returns = advs + values
        return advs, returns

    def ppo_update(self):
        """
        Convert buffers to torch tensors, create minibatches, and run PPO updates
        that optimize:
          loss = pg_loss + ent_loss + vf_loss + aux_loss
        where vf_loss includes both intrinsic and extrinsic heads' MSEs, and
        the policy gradient uses the combined advantage: adv = int_coeff*adv_int + ext_coeff*adv_ext
        """
        nenvs, nsteps = self.nenvs, self.nsteps
        batch_size = nenvs * nsteps
        assert batch_size % self.nminibatches == 0
        minibatch_size = batch_size // self.nminibatches

        # 1) Normalize intrinsic rewards
        rews_int_norm = self.normalize_intrinsic_rewards()
        rews_ext = self.buf_rews_ext

        # 2) Compute GAE for intrinsic and extrinsic separately
        adv_int, ret_int = self.compute_gae(rews_int_norm, self.buf_vpreds_int, self.buf_vpred_int_last,
                                           self.buf_news, self.gamma, self.lam,
                                           use_dones=self.use_news_for_intrinsic)
        adv_ext, ret_ext = self.compute_gae(rews_ext, self.buf_vpreds_ext, self.buf_vpred_ext_last,
                                           self.buf_news, self.gamma_ext, self.lam,
                                           use_dones=True)

        # 3) Combine advantages
        adv = self.int_coeff * adv_int + self.ext_coeff * adv_ext

        # 4) Flatten buffers (for feedforward policies)
        # If you have a recurrent policy, you should form minibatches of full sequences per env
        flat_obs = self.buf_obs[:, :-1].reshape((-1,) + self.obs_shape)  # (batch_size, obs_dim)
        flat_acs = self.buf_acs.reshape(-1)
        flat_oldnlp = self.buf_nlps.reshape(-1)
        flat_adv = adv.reshape(-1)
        flat_ret_int = ret_int.reshape(-1)
        flat_ret_ext = ret_ext.reshape(-1)
        flat_ent = self.buf_ent.reshape(-1)

        # optionally normalize advantages (common practice)
        adv_mean, adv_std = flat_adv.mean(), flat_adv.std() + 1e-8
        flat_adv = (flat_adv - adv_mean) / adv_std

        # Convert to tensors
        obs_tensor = torch.as_tensor(flat_obs, dtype=torch.float32, device=self.device)
        acs_tensor = torch.as_tensor(flat_acs, dtype=torch.int64, device=self.device)
        oldnlp_tensor = torch.as_tensor(flat_oldnlp, dtype=torch.float32, device=self.device)
        adv_tensor = torch.as_tensor(flat_adv, dtype=torch.float32, device=self.device)
        ret_int_tensor = torch.as_tensor(flat_ret_int, dtype=torch.float32, device=self.device)
        ret_ext_tensor = torch.as_tensor(flat_ret_ext, dtype=torch.float32, device=self.device)

        # pre-computed dataset indices for shuffling across epochs
        inds = np.arange(batch_size)

        # metrics accumulators
        avg_losses = {}
        for epoch in range(self.nepochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = inds[start:start + minibatch_size]
                mb_obs = obs_tensor[mb_inds]
                mb_acs = acs_tensor[mb_inds]
                mb_oldnlp = oldnlp_tensor[mb_inds]
                mb_adv = adv_tensor[mb_inds]
                mb_ret_int = ret_int_tensor[mb_inds]
                mb_ret_ext = ret_ext_tensor[mb_inds]

                # forward pass
                logits, v_int_new, v_ext_new = self.policy.forward(mb_obs)
                dist = Categorical(logits=logits)
                logp_new = -dist.log_prob(mb_acs)  # note: TF stored neglogp, so keep sign consistent
                entropy = dist.entropy().mean()

                # ratio = exp(oldnlp - neglogpac) following TF code
                ratio = torch.exp(mb_oldnlp.to(self.device) - logp_new)

                # policy loss (PPO clipped)
                pg_losses1 = -mb_adv * ratio
                pg_losses2 = -mb_adv * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
                pg_loss = torch.mean(torch.max(pg_losses1, pg_losses2))

                # value loss for both heads
                vf_loss_int = 0.5 * self.vf_coef * torch.mean((v_int_new - mb_ret_int) ** 2)
                vf_loss_ext = 0.5 * self.vf_coef * torch.mean((v_ext_new - mb_ret_ext) ** 2)
                vf_loss = vf_loss_int + vf_loss_ext

                ent_loss = - self.ent_coef * entropy

                # auxiliary loss: RND predictor MSE on minibatch observations
                rnd_loss = self.policy.rnd.predictor_loss(mb_obs)  # train predictor to match target

                total_loss = pg_loss + ent_loss + vf_loss + rnd_loss

                # backward + step
                self.optimizer.zero_grad()
                total_loss.backward()
                # gradient clipping on global norm
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                # Note: in distributed setup you would allreduce grads here or use DDP wrapper
                self.optimizer.step()

                # metrics: approxkl & clipfrac (approximations consistent with TF)
                approxkl = 0.5 * torch.mean((logp_new - mb_oldnlp.to(self.device)) ** 2).item()
                clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.cliprange).float()).item()

                # accumulate for reporting
                if 'tot' not in avg_losses:
                    avg_losses = {'tot': 0.0, 'pg': 0.0, 'vf': 0.0, 'ent': 0.0, 'rnd': 0.0,
                                  'approxkl': 0.0, 'clipfrac': 0.0, 'count': 0}
                avg_losses['tot'] += total_loss.item()
                avg_losses['pg'] += pg_loss.item()
                avg_losses['vf'] += vf_loss.item()
                avg_losses['ent'] += entropy.item()
                avg_losses['rnd'] += rnd_loss.item()
                avg_losses['approxkl'] += approxkl
                avg_losses['clipfrac'] += clipfrac
                avg_losses['count'] += 1

        # average metrics
        if avg_losses.get('count', 0) > 0:
            count = avg_losses['count']
            for k in list(avg_losses.keys()):
                if k != 'count':
                    avg_losses[k] /= count

        # Return some diagnostics similar to TF update() return info
        info = {
            'loss/total': avg_losses.get('tot', 0.0),
            'loss/pg': avg_losses.get('pg', 0.0),
            'loss/vf': avg_losses.get('vf', 0.0),
            'loss/ent': avg_losses.get('ent', 0.0),
            'loss/rnd': avg_losses.get('rnd', 0.0),
            'approxkl': avg_losses.get('approxkl', 0.0),
            'clipfrac': avg_losses.get('clipfrac', 0.0),
            'tps': (self.nsteps * self.nenvs) / max(1e-6, (time.time() - self.t0))
        }
        return info

    def train_one_iteration(self):
        """
        High-level convenience: collect rollout and perform PPO update.
        Returns diagnostic info.
        """
        self.collect_rollout()
        info = self.ppo_update()
        return info


# ---------------------------
# Minimal pseudo VecEnv for testing
# ---------------------------

class DummyVecEnv:
    """
    A tiny vectorized env wrapper around gym-like single envs for quick testing.
    Expects envs to follow reset() and step(action) -> obs, rew, done, info
    This minimal implementation returns batched arrays.
    """
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.nenvs = len(self.envs)

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.stack(obs, axis=0)

    def step(self, actions):
        results = [env.step(int(a)) for env, a in zip(self.envs, actions)]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs, axis=0), np.array(rews, dtype=np.float32), np.array(dones, dtype=np.bool_), list(infos)


# ---------------------------
# Example usage (pseudocode - not executed at import)
# ---------------------------

if __name__ == "__main__":
    # This pseudo-example demonstrates wiring everything together.
    # Replace `YourEnv` with an actual gym.Env class and ensure obs is flattened vector.
    try:
        import gym
    except Exception:
        gym = None

    if gym is None:
        print("Install gym to run an example.")
    else:
        def make_env():
            return gym.make("CartPole-v1")

        nenvs = 4
        envs = DummyVecEnv([make_env] * nenvs)
        obs0 = envs.reset()
        obs_dim = obs0.shape[1]  # flattened observation
        n_actions = envs.envs[0].action_space.n

        policy = Policy(obs_dim=obs_dim, n_actions=n_actions)
        agent = PPOAgent(policy=policy, envs=envs, nsteps=128, nminibatches=4,
                         nepochs=4, lr=2.5e-4, ent_coef=0.01, vf_coef=0.5,
                         int_coeff=0.1, ext_coeff=1.0, device='cpu')

        for update in range(10):
            info = agent.train_one_iteration()
            print(f"Update {update}: {info}")