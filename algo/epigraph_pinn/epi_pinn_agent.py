# from algo.maddpg.network import Critic, Actor

from algo.epigraph_pinn.network import DynamicsNet, RewardNet, ValueNet, PolicyNet

import torch
from copy import deepcopy
from torch.optim import Adam
from algo.memory import continuous_ReplayMemory
import os
from copy import deepcopy
import torch.nn as nn
import numpy as np
from algo.utils import device
scale_reward = 0.01
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
from algo.utils import *



def c_cost(x, s=20.0):
    """
    x : tensor shape (B, 4)  with ordering [x1, v1, x2, v2]
    return: tensor shape (B, 1)
    """
    x1 = x[:, 0:1]
    x2 = x[:, 2:3]
    raw = x1 - x2 + 0.02                  # small shift to keep it smooth/differentiable
    sigmoid = torch.sigmoid(s * raw)      # σ(s·raw)
    c_val = 2.0 * sigmoid - 1.0           # map to (-1, 1)
    return c_val                          # (B,1)

def smooth_violation_sigmoid(x, margin=0.002, scale=40.0):
    """
    A smooth violation function that:
    - returns ~ -1e-3 when x1 ≪ x2 - margin (safe)
    - returns ~ 0 near x1 ≈ x2 (on constraint boundary)
    - returns → 1 when x1 ≫ x2 (violation)

    Args:
        x: torch.Tensor
        margin: slack before constraint triggers
        scale: steepness of transition (higher = sharper step)
    Returns:
        torch.Tensor with values in (-1e-3, 1)
    """
    x1 = x[:, 0:1]
    x2 = x[:, 2:3]
    raw = scale * (x1 - x2 + margin)
    sig = torch.sigmoid(raw)              # ∈ (0,1)
    return sig * (1 + 1e-3) - 1e-3        # ∈ (-1e-3, 1)

def smooth_violation_spread(x_flat: torch.Tensor,
                            n_agents: int = None,
                            obs_dim: int = None,
                            radius: float = None,
                            delta: float = 1e-9,
                            tau: float = 0.05,
                            agg: str = "sum",       # "sum" | "lse" | "pmax"
                            lmbd: float = 0.2,
                            amplify: float = 1.0):

    n_agents = 3
    obs_dim  = 18
    radius   = 0.2

    B = x_flat.shape[0]
    x   = x_flat.view(B, n_agents, obs_dim)     # [B,N,D]
    pos = x[..., 2:4]                           # take absolute positions p_pos (consistent with your observation definition)

    # Pairwise distances (differentiable)
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)                        # [B,N,N,2]
    d    = torch.sqrt((diff ** 2).sum(dim=-1) )**0.5 + 1e-12          # [B,N,N]

    # Mask diagonal (self-self)
    eye = torch.eye(n_agents, dtype=torch.bool, device=x.device).unsqueeze(0).expand(B, -1, -1)  # [B,N,N]
    d = d.masked_fill(eye, radius*2+delta*2)                           # make s negative on diagonal (safe)

    # Signed gap s: >0 means violation (contact/overlap), <=0 means safe
    s = (2.0 * radius + delta) - d                                     # [B,N,N]

    # Piecewise scaling: amplify s>0 by 10x; keep s<=0 as negative
    g = torch.where(s > 0, s * 10.0, s)                                # [B,N,N]

    # Aggregate only upper triangle i<j (permutation-invariant)
    tri = torch.triu(
        torch.ones(n_agents, n_agents, device=x_flat.device, dtype=torch.bool),
        diagonal=1
    )
    g_pairs = g.masked_fill(~tri, 0.0)                                 # [B,N,N]

    # Aggregation
    if agg == "sum":
        c = g_pairs.sum(dim=(1, 2), keepdim=True)/2                    # [B,1]
    elif agg == "lse":
        gp = g_pairs.view(B, -1)
        c  = lmbd * torch.logsumexp(gp / lmbd, dim=1, keepdim=True)    # [B,1]
    elif agg == "mean":
        num_pairs = (n_agents * (n_agents - 1)) // 2                   # scalar
        denom = torch.tensor(num_pairs, dtype=g_pairs.dtype, device=g_pairs.device)
        c = g_pairs.sum(dim=(1, 2), keepdim=True) / denom              # [B,1]
    elif agg == "max":
        gp = g_pairs.view(B, -1)
        c, _ = gp.max(dim=1, keepdim=True)                             # [B,1]
    else:
        raise ValueError(f"Unknown agg: {agg}")

    return c  # [B,1]; can be negative (safe) or positive (violation, amplified)

def f_wrapper(x, u, dt, net):
    """Return dx/dt prediction so Jacobian is w.r.t. derivative."""
    return (net(x, u, dt) - x) / dt          # [B, ND]

# from functorch import jacrev  # torch >=1.13, or use functorch
from torch import vmap
from torch.func import jacrev
def batched_jacobian(f, x, u, dt):
    def f_wrapped(xi, ui, dti):
        return f(xi.unsqueeze(0), ui.unsqueeze(0), dti.unsqueeze(0)).squeeze(0)
    return vmap(jacrev(f_wrapped))(x, u, dt)  # output shape: [B, D_out, D_in]

def compute_time_to_go_sequence(delta_ts, T):
    """
    Args:
        delta_ts: array-like, [num_steps], delta_t at each step
        T: total time horizon
    Returns:
        time_to_go: array-like, [num_steps], remaining time at each step
    """
    cumulative_time = np.cumsum(delta_ts)  # [dt0, dt0+dt1, dt0+dt1+dt2, ...]
    time_to_go = T - cumulative_time       # remaining time
    return time_to_go


class epi_agent_new:
    def __init__(self, dim_obs, dim_act, n_agents, args, env):
        self.args = args
        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.batch_size = args.batch_size
        self.exploration_steps = args.exploration_steps
        self.episodes_before_train = args.episode_before_train
        self.use_cuda = torch.cuda.is_available()
        self.exploration_noise_std = 0.99

        # Initialize networks
        # self.epi_critic = ValueNet4(dim_obs, dim_act, n_agents, self.args.relu)
        self.cost_nets = [ValueNet(dim_obs, n_agents, self.args.relu) for _ in range(n_agents)]
        self.policy_nets = [PolicyNet(dim_obs, dim_act) for _ in range(n_agents)]
        self.value_nets = [ValueNet(dim_obs, n_agents, self.args.relu) for _ in range(n_agents)]
        self.dynamics_nets = [DynamicsNet(dim_obs, dim_act, n_agents) for _ in range(n_agents)]
        self.reward_nets = [RewardNet(dim_obs, dim_act, n_agents) for _ in range(n_agents)]
        self.single_cost_net = [RewardNet(dim_obs, dim_act, n_agents) for _ in range(n_agents)]

        self.sigma_init = 0.5
        self.sigma_min  = args.noise_level
        self.sigma_decay_steps = 5000
        self._sigma_scale = self.sigma_init

        self.cov_matrix = torch.eye(self.n_actions) * self.sigma_min
        # self.tilde_value_net = ValueNet3(dim_obs, n_agents, self.args.relu)

        self.policy_optimizers      = [Adam(net.parameters(), lr=args.a_lr)        for net in self.policy_nets]
        self.value_optimizers       = [Adam(net.parameters(), lr=args.c_lr)        for net in self.value_nets]
        self.cost_optimizers        = [Adam(net.parameters(), lr=args.lr_cost)     for net in self.cost_nets]
        self.dynamics_optimizers    = [Adam(net.parameters(), lr=args.lr_dynamics) for net in self.dynamics_nets]
        self.reward_optimizers      = [Adam(net.parameters(), lr=args.lr_reward)   for net in self.reward_nets]
        self.single_cost_optimizers = [Adam(net.parameters(), lr=args.c_lr)        for net in self.single_cost_net]

        self.env = env
        # self.tilde_optimizers = Adam(self.tilde_value_net.parameters(), lr=args.c_lr)
        self.z_min = args.z_min
        self.z_max = args.z_max
        if self.use_cuda:
            for i in range(n_agents):
                self.policy_nets[i].cuda()
                self.value_nets[i].cuda()
                self.reward_nets[i].cuda()
                self.cost_nets[i].cuda()
                self.single_cost_net[i].cuda()
                self.dynamics_nets[i].cuda()
            # self.tilde_value_net.cuda()
            self.cov_matrix = self.cov_matrix.cuda()

        self.target_value_nets = deepcopy(self.value_nets)
        self.memory = continuous_ReplayMemory(args.memory_length)
        self.var = [1.0 for _ in range(n_agents)]
        self.steps_done = 0
        self.episode_done = 0

    def _decay_sigma(self):
        # Linear annealing (can also be changed to exponential annealing)
        frac = min(1.0, float(self.episode_done) / float(self.sigma_decay_steps))
        self._sigma_scale = self.sigma_init + frac * (self.sigma_min - self.sigma_init)

    def soft_update_target_value_net(self, tau=0.01):
        for target_param, param in zip(self.target_value_nets.parameters(), self.value_nets.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def load_model(self):
        load_episode = 20000  # you can also parametrize this via args if needed
        model_path = "trained_model/pinn"

        path_flag = True
        for idx in range(self.n_agents):
            path_flag = path_flag \
                        and os.path.exists(f"{model_path}/policy_nets[{idx}]_{load_episode}.pth") \
                        and os.path.exists(f"{model_path}/value_nets[{idx}]_{load_episode}.pth")

        path_flag = path_flag \
                    and os.path.exists(f"{model_path}/dynamics_nets_{load_episode}.pth") \
                    and os.path.exists(f"{model_path}/reward_nets_{load_episode}.pth")

        if path_flag:
            print(f"Loading model from episode {load_episode}...")

            for idx in range(self.n_agents):
                policy_net = torch.load(f"{model_path}/policy_nets[{idx}]_{load_episode}.pth")
                value_net = torch.load(f"{model_path}/value_nets[{idx}]_{load_episode}.pth")
                self.policy_nets[idx].load_state_dict(policy_net.state_dict())
                self.value_nets[idx].load_state_dict(value_net.state_dict())

            self.dynamics_nets.load_state_dict(
                torch.load(f"{model_path}/dynamics_nets_{load_episode}.pth").state_dict())
            self.reward_nets.load_state_dict(torch.load(f"{model_path}/reward_nets_{load_episode}.pth").state_dict())

            print("Model loaded successfully.")
        else:
            print("Model files not found, skipping load.")

    def save_model(self, episode):
        save_dir = os.path.join("trained_model", str(self.args.scenario), str(self.args.algo))
        os.makedirs(save_dir, exist_ok=True)   # create all parent directories automatically

        for i in range(self.n_agents):
            torch.save(
                self.policy_nets[i],
                os.path.join(save_dir, f'policy_net[{i}]_{self.args.seed}_{episode}.pth')
            )

        torch.save(self.value_nets,
                os.path.join(save_dir, f'value_nets_{self.args.seed}_{episode}.pth'))
        torch.save(self.dynamics_nets,
                os.path.join(save_dir, f'dynamics_nets_{self.args.seed}_{episode}.pth'))
        torch.save(self.reward_nets,
                os.path.join(save_dir, f'reward_nets_{self.args.seed}_{episode}.pth'))

    def update(self, batch):
        dynamics_loss = self.dynamics_training(batch)
        reward_loss = self.reward_training(batch)
        for i in range(10):
            const_loss = self.const_training(batch)

        value_loss = self.value_training_epigraph(batch)
        single_cost_loss = self.single_cost_training(batch)

        tilde_value_loss = self.tilde_value_training(batch)
        vgi_loss = self.target_vgi_training(batch)
        policy_loss = self.policy_training(batch)
        self._decay_sigma()
        self.episode_done += 1

        return {
            "dynamics_loss": dynamics_loss,
            "reward_loss": reward_loss,
            "value_loss": value_loss,
            "cost_loss": const_loss,
            "tilde_value_loss": tilde_value_loss,
            "vgi_loss": vgi_loss,
            "Q_loss": single_cost_loss,
            "policy_loss": policy_loss
        }

    def choose_action(self, state, delta_ts):
        """
        Args:
            state: list or array-like, shape [n_agents, state_dim]
            delta_ts: array-like, shape [1] (shared delta_t)
            time_to_go: array-like, shape [1] (shared time_to_go)
        """
        obs = torch.from_numpy(np.stack(state)).float().to(device)
        obs = obs.view(self.n_agents, -1)  # [n_agents, state_dim]

        delta_ts = torch.from_numpy(np.array(delta_ts)).float().to(device)  # [1]
        delta_ts = delta_ts.expand(self.n_agents, 1)  # [n_agents, 1]

        extended_obs = torch.cat([obs, delta_ts], dim=-1)  # [n_agents, state_dim+2]

        actions = torch.zeros(self.n_agents, self.n_actions)
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        for i in range(self.n_agents):
            sb = extended_obs[i].detach()

            mean = self.policy_nets[i](sb).squeeze(0)  # [action_dim]

            dist = MultivariateNormal(mean.view(-1), covariance_matrix=self.cov_matrix)
            if self.args.mode == 'eval':
                act = mean
            else:
                act = dist.rsample()
            act = torch.clamp(act, -1.0, 1.0)

            actions[i, :] = act

        self.steps_done += 1
        return actions.data.cpu().numpy()

    def dynamics_training(self, batch):
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        states, actions, next_states, _, dt, _,_, z = batch
        states       = torch.stack(states).type(FloatTensor)      # [B, N, D]
        next_states  = torch.stack(next_states).type(FloatTensor) # [B, N, D]
        actions      = torch.stack(actions).type(FloatTensor)     # [B, N, A]
        dt           = torch.stack(dt).type(FloatTensor)          # [B, 1]

        B, N, D = states.shape
        for i in range(self.n_agents):
            x_t     = states.view(B, -1)          # [B, N*D]
            u_t     = actions.view(B, -1)         # [B, N*A]
            x_tp1   = next_states.view(B, -1)
            dt      = dt.view(B, -1)
            x_next_pred  = self.dynamics_nets[i](x_t, u_t, dt)                  # [B, N*D]

            loss  = nn.MSELoss()(x_next_pred , x_tp1)  # [B, N*D]
            self.dynamics_optimizers[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dynamics_nets[i].parameters(), 1.0)
            self.dynamics_optimizers[i].step()
        return loss.item()
    
    def compute_z_star(self, x, i, alpha=1.0, bias=0.0):
        """
        Semantics:
        - Safe region (Vh<=0): z* = Vl
        - Violation region (Vh>0): z* = Vl - alpha * Vh + bias
        where alpha>=0 controls penalty strength, and bias is optional.
        Finally clip to [z_min, z_max].
        """
        with torch.no_grad():
            Vh = self.cost_nets[i](x)     # constraint/violation value (>0 indicates violation)
            Vl = self.value_nets[i](x)   # performance/return value (make sure this is actually return-to-go)
            Vh_pos = torch.relu(Vh)      # keep only positive violation part

            z_star = Vl + alpha * Vh_pos + self.args.z_bias

            # z_star = z_star.clamp(min=self.z_min, max=self.z_max)
        return z_star

    def const_training(self, batch):
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        states, _, _, _, dts, returns,  c_vals, z = batch  # c_vals is max_t c(x(t))
        states = torch.stack(states).type(FloatTensor)
        B, N, D = states.shape
        states = states.view(B, -1)
        original_z = torch.stack(z).type(FloatTensor).view(B, 1).requires_grad_(True)  # [B, 1]
        z = original_z/self.args.return_factor  # scale z to match return factor

        c_vals = torch.stack(c_vals).type(FloatTensor)        # [B, N] or [B, 1]
        for i in range(self.n_agents):
            c_val = c_vals[:,i].view(B, -1)
            reversed_vals = torch.flip(c_val, dims=[0])            # ***dim=0***
            suffix_max_rev, _ = torch.cummax(reversed_vals, dim=0) # ***dim=0***
            c_max = torch.flip(suffix_max_rev, dims=[0]).view(B, -1)   # [B, N] / [B, 1]
            # reverse along time dimension

            pred_c = self.cost_nets[i](states).view(B, -1)
            loss = nn.MSELoss()(pred_c, c_max)

            self.cost_nets[i].zero_grad()
            self.cost_optimizers[i].zero_grad()
            loss.backward()
            self.cost_optimizers[i].step()  # reuse optimizer or create one
        return loss.item()
    
    def single_cost_training(self, batch):
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        states, actions, next_states, rewards, dt, returns, c_vals ,z = batch
        states = torch.stack(states).type(FloatTensor)        # [B, N, D]
        actions = torch.stack(actions).type(FloatTensor)      # [B, N, A]
        costs = -torch.stack(rewards).type(FloatTensor)       # [B, N]
        dt = torch.stack(dt).type(FloatTensor)                # [B, 1]

        B, N, D = states.shape
        c_vals = torch.stack(c_vals).type(FloatTensor)        # [B, N] or [B, 1]
        for i in range(self.n_agents):
            x_t = states.view(B, -1)                          # [B, N*D]
            u_t = actions.view(B, -1)                         # [B, N*A]
            c_val = c_vals[:,i].view(B, -1)
            dt = dt.view(B, -1)
            c_hat = self.single_cost_net[i](x_t, u_t, dt).view(B, -1)  # [B, N] or [B]
            loss = nn.MSELoss()(c_hat, c_val)

            self.single_cost_optimizers[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.single_cost_net[i].parameters(), 1)
            self.single_cost_optimizers[i].step()

        return loss.item()

    def value_training_epigraph(self, batch):
        """
        TD(0) critic with no bootstrap on the last step in the trajectory.
        - Non-terminal:  target_t = l_c(t) * dt_t + (gamma^{dt_t}) * V_next
        - Terminal:      target_T = l_c(T) * dt_T          # no bootstrap
        NOTE: if your reward is already integrated per step (not a rate),
            replace `l_c * dt` by just `l_c` (or `-reward`).
        """
        Float = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        states, actions, next_states, rewards, dts, returns, c_vals, z = batch

        B = len(states)
        x_t   = torch.stack(states).type(Float).view(B, -1)
        x_tp1 = torch.stack(next_states).type(Float).view(B, -1)
        dt    = torch.stack(dts).type(Float).view(B, 1)
        l_c   = -torch.stack(rewards).type(Float)      # cost = -reward
        int_returns = -returns  # for monitoring only

        log_gamma = torch.log(torch.as_tensor(self.args.gamma, device=dt.device, dtype=dt.dtype))
        gamma_dt  = torch.exp(log_gamma * dt)          # [B,1]
        for i in range(self.n_agents):
            per_return = int_returns[:, i].view(B, -1)  # [B,1] for agent i
            V_now = self.value_nets[i](x_t)
            loss  = F.mse_loss(V_now, per_return)

            self.value_optimizers[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_nets[i].parameters(), 1.0)
            self.value_optimizers[i].step()

        return loss.item()


    def tilde_value_training(self, batch):
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        states, actions, next_states, rewards, dts, returns, c_vals, z = batch

        B = len(states)
        gamma = self.args.gamma
        log_gamma = np.log(gamma)
        l_c = -torch.stack(rewards).type(FloatTensor)
        c_vals = torch.stack(c_vals).type(FloatTensor)
        returns = -returns
        for i in range(self.n_agents):
            # Step 1: state, time, and goal
            x_t = torch.stack(states).type(FloatTensor).view(B, -1).requires_grad_(True)
            u_t = torch.stack(actions).type(FloatTensor).view(B, -1)
            dtss = torch.stack(dts).type(FloatTensor).view(B, 1)
            per_l_c = l_c[:, i].view(B, -1)              # [B,1] for agent i
            c_val = c_vals[:, i].view(B, -1)             # [B,1] for agent i
            per_return = returns[:, i].view(B, -1)       # [B,1] for agent i
            x_tp1 = torch.stack(next_states).type(FloatTensor).view(B, -1)  # [B, N*D]
            reversed_vals = torch.flip(c_val, dims=[0])            # ***dim=0***
            suffix_max_rev, _ = torch.cummax(reversed_vals, dim=0) # ***dim=0***
            c_max = torch.flip(suffix_max_rev, dims=[0]).view(B, -1)   # [B, N] / [B, 1]

            # Step 2: predictions
            V_cons = self.cost_nets[i](x_t)   # constraint value
            V_ret = self.value_nets[i](x_t)  # return value

            # Sample z and compute z(t)
            z_star = self.compute_z_star(x_t, i).view(B, 1)         # [B, 1]
            value_ret =  V_ret - z_star                             # [B, 1]

            # Mask: route gradients to the active branch
            mask_ret  = (value_ret > V_cons).float()                # [B,1]
            mask_cons = 1.0 - mask_ret                              # [B,1]
            ratio = mask_ret.mean().item()                          # selection ratio

            epigraph_pred = torch.max(V_cons, V_ret - z_star)        # [B,1]
            S = epigraph_pred.sum()
            grad_x = torch.autograd.grad(
                outputs=S,
                inputs=x_t,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]

            # Compute branch selection and grad_z for residual (no grad needed here)
            with torch.no_grad():
                Vh = self.cost_nets[i](x_t)                         # [B,1]
                Vl = self.value_nets[i](x_t)                        # [B,1]
                z_star = self.compute_z_star(x_t, i)                # [B,1] typically detached

                # Branch selection: 1 selects Vl - z, 0 selects Vh
                mask_val = (Vl - z_star > Vh).float()               # [B,1]
                # grad_z: return branch is -1, constraint branch is 0
                grad_z = -mask_val                                  # [B,1] (no grad)

            # Fill None if needed
            if grad_x is None:
                grad_x = torch.zeros_like(x_t)

            f_xt = (x_tp1 - x_t) / dtss
            H_term = (grad_x * f_xt).sum(dim=1, keepdim=True)       # [B,1]
            z_term = grad_z * l_c
            gamma_term = log_gamma * epigraph_pred                  # [B,1]

            residual = torch.max(
                c_max - epigraph_pred,
                H_term - z_term + gamma_term
            )                                                       # [B,1]
            loss_HJB = (residual ** 2).mean()

            # Clear both optimizers
            self.value_optimizers[i].zero_grad()
            self.cost_optimizers[i].zero_grad()

            # Backprop HJB loss
            loss_HJB.backward()
            torch.nn.utils.clip_grad_norm_(self.cost_nets[i].parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.value_nets[i].parameters(), 1.0)
            self.cost_optimizers[i].step()
            self.value_optimizers[i].step()
        return loss_HJB.item()

    def target_vgi_training(self, batch):
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        states, actions, next_states, rewards, dts, returns, c_vals, z = batch

        B = len(states)
        l_c = -torch.stack(rewards).type(FloatTensor)
        int_returns = -deepcopy(returns)
        c_vals = torch.stack(c_vals).type(FloatTensor)
        log_gamma = np.log(self.args.gamma)
        for i in range(self.n_agents):
            # Step 1: state, time, and goal
            x_t = torch.stack(states).type(FloatTensor).view(B, -1).requires_grad_(True)
            u_t = torch.stack(actions).type(FloatTensor).view(B, -1)
            dtss = torch.stack(dts).type(FloatTensor).view(B, 1)

            z_t = torch.stack(z).type(FloatTensor).view(B, 1).requires_grad_(True)  # [B, 1]
            x_tp1 = torch.stack(next_states).type(FloatTensor).view(B, -1)          # [B, N*D]
            per_l_c = l_c[:, i].view(B, -1)                                         # [B,1] for agent i
            c_val = c_vals[:, i].view(B, -1)                                       # [B,1] for agent i
            per_return = int_returns[:, i].view(B, -1)                             # [B,1] for agent i

            z_t_next = self.compute_z_star(x_tp1, i).view(B, 1)                    # [B, 1]

            # Step 2: predictions
            x_t.requires_grad_(True)                                               # for ∇x Ṽ

            with torch.no_grad():
                Vh = self.cost_nets[i](x_t)                                       # [B,1]
                Vl = self.value_nets[i](x_t)                                      # [B,1]
                z_star = self.compute_z_star(x_t, i)                              # [B,1] typically detached

                # Branch selection: 1 selects Vl - z, 0 selects Vh
                mask_val = (Vl - z_star > Vh).float()                             # [B,1]
                # grad_z: return branch is -1, constraint branch is 0
                grad_z = -mask_val                                                # [B,1] (no grad)

            z_t.requires_grad_(True)                                              # for ∂z Ṽ

            v_cons = self.cost_nets[i](x_t)
            v_ret  = self.value_nets[i](x_t)
            v_tilde = torch.max(v_cons, v_ret - z_star.view(B,1))                 # element-wise max

            grad_v_x = torch.autograd.grad(
                v_tilde.sum(), x_t, create_graph=True, retain_graph=True
            )[0]                                                                  # [B, N*D]

            # Update VGI loss
            r_hat     = self.reward_nets[i](x_t, u_t, dtss).view(B, 1)            # (-cost)
            grad_r_x  = torch.autograd.grad(r_hat.sum(), x_t, create_graph=True)[0]   # (B, N*D)

            c_x       = self.single_cost_net[i](x_t, u_t, dtss)                   # (B,1)
            grad_c_x  = torch.autograd.grad(c_x.sum(), x_t, create_graph=True)[0] # (B, N*D)

            # Use mask to select: no violation -> grad_r_x; violation -> grad_c_x
            mask_val_exp = mask_val.expand_as(grad_r_x)                           # [B,ND]
            grad_rc_x = mask_val_exp * grad_r_x + (1.0 - mask_val_exp) * grad_c_x # [B,ND]
            grad_rc_x = -grad_rc_x.view(B, -1)

            # (iii)  ∇_x f(x,u) · ∇_x V(next)
            x_tp1_d   = x_tp1.clone().detach().requires_grad_(True)
            v_cons_n  = self.cost_nets[i](x_tp1_d)
            v_ret_n   = self.value_nets[i](x_tp1_d)
            v_tilde_n = torch.max(v_cons_n, v_ret_n - z_t_next.view(B,1))

            grad_v_x_n = torch.autograd.grad(v_tilde_n.sum(), x_tp1_d, create_graph=True)[0]  # (B,N*D)

            # dynamics Jacobian-vector product via autograd
            x_t_dyn = x_t.clone().detach().requires_grad_(True)
            x_t_next = self.dynamics_nets[i](x_t_dyn, u_t, dtss)
            f_pred = (x_t_next - x_t_dyn) / dtss
            Jt_v = torch.autograd.grad(
                outputs=f_pred,
                inputs=x_t_dyn,
                grad_outputs=grad_v_x_n,
                retain_graph=True, create_graph=True
            )[0]                                                                  # (B, D)

            gamma_dt  = torch.exp(log_gamma * dtss)
            g_hat_vec = grad_rc_x * dtss + gamma_dt * Jt_v

            vgi_loss  = ((grad_v_x - g_hat_vec).pow(2)).mean()

            self.cost_optimizers[i].zero_grad()
            self.value_optimizers[i].zero_grad()
            vgi_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.cost_nets[i].parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.value_nets[i].parameters(), 1.0)
            self.cost_optimizers[i].step()
            self.value_optimizers[i].step()

        return vgi_loss.item()

    def reward_training(self, batch):
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        states, actions, next_states, rewards, dt, returns,_,z = batch

        states = torch.stack(states).type(FloatTensor)     # [B, N, D]
        actions = torch.stack(actions).type(FloatTensor)   # [B, N, A]
        costs = -torch.stack(rewards).type(FloatTensor)    # [B, N]
        dt = torch.stack(dt).type(FloatTensor)             # [B, 1]

        B, N, D = states.shape
        for i in range(self.n_agents):
            x_t = states.view(B, -1)                       # [B, N*D]
            u_t = actions.view(B, -1)                      # [B, N*A]
            cost = costs[:,i]
            dt = dt.view(B, -1)
            r_hat = self.reward_nets[i](x_t, u_t, dt).squeeze()  # [B, N] or [B]
            loss = nn.MSELoss()(r_hat, cost)

            self.reward_optimizers[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_nets[i].parameters(), 1)
            self.reward_optimizers[i].step()

        return loss.item()

    def policy_training(self, batch, ent_coef=1e-3):
        """
        Minimize epigraph Hamiltonian w.r.t. actor via reparameterization.
        H_epi(x,z,px,pz,u) = px^T f(x,u) - pz * ( l_c(x,u) + ln(gamma)*z )
        loss_actor = E[ H_epi * dt ] - ent_coef * Entropy
        """
        Float = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        states, actions, next_states, rewards, dts, returns, c_vals, z = batch

        # Pack
        x_t   = torch.stack(states).type(Float)                  # [B, N, D]
        B, N, D = x_t.shape
        x_flat = x_t.view(B, -1)                                 # [B, N*D]
        z_t    = torch.stack(z).type(Float).view(B, 1)           # [B,1]
        x_tp = torch.stack(next_states).type(Float).view(B, -1)  # [B, N*D]
        device = x_flat.device

        # Generate actions from current policy (reparameterization)
        # Note: we need gradients w.r.t. actions, but do not update other networks here
        actions_list, ent_list = [], []
        for i in range(self.n_agents):
            # 1) Compute px, pz from V_tilde(x,z) = max(Vh(x), Vl(x)-z)
            # Only take gradients w.r.t. x,z, then detach and feed into actor
            x_req = x_flat.clone().detach().requires_grad_(True)
            z_req = z_t.clone().detach().requires_grad_(True)
            dt     = torch.stack(dts).type(Float).view(B, 1)      # [B,1]
            log_gamma = torch.log(torch.as_tensor(self.args.gamma, device=device, dtype=dt.dtype))

            whole_actions = torch.stack(actions).type(Float).view(B, -1)   # [B, N*A]
            with torch.no_grad():
                Vh = self.cost_nets[i](x_flat)                  # [B,1]
                Vl = self.value_nets[i](x_flat)                 # [B,1]
                z_star   = self.compute_z_star(x_flat, i).detach()  # [B,1]

            # Branch selection
            mask = (Vl - z_star > Vh).float()                    # 1 selects value branch, 0 selects constraint branch
            V_tilde = torch.max(Vh, Vl - z_star)

            # px via autograd
            x_req = x_flat.clone().detach().requires_grad_(True)
            V_tilde_for_grad = torch.where(mask>0.5, (self.value_nets[i](x_req)-z_star), self.cost_nets[i](x_req))
            px = torch.autograd.grad(V_tilde_for_grad.sum(), x_req, create_graph=False, retain_graph=False)[0].detach()

            # pz logic: value-branch=-1, constraint-branch=0
            pz = -mask                                           # [B,1]
            con_mask = (1.0 - mask)                               # [B,1] constraint branch mask

            obs_i = x_t[:, i, :]                                  # [B,D]
            pi_in = torch.cat([obs_i, dt], dim=-1)                 # policy input: [obs, dt]
            mean  = self.policy_nets[i](pi_in)                     # [B,A]
            dist  = torch.distributions.MultivariateNormal(mean, covariance_matrix=self.cov_matrix)
            a_i   = dist.rsample()
            whole_actions[:, i * self.n_actions:(i + 1) * self.n_actions] = a_i  # [B, N*A]

            actions_list.append(a_i)
            ent_list.append(dist.entropy().view(B, 1))

            # 3) Build f(x,u) and l_c(x,u)
            # dynamics_nets predicts x_{t+1}, so f ≈ (x_next_pred - x)/dt
            x_next_pred = self.dynamics_nets[i](x_flat, whole_actions, dt).view(B, -1)  # [B, N*D]
            f_hat = (x_next_pred - x_flat) / dt                                         # [B, N*D]

            # reward_nets target is (-reward) = l_c (performance cost)
            l_c_hat = self.reward_nets[i](x_flat, whole_actions, dt).view(B, 1)         # [B,1]
            con_hat = self.single_cost_net[i](x_flat, whole_actions, dt).view(B, 1)     # [B,1] constraint cost

            # 4) Hamiltonian & actor loss
            # H = px^T f - pz * ( l_c + ln(gamma)*z )
            monitor1 = (px * f_hat).sum(dim=1, keepdim=True)                             # [B,1]
            monitor2 = -pz * l_c_hat                                                     # [B,1]
            H = (px * f_hat).sum(dim=1, keepdim=True) - pz * l_c_hat + con_hat*con_mask + log_gamma * V_tilde  # [B,1]

            # Weight by dt (discrete approximation of integral)
            actor_loss = (H*dt).mean()
            self.policy_optimizers[i].zero_grad()
            self.policy_nets[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_nets[i].parameters(), 1.0)
            self.policy_optimizers[i].step()

        return float(actor_loss.detach().item())
