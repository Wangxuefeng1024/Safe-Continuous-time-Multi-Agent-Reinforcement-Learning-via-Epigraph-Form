import os
import sys
import argparse
import datetime
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

# =========================
# 1) Add project root to sys.path (no absolute path)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append('/local/scratch/a/wang6067/EPI_continuous_marl')
# =========================
# 2) Your imports
# =========================
from continuous_env.make_env import make_env
from algo.epigraph_pinn.epi_pinn_agent import epi_agent_new
from algo.utils import *  # consider replacing with explicit imports for camera-ready cleanliness

# =========================
# 3) Device (your original code uses "device" but did not define it)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_discounted_returns(rewards, dts, gamma=0.99):
    """
    Compute discounted returns for a single episode trajectory with irregular time steps.
    Args:
        rewards: list of tensors, each is [n_agents]
        dts:     list of scalar tensors or floats, each is dt for that step
        gamma:   discount factor; using gamma**dt to approximate exp(-rho*dt)
    Returns:
        returns: tensor [T, n_agents]
    """
    rewards = torch.stack(rewards).to(device)  # [T, n_agents]
    T, n_agents = rewards.shape[0], rewards.shape[1]

    if isinstance(dts[0], torch.Tensor):
        dts_t = torch.stack(dts).to(device).view(-1)  # [T]
    else:
        dts_t = torch.tensor(dts, dtype=torch.float32, device=device).view(-1)  # [T]

    returns = torch.zeros_like(rewards)
    future_return = torch.zeros(n_agents, device=device)

    for t in reversed(range(T)):
        discount = gamma ** dts_t[t]
        future_return = rewards[t] * dts_t[t] + discount * future_return
        returns[t] = future_return

    return returns


def compute_time_to_go_sequence(delta_ts, T_total):
    """
    Compute time-to-go (remaining time) for each step.
    Args:
        delta_ts: [num_steps] array of dt
        T_total:  total horizon
    Returns:
        time_to_go: [num_steps] array
    """
    cumulative_time = np.cumsum(delta_ts)
    time_to_go = T_total - cumulative_time
    return time_to_go


def sample_irregular_dts(num_steps, T_total, rng: np.random.Generator):
    """
    Sample an episode-specific irregular dt sequence with sum(dt)=T_total.
    Using Dirichlet ensures dt_i > 0 and total duration is fixed.
    """
    delta_ts = rng.dirichlet(np.ones(num_steps, dtype=np.float32)) * T_total
    time_to_go = compute_time_to_go_sequence(delta_ts, T_total)
    return delta_ts.astype(np.float32), time_to_go.astype(np.float32)


def main(args):
    # ---- Environment & directories ----
    env = make_env(args.scenario, args.seed)

    model_dir = Path("./trained_model") / args.algo
    model_dir.mkdir(parents=True, exist_ok=True)

    # Some envs expose env.world.seed; guard to avoid attribute errors
    if hasattr(env, "world") and hasattr(env.world, "seed"):
        env.world.seed = args.seed

    # ---- Seeding ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rng = np.random.default_rng(args.seed)

    # ---- Dimensions ----
    n_agents = env.n
    n_actions = env.world.dim_p
    args.n_agents = n_agents
    n_states = env.observation_space[0].shape[0]  # assume all agents share obs_dim

    # ---- Model ----
    if args.algo == "epi":
        model = epi_agent_new(n_states, n_actions, n_agents, args, env)

    model.model_dir = str(model_dir)
    print(model)

    # ---- z annealing ----
    initial_z_max = float(args.return_factor)
    final_z_min = float(args.z_lowerbound)
    z_decay_start = 0
    z_decay_end = 5000

    denom = max(1, (z_decay_end - z_decay_start))
    decay_rate = (initial_z_max - final_z_min) / denom
    z_range = initial_z_max

    # ---- Training settings ----
    T_total = 5.0
    model.T = T_total
    num_steps = int(args.episode_length)

    episode = 0

    while episode < args.max_episodes:
        # Re-sample irregular dt every episode (more sensible than fixing dt for all episodes)
        delta_ts, time_to_go = sample_irregular_dts(num_steps, T_total, rng)

        state = env.reset()

        # Trajectory buffers
        trajectory_obs = []
        trajectory_next_obs = []
        trajectory_actions = []
        trajectory_rewards = []
        trajectory_dts = []
        trajectory_constraints = []
        trajectory_z = []

        # Anneal z_range within window
        if z_decay_start <= episode <= z_decay_end:
            z_range = max(final_z_min, z_range - decay_rate)

        # Sample z once per episode
        sampled_z = float(rng.uniform(low=0.0, high=max(1e-8, z_range)))

        episode += 1
        step = 0

        # Episode logging
        accum_reward_total = 0.0

        # Rollout
        while True:
            if args.mode != "train":
                raise NotImplementedError("This cleaned script currently supports train only. Eval can be added if needed.")

            # Guard against dt index overflow
            if step >= num_steps:
                done = [True] * n_agents
                con_constraints = None
                break

            # Choose action
            action = model.choose_action(state, float(delta_ts[step]))

            # Env step
            next_state, reward, done, con_constraints = env.step_con(deepcopy(action), float(delta_ts[step]))
            con_reward, dis_reward = reward

            con_reward = np.asarray(con_reward, dtype=np.float32).reshape(-1)  # [n_agents]
            dis_reward = np.asarray(dis_reward, dtype=np.float32).reshape(-1)  # [n_agents]
            combined_reward = con_reward + dis_reward

            accum_reward_total += float(np.sum(combined_reward))

            # z update (keep your formula, but ensure numeric stability)
            dt_now = float(delta_ts[step])
            z_next = sampled_z - (-np.sum(con_reward) / args.normal_factor + np.log(args.gamma) * sampled_z) * dt_now
            z_next = float(np.clip(z_next, args.z_min, args.z_max))

            # Store trajectory (epi)
            obs = torch.from_numpy(np.stack(state)).float().to(device)         # [n_agents, obs_dim]
            obs_ = torch.from_numpy(np.stack(next_state)).float().to(device)   # [n_agents, obs_dim]

            # Keep your learning signal: normalized continuous reward only
            con_r_t = torch.from_numpy(con_reward / args.normal_factor).float().to(device)  # [n_agents]

            ac_tensor = torch.as_tensor(action, dtype=torch.float32, device=device)         # [n_agents, act_dim]
            dt_tensor = torch.tensor(dt_now, dtype=torch.float32, device=device)            # scalar
            constraint_tensor = (
                torch.tensor(con_constraints, dtype=torch.float32, device=device)
                if con_constraints is not None
                else torch.zeros(1, device=device)
            )
            z_tensor = torch.tensor(sampled_z, dtype=torch.float32, device=device)          # scalar

            trajectory_obs.append(obs)
            trajectory_next_obs.append(obs_)
            trajectory_actions.append(ac_tensor)
            trajectory_rewards.append(con_r_t)
            trajectory_dts.append(dt_tensor)
            trajectory_constraints.append(constraint_tensor)
            trajectory_z.append(z_tensor)

            # Advance
            state = next_state
            sampled_z = z_next
            step += 1

            # Termination
            if (step >= num_steps) or (True in done):
                break

        # Update at end of episode
        returns_tensor = compute_discounted_returns(
            rewards=trajectory_rewards,
            dts=trajectory_dts,
            gamma=args.gamma
        )  # [T, n_agents]

        batch = [
            trajectory_obs,
            trajectory_actions,
            trajectory_next_obs,
            trajectory_rewards,
            trajectory_dts,
            returns_tensor,
            trajectory_constraints,
            trajectory_z,
        ]

        losses = model.update(batch)

        # Use .get to avoid KeyError if some losses are disabled by ablations
        d_loss      = losses.get("dynamics_loss", None)
        r_loss      = losses.get("reward_loss", None)
        v_loss      = losses.get("value_loss", None)
        c_loss      = losses.get("cost_loss", None)
        a_loss      = losses.get("policy_loss", None)
        tildeV_loss = losses.get("tilde_value_loss", None)
        Q_loss      = losses.get("Q_loss", None)
        vgi_loss    = losses.get("vgi_loss", None)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[Episode {episode:05d}] total_reward={accum_reward_total:.4f}")


        # Save model
        if (episode % args.save_interval == 0) and (args.mode == "train"):
            model.save_model(episode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="formation", type=str,
                        help="simple_spread/line/corridor/formation/simple_tag/target")
    parser.add_argument("--max_episodes", default=30000, type=int)
    parser.add_argument("--algo", default="epi", type=str, help="cleaned for epi entrypoint")
    parser.add_argument("--mode", default="train", type=str, help="train/eval")
    parser.add_argument("--episode_length", default=50, type=int)

    parser.add_argument("--memory_length", default=int(1e4), type=int)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--seed", default=120, type=int)

    parser.add_argument("--a_lr", default=0.0001, type=float)
    parser.add_argument("--c_lr", default=0.001, type=float)

    parser.add_argument("--lr_dynamics", default=0.001, type=float)
    parser.add_argument("--lr_reward", default=0.001, type=float)
    parser.add_argument("--lr_cost", default=0.001, type=float)

    parser.add_argument("--return_factor", default=15, type=float)
    parser.add_argument("--z_lowerbound", default=0, type=float)
    parser.add_argument("--z_min", default=0.0, type=float)
    parser.add_argument("--z_max", default=50.0, type=float)
    parser.add_argument("--noise_level", default=0.1, type=float)

    parser.add_argument("--plot_frequency", default=1000, type=int)
    parser.add_argument("--normal_factor", default=2, type=float)

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--rnn_hidden_size", default=64, type=int)

    parser.add_argument("--ablation_hjb", default=False, type=bool)
    parser.add_argument("--ablation_target", default=False, type=bool)
    parser.add_argument("--ablation_vgi", default=False, type=bool)

    parser.add_argument("--render_flag", default=False, type=bool)
    parser.add_argument("--use_wandb", default=False, action="store_true")
    parser.add_argument("--exploration_steps", default=1000, type=int)

    parser.add_argument("--ou_theta", default=0.15, type=float)
    parser.add_argument("--ou_mu", default=0.0, type=float)
    parser.add_argument("--ou_sigma", default=0.2, type=float)
    parser.add_argument("--z_bias", default=0.2, type=float)

    parser.add_argument("--epsilon_decay", default=10000, type=int)

    parser.add_argument("--tensorboard", default=False, action="store_true")
    parser.add_argument("--ablation", default=False, action="store_true")
    parser.add_argument("--relu", default=False, action="store_true")

    parser.add_argument("--save_interval", default=3000, type=int)
    parser.add_argument("--model_episode", default=300000, type=int)
    parser.add_argument("--episode_before_train", default=10, type=int)
    parser.add_argument("--log_dir", default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    args = parser.parse_args()


    main(args)
