#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 3: PPO-Clip
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import random
from collections import deque
from typing import Deque, List, Tuple

import gymnasium as gym

import os, json, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer


class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        log_std_min: int = -20,
        log_std_max: int = 0,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mu_layer = init_layer_uniform(nn.Linear(64, out_dim))
        self.log_std = nn.Parameter(torch.zeros(out_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        mu = torch.tanh(self.mu_layer(x))  
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.v_layer = init_layer_uniform(nn.Linear(64, 1))
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        value = self.v_layer(x)
        #############################

        return value
    
def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float) -> List:
    """Compute gae."""

    ############TODO#############
    gae = 0
    values = values + [next_value]
    gae_returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        gae_returns.insert(0, gae + values[step])
    #############################
    return gae_returns

# PPO updates the model several times(update_epoch) using the stacked memory. 
# By ppo_iter function, it can yield the samples of stacked memory by interacting a environment.
def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Get mini-batches."""
    batch_size = states.size(0)
    for _ in range(update_epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], values[rand_ids], log_probs[
                rand_ids
            ], returns[rand_ids], advantages[rand_ids]

class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        update_epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, dist = self.actor(state)                # 只是在收資料，先不需要梯度
            selected_action = dist.mean if self.is_test else action
            selected_action = torch.clamp(selected_action, -1.0, 1.0)
            value = self.critic(state)
            old_logp = dist.log_prob(selected_action).sum(dim=-1, keepdim=True)

        if not self.is_test:
            self.states.append(state.detach())
            self.actions.append(selected_action.detach())
            self.values.append(value.detach())
            self.log_probs.append(old_logp.detach())

        return selected_action.cpu().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1 - done).to(self.device))

        return next_state, reward, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state)

        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.cat(self.states).view(-1, self.obs_dim)
        actions = torch.cat(self.actions).view(-1, self.action_dim)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values

        actor_losses, critic_losses = [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(
            update_epoch=self.update_epoch,
            mini_batch_size=self.batch_size,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        ):
            # calculate ratios
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action)

            # actor_loss
            ############TODO#############
            # actor_loss = ?
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            old_log_prob = old_log_prob.sum(dim=-1, keepdim=True)

            # 對 mini-batch 的 advantage 做標準化（關鍵！）
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            ratio = (log_prob - old_log_prob).exp()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv

            # 多維動作的 entropy 也要 sum 再平均
            entropy = dist.entropy().sum(dim=-1, keepdim=True).mean()

            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_weight * entropy
            #############################

            # critic_loss
            ############TODO#############
            # critic_loss = ?
            value_pred = self.critic(state)
            critic_loss = F.mse_loss(value_pred, return_)
            #############################
            
            # train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)

        return actor_loss, critic_loss

    def train(self):
        """Train the PPO agent."""
        self.is_test = False

        state, _ = self.env.reset(seed=self.seed)
        state = np.expand_dims(state, axis=0)

        actor_losses, critic_losses = [], []
        scores = []
        score = 0
        episode_count = 0
        for ep in tqdm(range(1, self.num_episodes)):
            score = 0
            print("\n")
            for _ in range(self.rollout_len):
                self.total_step += 1
                action = self.select_action(state)
                action = action.reshape(self.action_dim,)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward[0][0]

                # if episode ends
                if done[0][0]:
                    episode_count += 1
                    state, _ = self.env.reset(seed=self.seed)
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    print(f"Episode {episode_count}: Total Reward = {score}")

                    avg20 = float(np.mean(scores[-20:])) if len(scores) >= 20 else float(np.mean(scores))

                    # 初始化 best（存在 agent 上，不用改 __init__）
                    if not hasattr(self, "_best_avg20"):
                        self._best_avg20 = -float("inf")

                    # 只要更好就覆蓋存檔
                    if avg20 > self._best_avg20 + 1e-6:  # +1e-6 防止浮點數抖動
                        os.makedirs("snapshots", exist_ok=True)
                        torch.save(
                            {
                                "actor": self.actor.state_dict(),
                                "critic": self.critic.state_dict(),
                                "total_step": int(self.total_step),
                                "seed": int(self.seed),
                                "best_avg20": avg20,
                                "episode": int(episode_count),
                            },
                            "snapshots/LAB7_314551147_task3_best.pt",  # ← 把 StudentID 換成你的學號
                        )
                        wandb.log({"best_avg20": float(avg20), "total_step": int(self.total_step)})
                        self._best_avg20 = avg20
                        print(f"[CKPT] best updated: avg20={avg20:.2f}, episode={episode_count}, step={self.total_step}")
                    
                    wandb.log({
                        "episode": int(episode_count),
                        "episode_return": float(scores[-1]),
                        "avg20_return": float(avg20),
                        "total_step": int(self.total_step),
                    })
                    score = 0

            actor_loss, critic_loss = self.update_model(next_state)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            wandb.log({
                "actor_loss": float(actor_loss),
                "critic_loss": float(critic_loss),
                "total_step": int(self.total_step),
            })

            milestones = [1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000]
            tags = {1_000_000:"1m", 1_500_000:"1p5m", 2_000_000:"2m", 2_500_000:"2p5m", 3_000_000:"3m"}
            for m in milestones:
                if self.total_step >= m and not hasattr(self, f"_saved_{m}"):
                    os.makedirs("snapshots", exist_ok=True)
                    torch.save(
                        {
                            "actor": self.actor.state_dict(),
                            "critic": self.critic.state_dict(),
                            "total_step": int(self.total_step),
                            "seed": int(self.seed),
                        },
                        f"snapshots/LAB7_314551147_task3_ppo_{tags[m]}.pt",  # ← 把 StudentID 換成你的學號
                    )
                    setattr(self, f"_saved_{m}", True)



        # termination
        self.env.close()

    def test(self, video_folder: str):
        """Test the agent."""
        self.is_test = True

        tmp_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += float(reward)

        print("score: ", score)
        self.env.close()

        self.env = tmp_env
 
def _build_actor_from_env(env, device):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    actor = Actor(obs_dim, act_dim).to(device)
    actor.eval()
    return actor

def _load_actor_weights(actor, path, device):
    ckpt = torch.load(path, map_location=device)
    step = None
    if isinstance(ckpt, dict):
        step = int(ckpt.get("total_step", -1)) if "total_step" in ckpt else None
        if "actor" in ckpt and isinstance(ckpt["actor"], dict):
            actor.load_state_dict(ckpt["actor"])
        elif "actor_state_dict" in ckpt:
            actor.load_state_dict(ckpt["actor_state_dict"])
        else:
            actor.load_state_dict(ckpt)
    else:
        actor.load_state_dict(ckpt)
    return actor, step

def _evaluate_once(env, actor, device, seed, episodes_per_seed=1):
    total = 0.0
    for _ in range(episodes_per_seed):
        obs, _ = env.reset(seed=seed)
        done = False
        ep_r = 0.0
        while not done:
            s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            # 用 policy 的均值做評測
            with torch.no_grad():
                _, dist = actor(s)
            action = dist.mean.detach().cpu().numpy()[0]
            action = np.clip(action, -1.0, 1.0)
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_r += r
        total += ep_r
    return total / episodes_per_seed

def run_eval(model_path, episodes_per_seed, seed_start, seed_end, outdir, device):
    os.makedirs(outdir, exist_ok=True)
    env = gym.make("Walker2d-v4", render_mode=None)
    actor = _build_actor_from_env(env, device)
    actor, eval_step = _load_actor_weights(actor, model_path, device)
    print(f"[EVAL] evaluating checkpoint at total_step={eval_step}")

    seeds = list(range(seed_start, seed_end + 1))
    returns = []
    for sd in seeds:
        ret = _evaluate_once(env, actor, device, sd, episodes_per_seed)
        print(f"[EVAL] seed={sd}, return={ret:.2f}")
        returns.append(ret)

    avg_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    print(f"[EVAL] seeds {seed_start}-{seed_end} | avg={avg_ret:.2f} ± {std_ret:.2f}")

    # 寫檔（一次就好）
    with open(os.path.join(outdir, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "return"])
        for s, r in zip(seeds, returns):
            w.writerow([s, r])
        w.writerow(["avg", avg_ret])
        w.writerow(["std", std_ret])
        w.writerow(["eval_step", eval_step])
        w.writerow(["model_path", model_path])

    with open(os.path.join(outdir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "seeds": seeds,
            "returns": returns,
            "avg": avg_ret,
            "std": std_ret,
            "eval_step": eval_step,
            "model_path": model_path
        }, f, ensure_ascii=False, indent=2)

    env.close()
    return avg_ret, std_ret
        
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="walker-ppo-run")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=2000)  
    parser.add_argument("--update-epoch", type=int, default=10)

    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--eval-seed-start", type=int, default=0)
    parser.add_argument("--eval-seed-end", type=int, default=19)
    parser.add_argument("--eval-outdir", type=str, default="eval_outputs_task3")
    args = parser.parse_args()
 
    # environment
    env = gym.make("Walker2d-v4", render_mode=None)
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        wandb.init(project="DLP-Lab7-PPO-Walker", name=args.wandb_run_name, save_code=True, config=vars(args))
        agent = PPOAgent(env, args)
        agent.train()
    else:
        if not args.model_path:
            raise ValueError("請提供 --model-path 來載入要評估的權重檔 .pt")
        avg, std = run_eval(
            model_path=args.model_path,
            episodes_per_seed=args.episodes_per_seed,
            seed_start=args.eval_seed_start,
            seed_end=args.eval_seed_end,
            outdir=args.eval_outdir,
            device=device,
        )
        print(f"[EVAL DONE] avg={avg:.2f} ± {std:.2f}")
    """
    wandb.init(project="DLP-Lab7-PPO-Walker", name=args.wandb_run_name, save_code=True)

    
    agent = PPOAgent(env, args)
    agent.train()
    """