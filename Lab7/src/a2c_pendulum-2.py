#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh


import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm
from typing import Tuple
import os

def parse_range(spec: str):
    """把 '0-19' 或 '0,3,5' 轉成整數列表。"""
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip() != ""]

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize."""
        super(Actor, self).__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean_layer = nn.Linear(128, out_dim)

        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.constant_(self.fc2.bias, 0.0)
        initialize_uniformly(self.mean_layer)  # 小範圍初始化輸出層

        self.log_std = nn.Parameter(torch.full((out_dim,), -0.3)) 
        #############################
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        ############TODO#############
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # 將均值限制在環境動作範圍內，減少 clamp 對學習的破壞
        mean = self.mean_layer(x)                # ← 不做 tanh
        log_std = torch.clamp(self.log_std, -5.0, 0.5)
        std = log_std.exp().expand_as(mean)

        dist = Normal(mean, std)
        action = dist.rsample()
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value_layer = nn.Linear(128, 1)

        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.constant_(self.fc2.bias, 0.0)
        initialize_uniformly(self.value_layer)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value_layer(x)
        #############################

        return value
    

class A2CAgent:
    """A2CAgent interacting with environment.

    Atribute:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args=None):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes
        self.reward_scale = 20.0
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

        self.best_eval_mean = -1e18
        self.ckpt_dir = getattr(args, "ckpt_dir", os.getcwd())
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.saved_pass_checkpoint = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        _, dist = self.actor(state)
        raw = dist.mean if self.is_test else dist.rsample()
        tanh_a = torch.tanh(raw)
        action = 2.0 * tanh_a
        if not self.is_test:
            log_prob_raw = dist.log_prob(raw).sum(dim=-1)
            correction = torch.log(2.0 * (1.0 - tanh_a.pow(2)) + 1e-6).sum(dim=-1)
            log_prob = log_prob_raw - correction
            self.transition = [state, log_prob, None]
        return action.detach().cpu().numpy()

    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        if not self.is_test:
            scaled_reward = reward / self.reward_scale 
            self.transition.extend([next_state, scaled_reward, done])

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        state, log_prob, _, next_state, reward, done = self.transition

        state = state.to(self.device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        mask = 1.0 - done
        
        state_value = self.critic(state)
        next_state_value = self.critic(next_state)

        ############TODO#############
        target_value = reward + self.gamma * next_state_value * mask
        value_loss = F.smooth_l1_loss(state_value, target_value.detach())
        #############################

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        ############TODO#############
        advantage = (target_value - state_value).detach()
        _, cur_dist = self.actor(state)
        entropy = cur_dist.entropy().sum(-1)
        policy_loss = -(log_prob * advantage + self.entropy_weight * entropy).mean()
        #############################
        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        return policy_loss.item(), value_loss.item()

    def evaluate(self, num_episodes = 20, seeds=range(20)):
        was_test = self.is_test
        self.is_test = True
        returns = []
        for i, sd in enumerate(seeds[:num_episodes]):
            state, _ = self.env.reset(seed=int(sd))
            done, ep_ret = False, 0.0
            while not done:
                action = self.select_action(state)
                state, reward, done = self.step(action) 
                ep_ret += reward
            returns.append(ep_ret)
        self.is_test = was_test
        return float(np.mean(returns)), float(np.std(returns))

    def evaluate_verbose(self, seeds):
            """逐個 seed 評估並列印每個回報；回傳 (avg, std)。"""
            was_test = self.is_test
            self.is_test = True
            returns = []

            for sd in seeds:
                state, _ = self.env.reset(seed=int(sd))
                done, ep_ret = False, 0.0
                while not done:
                    action = self.select_action(state)   # is_test=True → 用 mean
                    state, reward, done = self.step(action)
                    ep_ret += reward
                print(f"[EVAL] seed={sd}, return={ep_ret:.2f}")
                returns.append(float(ep_ret))

            self.is_test = was_test
            avg = float(np.mean(returns))
            std = float(np.std(returns))
            # 總結行（照你範例格式；假設 seeds 是連續的）
            print(f"[EVAL] seeds {min(seeds)}-{max(seeds)} | avg={avg:.2f} ± {std:.2f}")
            return avg, std

    def _save_checkpoint(self, path, step, eval_mean, eval_std):
        torch.save({
        'actor': self.actor.state_dict(),
        'critic': self.critic.state_dict(),
        'step': step,
        'eval_mean': eval_mean,
        'eval_std': eval_std,
        }, path)

    def train(self):
        """Train the agent."""
        self.is_test = False
        step_count = 0
        
        for ep in tqdm(range(1, self.num_episodes)): 
            actor_losses, critic_losses, scores = [], [], []
            state, _ = self.env.reset()
            score = 0
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                state = next_state
                score += reward
                step_count += 1
                self.total_step += 1
                wandb.log({
                    "step": step_count,
                    "actor loss": actor_loss,
                    "critic loss": critic_loss,
                    }) 
                if done:
                    scores.append(score)
                    print(f"Episode {ep}: Total Reward = {score}")
                    wandb.log({
                        "episode": ep,
                        "return": score
                        })  


            if (not self.saved_pass_checkpoint) and (score >= -150.0):
                eval_mean, eval_std = self.evaluate(num_episodes=20, seeds=range(20))
                print(f"[AutoEval] ep={ep} 20-seed mean={eval_mean:.1f} ± {eval_std:.1f}")
                wandb.log({
                    "auto_eval/episode": ep,
                    "auto_eval/mean_return_0_19": eval_mean,
                    "auto_eval/std_return_0_19": eval_std,
                    "auto_eval/steps": self.total_step
                })
                if ( eval_mean >= self.best_eval_mean ) :
                    self.best_eval_mean = eval_mean
                    os.makedirs(self.ckpt_dir, exist_ok=True)
                    ckpt_path = os.path.join(self.ckpt_dir, "Task1_best_checkpoint.pt")
                    self._save_checkpoint(ckpt_path, self.total_step, eval_mean, eval_std)
                    print(f"Saved checkpoint → {ckpt_path} (mean={eval_mean:.1f})")
                if eval_mean >= -150.0:
                    os.makedirs(self.ckpt_dir, exist_ok=True)
                    ckpt_path = os.path.join(self.ckpt_dir, "pass_Task1_best_checkpoint.pt")
                    self._save_checkpoint(ckpt_path, self.total_step, eval_mean, eval_std)
                    self.saved_pass_checkpoint = True
                    print(f"Saved passing checkpoint → {ckpt_path} (mean={eval_mean:.1f})")

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
            score += reward

        print("score: ", score)
        self.env.close()

        self.env = tmp_env

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-a2c-run")
    parser.add_argument("--actor-lr", type=float, default=2e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=0) # entropy can be disabled by setting this to 0

    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ckpt-path", type=str, default="Task1_best_checkpoint.pt")
    parser.add_argument("--seeds", type=str, default="0-19")
    args = parser.parse_args()
    
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(project="DLP-Lab7-A2C-Pendulum", name=args.wandb_run_name, save_code=True)
    
    agent = A2CAgent(env, args)

    eval_seeds = parse_range(args.seeds)

    if args.eval_only:
        ckpt = torch.load(args.ckpt_path, map_location=agent.device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])

        ckpt_step = ckpt.get("step", ckpt.get("total_env_steps", None))
        if ckpt_step is not None:
            print(f"[EVAL] evaluating checkpoint at total_step={int(ckpt_step)}")
        else:
            print("[EVAL] evaluating checkpoint (total_step=unknown)")

        avg, std = agent.evaluate_verbose(eval_seeds)
        print(f"[EVAL DONE] avg={avg:.2f} ± {std:.2f}")
    else:
        agent.train()