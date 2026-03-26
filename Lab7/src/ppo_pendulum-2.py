#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 2: PPO-Clip
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import random
from collections import deque
from typing import Deque, List, Tuple

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
import os, shutil

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
        self.mean_layer = init_layer_uniform(nn.Linear(64, out_dim))
        self.log_std_layer = init_layer_uniform(nn.Linear(64, out_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

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
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value_layer = init_layer_uniform(nn.Linear(64, 1))
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value_layer(x)
        #############################

        return value
    
def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float) -> List:
    """Compute gae."""

    ############TODO#############
    values = values + [next_value]
    gae = torch.zeros_like(next_value)
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
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        self.obs_dim = obs_dim
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

        self.best_score = -1e9  # or very small
        self.checkpoint_path = "LAB7_314551147_task2_ppo_pendulum.pt"

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            value = self.critic(state)
            self.states.append(state)
            self.actions.append(selected_action)
            self.values.append(value)
            self.log_probs.append(dist.log_prob(selected_action))

        return selected_action.cpu().detach().numpy()

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
        actions = torch.cat(self.actions)
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
            # actor_loss
            ############TODO#############

            log_prob_new = dist.log_prob(action.detach())
            if log_prob_new.dim() > 1:
                log_prob_new = log_prob_new.sum(dim=-1, keepdim=True)

            if old_log_prob.dim() > 1:
                old_log_prob = old_log_prob.sum(dim=-1, keepdim=True)

            ratio = (log_prob_new - old_log_prob).exp()

            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            eps = float(self.epsilon) if hasattr(self, "epsilon") else 0.2
            entropy_coef = float(self.entropy_weight) if hasattr(self, "entropy_weight") else 1e-2

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * adv

            entropy = dist.entropy()
            if entropy.dim() > 1:
                entropy = entropy.sum(dim=-1)
            entropy = entropy.mean()

            actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy

            #############################

            # critic_loss
            ############TODO#############

            critic_loss = F.mse_loss(self.critic(state), return_)
            #############################
            
            # train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
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
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward[0][0]

                if done[0][0]:
                    episode_count += 1
                    state, _ = self.env.reset()
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    print(f"Episode {episode_count}: Total Reward = {score}")
                    wandb.log({
                        "train/episode_return": float(scores[-1]),
                        "train/episode": int(episode_count),
                        "env/total_steps": int(self.total_step)
                    }, step=int(self.total_step))
                    if self.total_step >= 100000 and score > -150 :
                        avg = self.evaluate_20_episodes()
                        print(f"[Eval@save] 20-seed avg = {avg:.2f}")
                        if avg >= -150.0:
                            self.best_score = score
                            torch.save({
                                "actor": self.actor.state_dict(),
                                "total_env_steps": int(self.total_step),
                                "seed": int(self.seed)
                            }, self.checkpoint_path)
                            print(f"[Checkpoint] Saved best actor to {self.checkpoint_path} "
                                f"(best_ep={self.best_score:.2f}, steps={self.total_step})")

                    score = 0

            actor_loss, critic_loss = self.update_model(next_state)
            wandb.log({
                "loss/actor": float(actor_loss),
                "loss/critic": float(critic_loss),
                "env/total_steps": int(self.total_step)
            }, step=int(self.total_step))
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

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
            score += reward

        print("score: ", score)
        self.env.close()

        self.env = tmp_env

    def load_actor(self, path):
        blob = torch.load(path, map_location=self.device)

        if isinstance(blob, dict) and "actor" in blob:
            self.actor.load_state_dict(blob["actor"])
            self.ckpt_steps = int(blob.get("step", blob.get("total_env_steps", -1)))
        else:
            self.actor.load_state_dict(blob)
            self.ckpt_steps = -1

        self.actor.eval()

    def evaluate_20_episodes(self):
        prev_mode = self.is_test
        self.is_test = True

        actor_was_training = self.actor.training
        critic_was_training = self.critic.training
        self.actor.eval(); self.critic.eval()

        seeds = list(range(20))  # 0-19
        scores = []

        # 標頭（依你要的格式）
        steps_note = getattr(self, "ckpt_steps", -1)
        if steps_note is not None and steps_note > 0:
            print(f"[EVAL] evaluating checkpoint at total_step={int(steps_note)}")
        else:
            print("[EVAL] evaluating checkpoint at total_step=unknown")

        with torch.no_grad():
            for s in seeds:
                state, _ = self.env.reset(seed=s)
                state = np.expand_dims(state, axis=0)
                done = False
                total = 0.0
                while not done:
                    action = self.select_action(state)
                    next_state, reward, done = self.step(action)
                    state = next_state
                    total += float(reward[0][0])
                scores.append(total)
                print(f"[EVAL] seed={s}, return={total:.2f}")
                wandb.log({
                    "eval/seed_reward": float(total),
                    "eval/seed": int(s),
                    "env/total_steps": int(steps_note if steps_note > 0 else self.total_step)
                }, step=int(steps_note if steps_note > 0 else self.total_step))

        avg = float(np.mean(scores))
        std = float(np.std(scores))
        print(f"[EVAL] seeds {seeds[0]}-{seeds[-1]} | avg={avg:.2f} ± {std:.2f}")
        print(f"[EVAL DONE] avg={avg:.2f} ± {std:.2f}")

        wandb.log({
            "eval/avg_reward": avg,
            "eval/std_reward": std,
            "env/total_steps": int(steps_note if steps_note > 0 else self.total_step)
        }, step=int(steps_note if steps_note > 0 else self.total_step))

        if actor_was_training: self.actor.train()
        if critic_was_training: self.critic.train()
        self.is_test = prev_mode
        return avg

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-ppo-run")
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=int, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=2000)  
    parser.add_argument("--update-epoch", type=int, default=64)
    parser.add_argument("--mode", choices=["train","eval"], default="train")
    parser.add_argument("--model-path", type=str, default="LAB7_314551147_task2_ppo_pendulum.pt")

    args = parser.parse_args()
 
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(project="DLP-Lab7-PPO-Pendulum", name=args.wandb_run_name, save_code=True)
    wandb.config.update(vars(args))
    wandb.define_metric("env/total_steps")
    wandb.define_metric("train/*", step_metric="env/total_steps")
    wandb.define_metric("loss/*",  step_metric="env/total_steps")
    wandb.define_metric("eval/*",  step_metric="env/total_steps")

    agent = PPOAgent(env, args)
    if args.mode == "train":
        agent.train()
    else:  # eval
        assert args.model_path is not None, "Please set --model-path"
        agent.load_actor(args.model_path)
        agent.evaluate_20_episodes()