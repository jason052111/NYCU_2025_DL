# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, env_name="CartPole-v1", num_actions = 4):
        super(DQN, self).__init__()
        if (env_name == "CartPole-v1") :
            self.network = nn.Sequential(
                nn.Linear(4, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, num_actions)
            )       
        elif (env_name == "ALE/Pong-v5") :
            self.network = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )   

    def forward(self, x):
        return self.network(x)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.buffer = [None] * capacity
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def maxlen(self):
        return self.capacity

    def add(self, transition, error):
        if error is None:
            max_prio = self.priorities.max() if self.size > 0 else 1.0
        else:
            max_prio = abs(error) + self.eps
        self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=None):
        if beta is None:
            beta = self.beta

        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs = probs / probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        N = self.size
        weights = (N * probs[indices]) ** (-beta)
        weights = weights / weights.max() 
        return indices, samples, weights.astype(np.float32)

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            self.priorities[idx] = abs(float(err)) + self.eps



class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env_name = env_name
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        if env_name == "CartPole-v1":
            self.preprocessor = None 
        else:
            self.preprocessor = AtariPreprocessor()


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)


        self.q_net = DQN(env_name, self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(env_name, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        if self.env_name=="CartPole-v1":
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=args.lr)
        elif (self.env_name=="ALE/Pong-v5"):
            self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=2.5e-4, alpha=0.95, eps=1e-5)

        self.use_double_dqn = args.use_double_dqn
        self.use_per = args.use_per
        self.n_step = args.n_step
        self.gamma = args.discount_factor
        if self.use_per:
            self.per_alpha = args.per_alpha
            self.per_beta0 = args.per_beta0
            self.per_beta_anneal_steps = args.per_beta_anneal_steps
            self.per_beta = self.per_beta0
            self.memory = PrioritizedReplayBuffer(args.memory_size, alpha=self.per_alpha, beta=self.per_beta0)
        else:
            self.memory = deque(maxlen=args.memory_size)

        if self.n_step > 1:
            self.n_step_buffer = deque(maxlen=self.n_step)

        self.batch_size = args.batch_size
        self.epsilon = args.epsilon_start
        self.epsilon_start = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay_steps = args.epsilon_decay_steps

        self.env_count = 0
        self.train_count = 0
        self.best_reward = float("-inf")  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def get_n_step_transition(self):
        total_reward = 0.0
        gamma = 1.0
        for (_, _, reward, _, done) in self.n_step_buffer:
            total_reward += gamma * reward
            gamma *= self.gamma
            if done:
                break
        state_first, action_first, _, _, _ = self.n_step_buffer[0]
        _, _, _, next_state_last, done_last = self.n_step_buffer[-1]
        return (state_first, action_first, total_reward, next_state_last, done_last, gamma) 

    def store_transition(self, transition):
        if self.use_per:
            self.memory.add(transition, error=None)
        else:
            self.memory.append(transition)

    def mem_len(self):
        if not self.use_per :
            return len(self.memory) 
        else :
            return self.memory.size

    def mem_max_len(self):
        if not self.use_per :
            return self.memory.maxlen 
        else :
            return self.memory.capacity

    def _fire_if_needed(self, env):
        try:
            meanings = env.unwrapped.get_action_meanings()
            if "FIRE" in meanings:
                    a_fire = meanings.index("FIRE")
                    obs, _, _, _, _ = env.step(a_fire)
                    return obs
        except Exception:
            return None

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            if (self.env_name=="ALE/Pong-v5"):
                state_tensor = state_tensor / 255.0
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset()

            if (self.env_name=="CartPole-v1") :
                state = np.asarray(obs, dtype=np.float32)
            elif (self.env_name=="ALE/Pong-v5"):
                fired = self._fire_if_needed(self.env)
                if fired is not None :
                    obs = fired
                state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                if (self.env_name=="CartPole-v1") :
                    next_state = np.asarray(next_obs, dtype=np.float32)
                elif (self.env_name=="ALE/Pong-v5"):
                    next_state = self.preprocessor.step(next_obs)

                if self.n_step > 1:
                    self.n_step_buffer.append((state, action, reward, next_state, done))
                    if len(self.n_step_buffer) == self.n_step or done:
                        s0, a0, Rn, sN, dN, gamma_n = self.get_n_step_transition()
                        self.store_transition((s0, a0, Rn, sN, dN, gamma_n))
                        if not done:
                            self.n_step_buffer.popleft()
                else:
                    self.store_transition((state, action, reward, next_state, done, self.gamma))

                if self.env_count % 1000 == 0:
                    print(f"[Buffer] Current size: {self.mem_len()} / {self.mem_max_len()}")

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count == 400000 or self.env_count == 800000 or self.env_count == 1200000 or self.env_count == 1600000 or self.env_count == 2000000 :
                    save_path = os.path.join(self.save_dir, f"model_{self.env_count}steps.pt")
                    torch.save(self.q_net.state_dict(), save_path)
                    print(f"[Snapshot] Saved model at {self.env_count} steps -> {save_path}")
                
                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })

            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })

            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        if (self.env_name=="CartPole-v1") :
            state = np.asarray(obs, dtype=np.float32)
        elif (self.env_name=="ALE/Pong-v5"):
            fired = self._fire_if_needed(self.test_env)
            if fired is not None :
                obs = fired
            state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0
         
        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            if (self.env_name=="ALE/Pong-v5"):
                state_tensor = state_tensor / 255.0
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            if (self.env_name=="CartPole-v1") :
                state = np.asarray(next_obs, dtype=np.float32)
            elif (self.env_name=="ALE/Pong-v5"):
                state = self.preprocessor.step(next_obs)

        return total_reward


    def train(self):

        if self.mem_len() < self.replay_start_size:
            return
        if self.epsilon > self.epsilon_min:
            if (self.env_name=="ALE/Pong-v5"):
                decay_ratio = min(1.0, self.env_count / self.epsilon_decay_steps)
                self.epsilon = self.epsilon_start - decay_ratio * (self.epsilon_start - self.epsilon_min)
            elif (self.env_name=="CartPole-v1") :
                self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        if self.use_per:
            if self.per_beta_anneal_steps > 0:
                self.per_beta = min(1.0, self.per_beta + (1.0 - self.per_beta0) / self.per_beta_anneal_steps)
            indices, batch, is_weights = self.memory.sample(self.batch_size, beta=self.per_beta)
            is_weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device)
        else:
            batch = random.sample(self.memory, self.batch_size)
            indices = None
            is_weights = None
        states, actions, rewards, next_states, dones, gammas = zip(*batch)   

        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        gammas      = torch.tensor(gammas,   dtype=torch.float32, device=self.device)  # gamma^n
        if (self.env_name=="ALE/Pong-v5"):
            states = states / 255.0
            next_states = next_states / 255.0

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.use_double_dqn:
                a_star = self.q_net(next_states).argmax(dim=1, keepdim=True)
                next_q_max = self.target_net(next_states).gather(1, a_star).squeeze(1)
            else:
                next_q_max = self.target_net(next_states).max(dim=1).values

            targets = rewards + gammas * (1.0 - dones) * next_q_max  # r + γ(1-done)max_a' Q_target(s',a')

        td_error = q_values - targets
        if is_weights is not None:
            loss = (is_weights * torch.nn.functional.smooth_l1_loss(q_values, targets, reduction="none")).mean()
        else:
            loss = torch.nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        if self.train_count % 1000 == 0:
            wandb.log({
                "Loss": loss.item(),
                "buffer_len": self.mem_len(),
                "use_per": int(self.use_per),
                "use_ddqn": int(self.use_double_dqn),
                "n_step": self.n_step,
                "per_beta": getattr(self, "per_beta", 0.0),
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
            })

        if self.use_per:
            td_abs = td_error.detach().abs().cpu().numpy()
            self.memory.update_priorities(indices, td_abs)

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=500000)
    parser.add_argument("--target-update-frequency", type=int, default=10000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--use-double-dqn", action="store_true")
    parser.add_argument("--use-per", action="store_true")
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta0", type=float, default=0.4)
    parser.add_argument("--per-beta-anneal-steps", type=int, default=1000000)  
    parser.add_argument("--n-step", type=int, default=1, help="n-step return; set >1 to enable")
    args = parser.parse_args()


    if (args.wandb_run_name=="cartpole-run") :
        wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True)
        agent = DQNAgent(env_name="CartPole-v1", args=args)
    elif (args.wandb_run_name=="ale-run") :
        wandb.init(project="DLP-Lab5-DQN-ALE", name=args.wandb_run_name, save_code=True)
        agent = DQNAgent(env_name="ALE/Pong-v5", args=args)
    agent.run()