
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import imageio

class DQN(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        # Must match the training architecture used for CartPole
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
    def forward(self, x):
        return self.network(x)

def evaluate(model_path: str, episodes: int = 10, seed: int = 0, output_dir: str = "./eval_videos", fps: int = 30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    num_actions = env.action_space.n

    # Build model and load weights
    model = DQN(num_actions).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        frames = []

        while not done:
            # Render frame for video
            frame = env.render()
            frames.append(frame)

            state = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                q = model(state)
                action = int(torch.argmax(q, dim=1).item())

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

        # Write video
        out_path = os.path.join(output_dir, f"cartpole_ep{ep}_R{int(total_reward)}.mp4")
        with imageio.get_writer(out_path, fps=fps) as writer:
            for fr in frames:
                writer.append_data(fr)
        print(f"[Eval] ep={ep} total_reward={total_reward:.1f} → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to CartPole .pt checkpoint")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="./eval_videos_cartpole")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    evaluate(args.model_path, episodes=args.episodes, seed=args.seed, output_dir=args.output_dir, fps=args.fps)
