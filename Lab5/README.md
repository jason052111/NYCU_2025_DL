# Deep Learning Lab 5: Deep Q-Network (DQN)

---

## 📝 1. Introduction

This lab consists of three main tasks focusing on Reinforcement Learning using Deep Q-Networks (DQN):
* **Task 1**: Uses Vanilla DQN to train the CartPole environment.
* **Task 2**: Uses Vanilla DQN to train an Atari environment.
* **Task 3**: Applies advanced techniques including Double DQN, Prioritized Experience Replay (PER), and Multi-Step Return to train the Atari environment.

**Network Architecture:**
Two versions of the main DQN network were implemented:
1. **Without Convolutional Layers**: Used for CartPole, as its input is a low-dimensional numerical vector rather than a stack of images.
2. **With Convolutional Layers**: Used for Atari, since its input consists of a stack of image frames.

**Optimization & Loss:**
* The final loss is computed using **Huber Loss**.
* **RMSprop** is used as the optimizer to improve training stability.
* A **linear decay schedule** is adopted for the epsilon-greedy exploration strategy.

---

## ⚙️ 2. Implementation Details

### TD Target Formulation
For Task 1 and Task 2 (Vanilla DQN), the TD target is defined as:
`Q(s, a) = r + γ(1 - done) * max_a' Q_target(s', a')`

For Task 3 (Double DQN), the TD target is defined as:
`Q(s, a) = r + γ^n(1 - done) * Q_target(s', argmax_a Q_online(s', a))`

### Bellman Error Calculation
The calculation of the Bellman error and the TD target is implemented as follows:

```python
with torch.no_grad():
    if self.use_double_dqn:
        a_star = self.q_net(next_states).argmax(dim=1, keepdim=True)
        next_q_max = self.target_net(next_states).gather(1, a_star).squeeze(1)
    else:
        next_q_max = self.target_net(next_states).max(dim=1).values
        
    targets = rewards + gammas * (1.0 - dones) * next_q_max

td_error = values - targets
```

---

## 📊 3. Results & Discussion

### Sample Efficiency Analysis (Task 2 vs. Task 3)
* **Task 2 (Vanilla DQN)**: Required a significantly longer training time (approximately 10 million frames) to escape the low-reward range (around -20) and slowly climb to positive rewards (10+). This indicates substantially lower learning efficiency.
* **Task 3 (Enhanced DQN)**: Achieved positive rewards with relatively fewer training samples. The integration of DQN enhancements—such as Double DQN, Dueling Network architectures, and Prioritized Experience Replay (PER)—effectively and noticeably improved sample efficiency.

---

## 🚀 4. Execution Commands

**Run Task 1 (CartPole):**
```bash
python dqn.py --wandb-run-name cartpole-run --replay-start-size 1000 --target-update-frequency 500 --epsilon-decay 0.9995 --lr 0.001 --batch-size 64 --discount-factor 0.98
```

**Run Task 2 (Atari):**
```bash
python dqn.py --wandb-run-name ale-run --memory-size 100000 --replay-start-size 50000 --target-update-frequency 10000
```

</details>
