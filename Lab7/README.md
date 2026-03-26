

# Deep Learning Lab 7: Policy Gradient Methods (A2C & PPO)

---

## 📝 1. Introduction

In this experiment, three policy-gradient methods were implemented and compared across different reinforcement learning environments:

1. **A2C (Pendulum-v1)**: A two-layer MLP was used for both the actor and the critic. The actor was updated using stochastic policy gradients guided by TD errors, while the critic was trained with a regression loss.
2. **PPO-Clip + GAE (Pendulum-v1)**: A two-layer MLP was also applied, but with 64 hidden units instead of 128. Advantages and rewards were estimated using Generalized Advantage Estimation (GAE). Policy updates were performed with a clipped objective function using multiple epochs and minibatches.
3. **PPO-Clip (Walker2d-v4)**: The same network architecture as Task 2 was utilized and extended to a much more complex, high-dimensional continuous control task.

**Key Findings:**
On `Pendulum-v1`, PPO-Clip + GAE converged significantly faster and demonstrated more stable learning dynamics compared to A2C, proving that objective clipping and GAE effectively improve stability and sample efficiency. On `Walker2d-v4`, performance was highly sensitive to the clipping parameter and entropy coefficient, with moderate values achieving the best average returns.

---

## ⚙️ 2. Implementation Details

### Stochastic Policy Gradient & Action Sampling
For continuous action spaces, the action is sampled and scaled properly. The implementation for obtaining the action distribution and computing the stochastic policy gradient in A2C is as follows:

```python
state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
dist = self.actor(state)

# Sample action during training; use mean during testing
raw = dist.mean if self.is_test else dist.rsample()
tanh_a = torch.tanh(raw)

# Scale action to the environment's acceptable range (e.g., [-2.0, 2.0])
action = 2.0 * tanh_a
```

---

## 📊 3. Results & Discussion

### i. Hyperparameter Tuning Analysis
* **Clipping Parameter (PPO)**:
  An excessively large clipping parameter caused instability and massive oscillations in the returns. Overall, a **clipping parameter of 0.2** provided the optimal balance, maintaining both fast convergence speed and training stability.
* **Entropy Coefficient**:
  * `0.0`: The policy quickly converged to a narrow distribution, leading to insufficient exploration and poor performance.
  * `0.01`: Achieved an excellent balance between exploration and exploitation, resulting in the most stable return curve and the best overall performance.
  * `0.1`: Made the policy overly stochastic, causing larger fluctuations in the learning curve and unstable training.

### ii. Method Comparisons
* **Sample Efficiency**: 
  For PPO (Tasks 2 and 3), near-optimal performance was successfully achieved within 200k to 300k steps. In stark contrast, A2C (Task 1) continued to oscillate heavily even after 600k training steps.
* **Training Stability**: 
  * **A2C**: Exhibited extremely high variance, with the learning curve showing massive oscillations.
  * **PPO**: Displayed a much smoother learning curve. Although minor fluctuations remained, overall stability was significantly improved.
* **Scalability**: 
  * **A2C**: Almost failed to achieve positive returns in the complex `Walker2d` environment.
  * **PPO**: Successfully converged in `Walker2d`, demonstrating strong adaptability to high-dimensional continuous action spaces.

### iii. Additional Training Strategies
* **Reward Normalization**: 
  Reward normalization was applied during certain experiments. This technique effectively reduced the variance of returns across different episodes, which in turn mitigated the high variance of TD errors and helped stabilize the critic's learning process.

</details>
