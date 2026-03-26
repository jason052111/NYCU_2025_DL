# 🚀 Deep Learning Course Assignments (Lab 1 - Lab 7)

Welcome to my Deep Learning repository! This workspace contains a complete series of hands-on projects, ranging from building foundational neural networks from scratch to implementing advanced Generative AI and Reinforcement Learning models.

## 🌟 My Journey & Key Learnings

Throughout these seven comprehensive labs, I progressively built a deep understanding of modern artificial intelligence architectures:

1. **Foundations of Optimization**: I started by manually deriving and implementing Backpropagation and Gradient Descent without relying on auto-grad frameworks. This solidified my mathematical understanding of how neural networks truly learn.
2. **Computer Vision & Attention**: I explored dense prediction tasks (Semantic Segmentation) and learned how to leverage Spatial and Channel Attention (CBAM) to force models to focus on critical features.
3. **Generative AI Masterclass**: I heavily researched and implemented state-of-the-art generative models. I transitioned from discrete token space generation (VQGAN + MaskGIT) and Conditional VAEs for video prediction, all the way to continuous conditional generation using Denoising Diffusion Probabilistic Models (DDPM) with Classifier-Free Guidance.
4. **Reinforcement Learning (RL)**: I tackled the challenge of decision-making agents, progressing from Value-based methods (Vanilla DQN, Double DQN, PER) to advanced Policy-Gradient methods (A2C, PPO with GAE). I learned how to stabilize training and balance the exploration-exploitation trade-off in complex, continuous action spaces.

---

## 📂 Projects Overview

| Lab | Topic | Core Technologies & Architectures |
| :---: | :--- | :--- |
| **1** | Neural Network from Scratch | MLP, Backpropagation, Activation Functions, MSE |
| **2** | Semantic Segmentation | U-Net, ResNet34, CBAM, Dice Loss |
| **3** | Image Inpainting | VQGAN, MaskGIT, Bidirectional Transformer, MVTM |
| **4** | Video Prediction | Conditional VAE (CVAE), Teacher Forcing, KL Annealing |
| **5** | Value-Based RL | DQN, Double DQN, Prioritized Experience Replay (PER) |
| **6** | Conditional Generation | DDPM (Diffusion), UNet, Classifier-Free Guidance |
| **7** | Policy-Based RL | A2C, PPO-Clip, Generalized Advantage Estimation (GAE) |

---

## 🔍 Detailed Lab Reports

<details>
<summary><b>👉 Lab 1: Two-Layer Neural Network</b></summary>

* **Objective**: Build a two-layer neural network from scratch to understand the math behind deep learning.
* **Implementation**: Forward pass applying linear transformations ($z = xW + b$) and activation functions (Sigmoid, ReLU, Tanh). Implemented manual Backpropagation using the Chain Rule to update weights and biases based on Mean Squared Error (MSE).
* **Key Takeaway**: Achieved 100% accuracy on both Linear and XOR datasets, demonstrating a profound understanding of how gradients flow through a network.

</details>

<details>
<summary><b>👉 Lab 2: Semantic Segmentation</b></summary>

* **Objective**: Accurately segment foreground objects (pets) from backgrounds.
* **Implementation**: Developed **U-Net** and **ResNet34+U-Net** models. Integrated **CBAM** (Convolutional Block Attention Module) to enhance spatial and channel feature extraction. Utilized a hybrid loss function combining `BCELoss` and `Dice Loss`.
* **Key Takeaway**: Mastered the use of skip connections to prevent vanishing gradients and proved that attention mechanisms significantly boost segmentation accuracy (Dice score: 0.9327).

</details>

<details>
<summary><b>👉 Lab 3: Image Inpainting with VQGAN and MaskGIT</b></summary>

* **Objective**: Generate semantically consistent content to fill missing regions in images.
* **Implementation**: Used VQGAN to encode images into discrete tokens. Built a custom Multi-Head Self-Attention Transformer to perform parallel predictions. Implemented Masked Visual Token Modeling (MVTM) with Iterative Decoding using Cosine, Linear, and Square mask scheduling.
* **Key Takeaway**: Achieved an impressive FID score of 37.30. Learned how to balance inference speed and generation quality using non-autoregressive iterative decoding.

</details>

<details>
<summary><b>👉 Lab 4: Video Prediction with CVAE</b></summary>

* **Objective**: Predict future video frames conditioned on skeleton pose maps.
* **Implementation**: Built a Conditional Variational Autoencoder (CVAE). Handled the latent space using the Reparameterization Trick. Implemented a dynamic Teacher Forcing Ratio and mitigated the KL Vanishing problem using Monotonic and Cyclical KL Annealing scheduling.
* **Key Takeaway**: Successfully handled sequence modeling challenges. Discovered how auto-regressive error accumulation impacts training and how cyclical annealing prevents the model from ignoring the latent distribution.

</details>

<details>
<summary><b>👉 Lab 5: Deep Q-Network (DQN)</b></summary>

* **Objective**: Train agents to solve CartPole and Atari environments using Value-based RL.
* **Implementation**: Evolved from a Vanilla DQN to a highly optimized agent featuring Double DQN, Dueling Network architecture, Multi-Step Returns, and Prioritized Experience Replay (PER).
* **Key Takeaway**: Demonstrated how advanced RL enhancements dramatically improve sample efficiency and prevent the overestimation bias inherent in standard Q-learning.

</details>

<details>
<summary><b>👉 Lab 6: Conditional Image Generation with DDPM</b></summary>

* **Objective**: Generate complex, multi-object images based on specific conditions using Diffusion models.
* **Implementation**: Designed a UNet architecture enhanced with CBAM to process time and condition embeddings jointly. Compared Linear and Cosine noise schedules. Implemented Classifier-Free Guidance (CFG) during sampling to balance diversity and fidelity.
* **Key Takeaway**: Achieved an 86.9% accuracy on novel test sets. Deeply understood the forward noise-adding process and the reverse denoising phase, mastering how CFG scales control prompt alignment.

</details>

<details>
<summary><b>👉 Lab 7: Policy Gradient Methods (A2C & PPO)</b></summary>

* **Objective**: Solve high-dimensional continuous control tasks (Pendulum-v1, Walker2d-v4) using Policy-based RL.
* **Implementation**: Built an Actor-Critic architecture. Upgraded from standard A2C to Proximal Policy Optimization (PPO). Utilized Generalized Advantage Estimation (GAE) to reduce variance and applied objective clipping to constrain policy updates.
* **Key Takeaway**: Discovered the fragility of basic policy gradients. Proved that PPO's clipping mechanism and GAE provide a massive leap in training stability and scalability for complex robotics/control tasks.

</details>

---

## 🛠️ Tech Stack & Tools
* **Deep Learning Framework**: PyTorch
* **Computer Vision**: Torchvision, PIL, Image Inpainting, Semantic Segmentation
* **Generative AI**: VAE, VQGAN, MaskGIT, DDPM (Diffusion)
* **Reinforcement Learning**: OpenAI Gym / Gymnasium, DQN, PPO, A2C
* **Experiment Tracking**: Weights & Biases (Wandb)
