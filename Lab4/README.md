# Deep Learning Lab 4: Video Prediction with CVAE

---

## 📝 1. Introduction

This assignment implements a **Conditional Variational Autoencoder (CVAE)** for video frame prediction. The primary distinction between this model and a standard VAE is the incorporation of a "Condition" (e.g., a skeleton pose map) to guide the prediction of future frames.

**Core Prediction Workflow:**
1. Encode the current frame (RGB) and its corresponding skeleton map (Label).
2. Pass the encoded features through the **Gaussian Predictor** to calculate the mean (`mu`) and log-variance (`logvar`).
3. Apply the **Reparameterization Trick** to sample a latent vector `z` from the Gaussian distribution.
4. Fuse this latent vector `z` with the encoded RGB and Label features in the **Decoder**.
5. Feed the fused parameters into the **Generator** to output the predicted next frame.

The main forward pass workflow is implemented as follows:

```python
f_rgb = self.frame_transformation(frame)
f_label = self.label_transformation(label[:, i])
z, mu, logvar = self.Gaussian_Predictor(f_rgb, f_label)
parm = self.Decoder_Fusion(f_rgb, f_label, z)
out = self.Generator(parm)
```

---

## ⚙️ 2. Implementation Details

### i. Training Protocol (`training_one_step`)
* **Teacher Forcing Strategy**: A Teacher Forcing strategy is employed during training. When Teacher Forcing is active, the ground-truth previous frame is used as the input. When inactive, a mixture of the model's self-generated previous frame and the ground-truth frame is used.
* **Loss Function**: The total loss is a combination of Reconstruction Loss (MSE + L1 Loss, included to better preserve image details) and KL Divergence Loss. The reconstruction loss ratio is set to `0.6 * MSE + 0.4 * L1`. A dynamic `beta` parameter controls the weight of the KL Loss.

### ii. KL Annealing Scheduling (`frange_cycle_linear`)
To prevent the KL Vanishing problem, different `beta` scheduling strategies are implemented:
* **Monotonic**: The `beta` value gradually increases from 0 to 1 over the epochs.
* **None**: The `beta` value remains permanently fixed at 1.
* **Cyclical**: This is implemented in the `frange_cycle_linear` function. It defines a `cycle length` for one period. During the increasing phase of a cycle, `beta` is calculated as `start_value + current_step * increment_per_step`. If the calculated `beta <= 0`, it is clamped to a very small positive number to prevent completely ignoring the KL Loss when beta is zero.

### iii. Key Functions

**Reparameterization Trick**:
```python
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
```

**Teacher Forcing Ratio Update**:
```python
def teacher_forcing_ratio_update(self):
    if self.current_epoch > self.tfr_sde:
        self.tfr = max(0.0, self.tfr - self.tfr_d_step)
```

---

## 📊 3. Analysis & Discussion

### Impact of Teacher Forcing Ratio on the Loss Curve

Observing the training process, the behavior of the Loss curve can be divided into three distinct stages based on the changes in the Teacher Forcing Ratio (TFR):

1. **Early Stage (High TFR)**
   The Loss curve drops rapidly. Because Teacher Forcing provides stable, ground-truth previous frames as inputs, the impact of error accumulation is minimized, allowing the model to learn efficiently in the beginning.
2. **Mid Stage (Decaying TFR)**
   As the Teacher Forcing Ratio begins to decrease, the model is forced to increasingly rely on its own generated outputs. The rate of Loss reduction slows down, and minor oscillations or slight spikes may occur during this transition. This happens because prediction errors are amplified during the auto-regressive generation process.
3. **Late Stage (TFR = 0)**
   The Teacher Forcing Ratio has dropped to 0, and the model enters a fully auto-regressive prediction mode. At this point, the Loss curve tends to flatten and stabilize, representing the model's inherent prediction capability without external correction.

</details>
