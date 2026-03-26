# Deep Learning Lab 6: Conditional Image Generation with DDPM

---

## 📝 1. Introduction

In this lab, a **Denoising Diffusion Probabilistic Model (DDPM)** is implemented to perform conditional image generation on the CLEVR dataset. To jointly process time embeddings and condition embeddings, a **UNet** architecture is utilized and enhanced with the **CBAM** (Convolutional Block Attention Module)—a technique previously applied in Lab 2—to strengthen the model's focus on critical spatial and channel features.

**Training Objectives & Strategies:**
* The model is trained using **MSE loss** to regress the true noise.
* A comparison is made between **Linear** and **Cosine** noise schedules.
* To further improve generation quality and condition alignment, **Conditional Dropout** is incorporated during training, and **Classifier-Free Guidance (CFG)** is applied during sampling. This enables flexible control over the influence of conditions on the generated results.

---

## ⚙️ 2. Implementation Details

### i. Dataset Preparation (`CLEVRDataset`)
The dataset pipeline is designed to read `train.json` and `objects.json`. It loads each image along with its corresponding object labels and converts them into **one-hot vectors**. The images are then resized to `64x64` and normalized to the `[-1, 1]` range.

```python
class CLEVRDataset(Dataset):
    def __init__(self, data_root, train_json, objects_json, image_size=64):
        super().__init__()
        self.data_root = data_root
        self.mapping = load_objects(objects_json)
        self.num_classes = 24
        
        with open(train_json, encoding="utf-8") as f:
            self.meta = json.load(f)
            
        self.filenames = list(self.meta.keys())
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
```

### ii. Sampling & Denoising Process
During inference, pure Gaussian noise is initialized. The model then iteratively denoises this input by making reverse calls to the `p_sample` function, ultimately generating a clear image that satisfies the given input conditions.

---

## 📊 3. Results & Discussion

### Model Accuracy & Generalization
* **Accuracy on `test.json`**: **80.56%**
  The model successfully generates most samples. It demonstrates excellent generation quality under single-object conditions (where color and shape are mostly accurate). However, in complex multi-object scenarios, some objects exhibit minor color deviations or blurred shapes.
* **Accuracy on `new_test.json`**: **86.90%**
  The accuracy further improves on the new test set, indicating that the model possesses a strong generalization ability when facing novel and unseen condition combinations.

### Ablation Studies & Hyperparameter Analysis

1. **Noise Schedule (Linear vs. Cosine)**:
   The **Cosine schedule** converges faster, produces visually smoother generated images, and achieves slightly higher accuracy compared to the Linear schedule.

2. **Classifier-Free Guidance (CFG) Scale**:
   By adjusting the guidance scale, the following behaviors were observed:
   * **Scale < 2**: Leads to higher diversity but insufficient condition alignment (the model often ignores the prompt).
   * **Scale = 4**: Condition alignment is extremely strong, but the model tends to suffer from mode collapse (generating repetitive or unnatural images).
   * **Optimal Range**: The best balance between diversity and condition alignment is found when the CFG scale is between **2 and 3**.

3. **CBAM Attention Module**:
   After integrating the CBAM attention module into the UNet blocks, the model effectively learned to focus on essential features, resulting in much more stable colors and sharper shapes, especially in challenging multi-object scenarios.

</details>
