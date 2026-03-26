# Deep Learning Lab 3: Image Inpainting with VQGAN and MaskGIT

## 📝 Project Overview
This project focuses on **image inpainting**, where the model takes an image with missing regions as input and generates semantically consistent and visually natural content to fill them in. 

The approach adopts a combination of **VQGAN and MaskGIT**:
* **VQGAN**: Encodes the image into the discrete token space of a predefined codebook.
* **MaskGIT**: Utilizes a bidirectional Transformer to perform parallel predictions over the token sequence. 

Through Masked Visual Token Modeling (MVTM), the inpainting process is carried out progressively from easy to hard. Compared to traditional autoregressive token-by-token generation, MaskGIT's iterative decoding updates multiple unknown tokens simultaneously at each step, achieving a superior balance between generation speed and image quality.

## ⚙️ Implementation Details

### 1. Multi-Head Self-Attention
A custom Multi-Head Self-Attention module was implemented for the Transformer architecture. 
* **Key Parameters**: `dim=768`, `num_heads=16`, `attn_drop=0.1`.
* The module computes Queries, Keys, and Values (QKV) through linear transformations, applies scaling, and utilizes dropout for regularization.

### 2. MaskGIT Architecture
* **Embedding**: Uses `nn.Embedding(codebook_size, embed_dim)` to map discrete tokens into dense vectors, combined with absolute positional embeddings.
* **Masking**: Missing regions are replaced with a special mask token.
* **Transformer Blocks**: The token sequence is fed into multiple Transformer blocks, each containing Multi-Head Self-Attention and an MLP layer, complete with LayerNorm and Dropout.

### 3. Training and Iterative Decoding
* **Training**: The model forward pass outputs predicted tokens, and the network is optimized using `CrossEntropyLoss`.
* **Iterative Decoding (Inference)**: 
  1. Starts with all missing regions represented as mask tokens.
  2. Predicts probability distributions and computes confidence scores for the mask tokens.
  3. Retains the top-$k$ tokens with the highest confidence, leaving the rest masked for the next iteration.
  4. Gumbel noise is introduced during the process to add beneficial stochasticity to the generation.

## 📊 Experimental Results

### Training Convergence
During training, the CrossEntropy Loss successfully converged, dropping steadily from approximately 8.0 down to around 5.3 over 100 epochs.

### Mask Scheduling Methods Comparison
Three different scheduling functions were tested to determine how the mask ratio should decrease during iterative decoding:

* 📉 **Cosine Scheduling**: 
  * *Mechanism*: Decreases smoothly, retaining more masks in the early stages and gradually reducing them later.
  * *Result*: The reconstruction process is the smoothest, with the image gradually becoming clearer step by step.
* 📉 **Linear Scheduling**: 
  * *Mechanism*: The number of masks decreases linearly, removing a constant proportion at each iteration.
  * *Result*: Convergence speed is fast, allowing a relatively complete image to be visible by the mid-stage of decoding.
* 📉 **Square Scheduling**: 
  * *Mechanism*: The mask ratio drops drastically at the beginning, but the decrease slows down significantly in the later stages.
  * *Result*: Early-stage generation performs exceptionally well, though the visual improvement becomes smaller toward the end of the process.

### Best Performance Evaluation
* **Best FID Score**: **37.3001**
* Achieved using the **Square** scheduling method during inference.
