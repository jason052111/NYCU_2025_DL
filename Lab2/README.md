# Deep Learning Lab 2: Semantic Segmentation

## 📝 Project Overview
This project implements U-Net and ResNet34+U-Net models for Semantic Segmentation. To enhance the models' ability to capture foreground objects, both architectures integrate the CBAM (Convolutional Block Attention Module), including Channel Attention and Spatial Attention mechanisms.

## ⚙️ Implementation Details

### 1. Model Architecture
* **U-Net**: 
    * Incorporates residual connections to mitigate the vanishing gradient problem in deep networks.
    * Attention mechanisms are added before and after two consecutive `3x3` convolutions, followed by feature map normalization after each convolution.
    * Downsampling layers use `2x2` Max Pooling. Upsampling layers utilize `ConvTranspose2d` to restore feature map resolution and apply Skip Connections using the corresponding feature maps from the downsampling path.
* **ResNet34 + U-Net**: 
    * Utilizes ResNet's BasicBlock, enhanced with attention mechanisms before and after convolutions.
    * The downsampling process starts with a `7x7` convolution and MaxPool (halving the resolution), followed by BasicBlocks in the sequence of `3, 4, 6, 3`. The final feature map transforms from `256x256 (Channel: 3)` to `8x8 (Channel: 512)`.
    * The upsampling part adopts the U-Net Decoder concept but replaces `ConvTranspose2d` with `F.interpolate` for parameter-free resolution restoration.

### 2. Loss Function & Metrics
* **Loss Function**: Uses a hybrid loss function of `BCELoss + (1 - Dice score)` to balance the training process, ensuring the model focuses not only on the background but also accurately on foreground objects.
* **Dice Score Calculation**: When calculating the intersection (A∩B), it only considers pixels where both the Prediction and Ground Truth are 1. An `eps` value is added to prevent division by zero, yielding the formula `(2*|A∩B|+eps) / (|A|+|B|+eps)`.

## 📊 Data Preprocessing & Augmentation
* **Data Preprocessing**: All images are resized to a uniform `256x256` resolution, and pixel values are normalized to ensure more stable weight updates during forward and backward propagation.
* **Data Augmentation**: 50% Horizontal Flip and 10% Gaussian Noise are applied during training. Vertical flips and 90-degree rotations were tested during experiments but were discarded as they provided limited improvements to the Dice score.

## 🏆 Experimental Results
* Due to hardware limitations (8GB VRAM), the training batch size was set to 8.
* The models achieved initial convergence at around 100 epochs and reached their best performance at approximately 160 epochs.

| Model | Average Dice Score (Test Set) |
| :--- | :---: |
| **U-Net** | **0.9327** |
| **ResNet34 + U-Net** | **0.9155** |

## 🚀 Execution Steps

**1. Environment Setup**
* Python requirement: `python >= 3.8, < 3.12`
* Environment variable setting (Windows): 
    ```bash
    set KMP_DUPLICATE_LIB_OK=TRUE
    ```

**2. Training**
* **U-Net**:
    ```bash
    python train.py --data_path ../dataset/oxford-iiit-pet --batch_size 4 --epochs 1 --learning-rate 1e-3 --model_type unet
    ```
* **ResNet34 + U-Net**:
    ```bash
    python train.py --data_path ../dataset/oxford-iiit-pet --batch_size 4 --epochs 1 --learning-rate 1e-3 --model_type resnet34unet
    ```

**3. Inference (Testing)**
* **U-Net**:
    ```bash
    python inference.py --model ../saved_models/unet_best.pth --data_path ../dataset/oxford-iiit-pet --batch_size 1 --model_type unet
    ```
* **ResNet34 + U-Net**:
    ```bash
    python inference.py --model ../saved_models/resnet34unet_best.pth --data_path ../dataset/oxford-iiit-pet --batch_size 1 --model_type resnet34unet
    ```

## 💡 Discussion & Future Work
* **Loss Function Optimization**: Future work could explore using Tversky Loss, which is expected to yield better segmentation performance when the foreground proportion is extremely low or when classes are highly imbalanced.
* **Architecture Upgrade**: Integrating FPN (Feature Pyramid Network) or ASPP (Atrous Spatial Pyramid Pooling) into the U-Net's skip connections could more effectively utilize multi-scale contextual information, further improving the model's ability to handle object boundaries.

## 🎥 Demo Video
* [Google Drive Demo Link](https://drive.google.com/file/d/12HbENxft-yPLAYxyYPmFUGOK0C7Wos_X/view?usp=sharing)
