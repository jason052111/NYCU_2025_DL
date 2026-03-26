# Deep Learning Lab 1: Two-Layer Neural Network

> 國立陽明交通大學 (或相應學校) 資工所 / Deep Learning 課程作業一
> Author: 楊承恩 (314551147)

## 📝 Project Overview

此專案實作了一個包含兩層 Hidden Layer 的神經網路 (Neural Network)。
網路的每一層會先將輸入數據進行線性變換（$z = xW + b$），接著通過 Activation Function。模型使用均方誤差（MSE, Mean Squared Error）來計算 Loss，並透過反向傳播（Backward Propagation）來計算梯度與更新權重。

## ⚙️ Implementation Details

### Network Architecture
- **Hidden Units**: 每一層各包含 10 個 hidden units。
- **Output Layer**: 輸出層使用 Sigmoid 函數進行二元分類。
- **Loss Function**: Mean Squared Error (MSE)。
- **Weights Initialization**: 權重使用隨機數初始化（若初始化為 0 會導致整層 unit 輸出相同而無法有效學習）。

### Activation Functions
專案支援三種 Activation Functions，可在建立 Model 時根據需求宣告並切換：
1. `Sigmoid`
2. `ReLU`
3. `Tanh`

### Backpropagation
根據上述三種 Activation Functions 實作了對應的微分函式。在反向傳播過程中，利用連鎖律 (Chain Rule) 計算 Loss 對 weight 及 bias 的微分，並於每個 epoch 進行權重更新。

## 📊 Experimental Results

### 1. Linear Data (`generate_linear`, n=100)
- **Activation Function**: Sigmoid
- **Hidden Units**: 10
- **Learning Rate**: 0.001
- **Result**: 模型預測結果與實際完全相符。採用二分法（預測值 > 0.5 視為 1，反之為 0），Accuracy 達到 **100%**。Loss 在初期迅速下降，約 2000 epochs 時已降至極低。

### 2. XOR Data (`generate_XOR_easy`)
- **Activation Function**: Sigmoid
- **Hidden Units**: 10
- **Learning Rate**: 0.001
- **Result**: 模型預測結果與實際完全相符，Accuracy **100%**。與 Linear data 相比，Loss 下降的趨勢較為平緩，但最終仍成功收斂。

## 💡 Discussions

- **Learning Rate (學習率的影響)**
  學習率較大時，Loss 初期會急遽下降甚至震盪不穩；學習率較小時，Loss 下降緩慢且模型學習效率低。適中的學習率能讓 Loss 平滑收斂。
  
- **Number of Hidden Units (隱藏層神經元數量)**
  若將單位數降為 2，模型會過於簡單而無法預測複雜的邏輯；若設置過多，雖然在此簡單資料集差異不大，但在複雜情境下容易產生 Overfitting（過擬合）。選擇適當的數量能平衡模型複雜度與泛化能力。

- **Without Activation Function (移除激勵函數的影響)**
  若拔除 Activation Function，整個模型將退化為純線性模型，無法學習任何非線性的複雜問題。在此情況下若設定 0.001 的學習率會導致梯度爆炸（數據爆掉），只有極小的學習率才能順利運作。不過在面對純 Linear 的輸入資料時，即使沒有激勵函數，其 Accuracy 表現依然良好。
