# [cite_start]DL_LAB1 [cite: 1]

- [cite_start]**系級**: 資工碩一 [cite: 1]
- [cite_start]**ID**: 314551147 [cite: 1]
- [cite_start]**Name**: 楊承恩 [cite: 1]

## [cite_start]1. Introduction [cite: 1]

[cite_start]此作業建構了兩層的 hidden layer 的 neuron network，每層中我先讓 input 經過線性變換 $z=xW+b$ 再將 z 傳入 activation function，利用估計值跟實際值配上 MSE 方法來算出 Loss，再用 Loss 來做 backward propagation [cite: 1]。

## [cite_start]2. Implementation Details [cite: 1]

### [cite_start]A. Network Architecture [cite: 1]
[cite_start]我的 neuron network 中的 layer 中各包含了 10 個 hidden units，輸出層的部分使用了 sigmoid，計算 Loss 的方法使用了 MSE [cite: 1]。

### [cite_start]B. Activation Functions [cite: 1]
[cite_start]Activation Functions 使用了 Sigmoid, ReLU, Tanh 這三種，可以在建 Model 時做宣告看此 Model 想使用哪個 Activation Function [cite: 1]。

### [cite_start]C. Backpropagation [cite: 1]
[cite_start]我先根據 Sigmoid, ReLU, Tanh 建了三個微分函式，在 Backpropagation 中利用了 Chain rule 的方法來算出 Loss 對 weight 和 bias 的微分，並且在每個 epoch 進行對 weight 和 bias 的更新 [cite: 1]。

## [cite_start]3. Experimental Results [cite: 1]

### [cite_start]Input_data: generate_linear (n=100) [cite: 1]
- [cite_start]**activation functions**: sigmoid [cite: 1]
- [cite_start]**numbers of hidden units**: 10 [cite: 1]
- [cite_start]**lr**: 0.001 [cite: 2]

[cite_start]根據模型的訓練後，預測出來的結果跟實際的一樣，因為採用二分法，所以預測出來的值 > 0.5 則預測為 1 反之則為 0，可看到 accuracy 為 100% [cite: 2][cite_start]。並且 Loss 在一開始則直接降了很多，在大概 2000 epoch 時 Loss 就降得很低了 [cite: 2]。

### [cite_start]Input_data: generate_XOR_easy() [cite: 5]
- [cite_start]**activation functions**: sigmoid [cite: 5]
- [cite_start]**numbers of hidden units**: 10 [cite: 5]
- [cite_start]**lr**: 0.001 [cite: 5]

[cite_start]經過訓練後，模型的預測結果跟實際一模一樣，因為採用二分法，所以預測出來的值 > 0.5 則預測為 1 反之則為 0，可看到 accuracy 為 100% [cite: 5][cite_start]。在此 input_data 下不像 linear Loss，而是下降的比較平緩 [cite: 5]。

## [cite_start]4. Discussions [cite: 8]

- **Learning rate**: 
[cite_start]在調整 learning rate 時發現，當 learning rate 較大時會導致 Loss 一開始突然的降的很快；當 learning rate 較小時，Loss 會降得很慢模型也學習的慢 [cite: 8][cite_start]。只有在 learning rate 適中時 Loss 才會下降得比較平滑 [cite: 8]。

- **Number of hidden units**:
[cite_start]hidden units 設在 2 時，會使訓練出來的模型無法預測複雜的邏輯；但當 units 太多時雖然在這次的資料中無法看出太多差異，但是會產生過擬合的情況的 [cite: 8][cite_start]。只有在 units 數剛好的情況下才不會使模型太簡單，也不容易過擬合 [cite: 8]。

- **Without activation function**:
[cite_start]在一開始拿掉 activation function 時我的 lr 設 0.001 但發現數據會爆掉，只有在 lr 設很小時才看得出差別 [cite: 8][cite_start]。拿掉 activation function 就等於直接讓 x 跑一堆線性模型，無法學習一些非線性的問題，但在 linear 的 input_data 中他的 accuracy 還是很好的 [cite: 8]。

- **Extra Implementation Discussions**:
[cite_start]我的 Loss function 是採用 MSE 的方法，但經過查資料的過程中發現可以用 cross entropy，因為在二分法時使用 cross entropy 會比較好 [cite: 8, 9][cite_start]。一開始沒有想太多直接把 weight 設為 0，但發現 $z=xW+b$ 如果 w 為 0，整層 unit 輸出則都會一樣，因此把 weight 改為隨機數 [cite: 9][cite_start]。這次作業遇到最大的問題則是在做 backward propagation 不太清楚該如何算 Loss 對各值的微分，於是花了點時間在研究這部分 [cite: 9]。

## [cite_start]5. Questions [cite: 9]

### [cite_start]A. What are the purposes of activation functions? [cite: 9]
[cite_start]這跟 4(C) 有關，如果沒有了 activation function 則模型就會呈現完全線性的，沒有辦法適應一些複雜的問題，也可以說是沒辦法學習非線性的問題 [cite: 10]。

### [cite_start]B. What if the learning rate is too large or too small? [cite: 10]
[cite_start]Learning rate 如果太小則模型會訓練的較慢，並且如果卡在一個 local minima 會因為步伐太小而出不來 [cite: 11][cite_start]。如果 learning rate 太大 Loss 可能會一直都不穩定，會一直上下震盪 [cite: 11]。

### [cite_start]C. What are the purposes of weights and biases in a neural network? [cite: 11]
[cite_start]weight 和 bias 在 neural network 中是一個可調式的參數 [cite: 12][cite_start]。weight 主要可以看做是當此輸入越重要時，就可以賦予較大的 weight，而在訓練的過程就是不斷的去找 training data 與 label 的接近度來看是否此輸入是重要與否，並且調整 weight 大小 [cite: 12][cite_start]。bias 則是可以使模型更加靈活 [cite: 12]。

## [cite_start]References [cite: 12]
- [cite_start][IT 邦幫忙文章](https://ithelp.ithome.com.tw/m/articles/10323813) [cite: 12]
- [cite_start][Hung-yi Lee YouTube 教學](https://www.youtube.com/watch?v=ibJpTrp5mcE&ab_channel=Hung-yiLee) [cite: 12]

[cite_start]並且使用 GPT 幫助我理解 backward propagation，詢問如何建造 constructor 為了讓 Layer 跟 Model 可以有一個 class 使我可以只需要建構並改變 parameter 則可以改變 lr, numbers of hidden units, activation function [cite: 12][cite_start]。並且詢問了 weight 和 bias 的初始值怎麼設比較好 [cite: 12]。
