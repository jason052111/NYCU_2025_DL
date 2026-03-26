import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n=100):
    # 線性的 input_data label 在 x = y 線下為 0，反之為 1
    import numpy as np
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    # XOR 的 input_data
    import numpy as np
    inputs = []
    labels = []
    
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        
        if(0.1*i == 0.5):
            continue
            
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
        
    return np.array(inputs), np.array(labels).reshape(21,1)


def show_result(x, y, pred_y):
    # 顯示圖
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.where(x > 0, x, 0)

def Tanh(x):
    return np.tanh(x)

# 各個 activation funtion 的微分公式
def Sigmoid_deriv(x):
    return np.multiply(x, 1 - x)

def ReLU_deriv(x):
    return np.where(x > 0, 1, 0)

def Tanh_deriv(x):
    return 1 - np.multiply(x, x)

class Layer:
    def __init__(self, input_size, output_size, activation='Sigmoid'):
        self.w = np.random.randn(input_size, output_size)
        self.b = np.zeros((1, output_size))
        self.activation = activation
    
    def forward(self, input):
        # 每層先經過 Linear Transformation 再將 z 丟入 activation function
        self.input = input
        z = np.dot(input, self.w) + self.b # xW+b
        if self.activation == "Sigmoid":
            self.output = Sigmoid(z)
        elif self.activation == "ReLU":
            self.output = ReLU(z)
        elif self.activation == "Tanh":
            self.output = Tanh(z)
        
        return self.output
    
    def backward(self, grad_output, lr=0.1):
        # 計算 weight 和 bias 的 gradient 來更新 parameter

        # grad_z 為 Loss 對 output 微分 (利用到Chain rule)
        if self.activation == "Sigmoid":
            grad_z = grad_output * Sigmoid_deriv(self.output)
        elif self.activation == "ReLU":
            grad_z = grad_output * ReLU_deriv(self.output)
        elif self.activation == "Tanh":
            grad_z = grad_output * Tanh_deriv(self.output)
      
        grad_w = np.dot(self.input.T, grad_z)           # Loss 對 weight 微分
        grad_b = np.sum(grad_z, axis=0, keepdims=True)  # Loss 對 bias 微分
        grad_input = np.dot(grad_z, self.w.T)           # Loss 對 input 微分

        # 更新 weight 和 bias
        self.w -= lr * grad_w
        self.b -= lr * grad_b

        return grad_input

class Model:
    def __init__(self, input_size=2, hidden_size=8, output_size=1, lr=0.05, activation='Sigmoid'):
        self.lr = lr
        self.hidden1_layer = Layer(input_size, hidden_size, activation)   # 隱藏層1
        self.hidden2_layer = Layer(hidden_size, hidden_size, activation)  # 隱藏層2
        self.output_layer = Layer(hidden_size, output_size, 'Sigmoid')    # 輸出層 (因為label為0,1先用sigmoid再將>0.5改為1)
    
    def train(self, input, label, epochs=100000, interval=5000):
        loss_graph = []
        for epoch in range(epochs):
            pred = self.output_layer.forward(self.hidden2_layer.forward(self.hidden1_layer.forward(input)))
            loss = np.mean((pred - label) ** 2)
            loss_graph.append(loss)
            self.hidden1_layer.backward(self.hidden2_layer.backward(self.output_layer.backward(pred - label, self.lr), self.lr), self.lr)
            if epoch % interval == 0:
                print(f"epoch {epoch:5d} loss : {loss:.15f}")

        plt.subplot(2,1,1)
        plt.plot(loss_graph)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning Curve")
        plt.show()
    
    def evaluate(self, input, label):
        pred = self.output_layer.forward(self.hidden2_layer.forward(self.hidden1_layer.forward(input)))
        pred_label = np.where(pred > 0.5, 1, 0)
        acc = np.mean(pred_label == label)   # 計算準確度
        loss = np.mean((pred - label) ** 2)  # 計算 Loss

        for i in range(len(input)):
            print(f"Iter{i:02d} | Ground truth: {float(label[i]):.1f} | prediction: {float(pred[i]):.5f} |")

        print(f"loss={loss:.5f} accuracy={acc:.2%}")
        show_result(input, label, pred_label)    

x, y = generate_linear(n=100)
#x, y = generate_XOR_easy()
model = Model(2, 10, 1, 0.001, 'Sigmoid')
model.train(x, y, 100000, 5000)
model.evaluate(x, y)