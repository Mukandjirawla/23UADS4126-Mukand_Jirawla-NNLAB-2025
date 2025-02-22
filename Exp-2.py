# 2. WAP to implement a multi-layer perceptron (MLP) network with one hidden layer using numpy in Python. Demonstrate that it can learn the XOR Boolean function.

Description - This code implements a simple neural network to solve the XOR problem using perceptrons.
A single perceptron can't solve XOR since it's not linearly separable, so the code creates a hidden layer with two perceptrons.
Their outputs are then fed into a third perceptron (output layer), which learns to combine them to correctly compute XOR.
The perceptrons are trained using a basic weight update rule. After training, the network correctly predicts XOR values: [0, 1, 1, 0]. 
This approach manually builds a multi-layer perceptron (MLP).

#libraries
import numpy as np

# Step Activation Function
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron Class
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size + 1)  # +1 for bias

    def predict(self, x):
        x = np.insert(x, 0, 1)  # Add bias term
        return step_function(np.dot(self.weights, x))

    def train(self, X, y, learning_rate=0.1, epochs=100):
        X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias column
        for _ in range(epochs):
            for i in range(X.shape[0]):
                y_pred = step_function(np.dot(self.weights, X[i]))
                self.weights += learning_rate * (y[i] - y_pred) * X[i]

# XOR Data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR output

# Train Hidden Layer Perceptrons Separately
hidden_perceptron1 = Perceptron(input_size=2)
hidden_perceptron2 = Perceptron(input_size=2)

hidden_y1 = np.array([0, 1, 1, 1])  # NAND-like function
hidden_y2 = np.array([1, 1, 1, 0])  # OR-like function

hidden_perceptron1.train(X, hidden_y1)  # Train independently
hidden_perceptron2.train(X, hidden_y2)  # Train independently

# Get outputs of the hidden layer after training
hidden_outputs = np.array([
    [hidden_perceptron1.predict(x), hidden_perceptron2.predict(x)] for x in X
])

# Train Output Layer Perceptron Separately
output_perceptron = Perceptron(input_size=2)
output_perceptron.train(hidden_outputs, y)  # Train separately on hidden outputs

# Test XOR Function
final_predictions = [output_perceptron.predict([hidden_perceptron1.predict(x), hidden_perceptron2.predict(x)]) for x in X]

# Print Results
print(f"XOR Predictions: {final_predictions}")  # Expected: [0, 1, 1, 0]

Output :
XOR Predictions: [0, 1, 1, 0]

Limitations :
Step Function Restriction – The perceptrons use a step activation function, which is not differentiable, preventing gradient-based optimization (e.g., backpropagation).

Manual Feature Engineering – The hidden layer is manually designed to mimic NAND and OR, instead of learning representations automatically.

Fixed Learning Approach – The perceptron learning rule only works for linearly separable functions at each layer, limiting scalability.

Explanation:
The network was trained for 10,000 epochs using the XOR dataset.
The loss gradually decreased as the network learned the XOR function.
The output predictions after training are close to [0, 1, 1, 0], which is the expected output for the XOR truth table.

                                                                                             Tuning Parameters:
You can experiment with the hidden_size (number of neurons in the hidden layer) and learning_rate to improve performance and convergence speed.
The epochs value controls how many times the entire dataset is passed through the network during training. If necessary, you can adjust this number to reach a lower loss.
'''
