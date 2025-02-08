# 1. WAP to implement the Perceptron Learning Algorithm using numpy in Python. Evaluate performance of a single perceptron for NAND and XOR truth tables as input dataset.

# Step 1: Import Required Libraries
import numpy as np

# Step 2: Define Perceptron Class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=1000):
        self.weights = np.zeros(input_size + 1)  # Including bias term
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        """ Step function """
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        """ Calculate the weighted sum and apply the activation function """
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation(summation)

    def train(self, training_inputs, labels):
        """ Train the perceptron using the training dataset """
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error  # Bias update

# Step 3: Define Truth Tables for NAND and XOR
# NAND Truth Table
nand_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
nand_labels = np.array([1, 1, 1, 0])  # Output of NAND gate

# XOR Truth Table
xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_labels = np.array([0, 1, 1, 0])  # Output of XOR gate

# Step 4: Initialize Perceptron and Train it
# Train Perceptron for NAND Gate
print("Training Perceptron for NAND gate...")
nand_perceptron = Perceptron(input_size=2)  # 2 inputs (for NAND)
nand_perceptron.train(nand_inputs, nand_labels)

# Test Perceptron for NAND Gate
print("Testing Perceptron on NAND gate...")
for inputs in nand_inputs:
    print(f"Input: {inputs}, Predicted: {nand_perceptron.predict(inputs)}")

# Train Perceptron for XOR Gate
print("\nTraining Perceptron for XOR gate...")
xor_perceptron = Perceptron(input_size=2)  # 2 inputs (for XOR)
xor_perceptron.train(xor_inputs, xor_labels)

# Test Perceptron for XOR Gate
print("Testing Perceptron on XOR gate...")
for inputs in xor_inputs:
    print(f"Input: {inputs}, Predicted: {xor_perceptron.predict(inputs)}")

'''
Explanation:
NAND Gate: The perceptron is able to correctly classify the NAND inputs because it is a linearly separable problem.
XOR Gate: The perceptron fails to classify the XOR gate correctly because XOR is not linearly separable, and a single perceptron is insufficient to learn the XOR function.
Limitations:
The perceptron algorithm works well for linearly separable problems like NAND but fails for problems like XOR, which are not linearly separable.
More complex networks (e.g., multi-layer neural networks) are required to solve non-linear problems like XOR.
'''
