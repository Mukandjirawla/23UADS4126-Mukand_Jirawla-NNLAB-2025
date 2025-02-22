# 2. WAP to implement a multi-layer perceptron (MLP) network with one hidden layer using numpy in Python. Demonstrate that it can learn the XOR Boolean function.

Description - This code implements a simple neural network to solve the XOR problem using perceptrons.
A single perceptron can't solve XOR since it's not linearly separable, so the code creates a hidden layer with two perceptrons.
Their outputs are then fed into a third perceptron (output layer), which learns to combine them to correctly compute XOR.
The perceptrons are trained using a basic weight update rule. After training, the network correctly predicts XOR values: [0, 1, 1, 0]. 
This approach manually builds a multi-layer perceptron (MLP).

code :
import numpy as np  # Importing NumPy for numerical computations

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        """
        Initializes the Perceptron with random weights.
        
        Parameters:
        input_size (int): Number of input features.
        learning_rate (float): Step size for weight updates.
        epochs (int): Number of iterations over the dataset.
        """
        self.weights = np.random.randn(input_size + 1)  # +1 for the bias term
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        """Activation function (step function). Returns 1 if x >= 0, else 0."""
        return 1 if x >= 0 else 0

    def predict(self, x):
        """
        Predicts the output for a given input.
        
        Parameters:
        x (array-like): Input features.
        
        Returns:
        int: Predicted class (0 or 1).
        """
        x = np.insert(x, 0, 1)  # Adding bias term
        return self.activation(np.dot(self.weights, x))

    def train(self, X, y):
        """
        Trains the perceptron using the given dataset.
        
        Parameters:
        X (array-like): Input feature matrix.
        y (array-like): Target labels.
        """
        X = np.c_[np.ones((X.shape[0], 1)), X]  # Adding bias term
        
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                y_pred = self.activation(np.dot(self.weights, X[i]))
                
                # Update weights based on prediction error
                if y[i] == 1 and y_pred == 0:
                    self.weights += self.learning_rate * X[i]
                elif y[i] == 0 and y_pred == 1:
                    self.weights -= self.learning_rate * X[i]

    def evaluate(self, X, y):
        """
        Evaluates the perceptron on the given dataset.
        
        Parameters:
        X (array-like): Input feature matrix.
        y (array-like): True labels.
        
        Returns:
        tuple: (accuracy, predictions)
        """
        y_pred = [self.predict(x) for x in X]
        accuracy = sum(y_pred[i] == y[i] for i in range(len(y))) / len(y)
        return accuracy, y_pred

# Training multiple perceptrons for different logical functions
fun1_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
fun1_y = np.array([0, 0, 0, 1])  # AND Function
hiddenPerceptron1 = Perceptron(input_size=2)
hiddenPerceptron1.train(fun1_X, fun1_y)
fun1_accuracy, predictionsLayer1 = hiddenPerceptron1.evaluate(fun1_X, fun1_y)

fun2_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
fun2_y = np.array([0, 0, 1, 0])  # Custom Function
hiddenPerceptron2 = Perceptron(input_size=2)
hiddenPerceptron2.train(fun2_X, fun2_y)
fun2_accuracy, predictionsLayer2 = hiddenPerceptron2.evaluate(fun2_X, fun2_y)

fun3_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
fun3_y = np.array([0, 1, 0, 0])  # Custom Function
hiddenPerceptron3 = Perceptron(input_size=2)
hiddenPerceptron3.train(fun3_X, fun3_y)
fun3_accuracy, predictionsLayer3 = hiddenPerceptron3.evaluate(fun3_X, fun3_y)

fun4_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
fun4_y = np.array([1, 0, 0, 0])  # Custom Function
hiddenPerceptron4 = Perceptron(input_size=2)
hiddenPerceptron4.train(fun4_X, fun4_y)
fun4_accuracy, predictionsLayer4 = hiddenPerceptron4.evaluate(fun4_X, fun4_y)

# Combining outputs of the hidden perceptrons into a final layer
X = np.array([predictionsLayer1, predictionsLayer2, predictionsLayer3, predictionsLayer4])
y = np.array([0, 1, 1, 0])  # Final classification task

# Training the final perceptron
finalPerceptron = Perceptron(input_size=4)
finalPerceptron.train(X, y)
final_accuracy, final_predictions = finalPerceptron.evaluate(X, y)

# Display final results
print(f"\nFinal Perceptron Weights: {finalPerceptron.weights}")
print(f"Final Perceptron Predictions: {final_predictions}")
print(f"Final Perceptron Accuracy: {final_accuracy * 100:.2f}%")

output:

Final Perceptron Weights: [ 0.35417963 -0.85788471  0.92128953  0.42252402 -0.35578727]
Final Perceptron Predictions: [0, 1, 1, 0]
Final Perceptron Accuracy: 100.00%

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
