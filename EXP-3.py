#3. WAp for Implementation of Backpropagation Neural Network (BPNN) using TensorFlow using TensorFlow Library.

# Description:
This TensorFlow-based Python script implements a simple feedforward neural network to classify handwritten digits from the MNIST dataset.
The network consists of an input layer (784 neurons), two hidden layers (128 and 64 neurons, respectively) with ReLU activation functions, and an output layer (10 neurons for digit classification).
The model is trained using the Stochastic Gradient Descent (SGD) optimizer with cross-entropy loss. 
It iterates through 20 epochs, processing mini-batches of 100 samples.
After each epoch, the test accuracy is computed and displayed.

code : 
#libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

# Hyperparameters
learning_rate = 0.01
num_epochs = 20
batch_size = 100
n_input = 784   
n_hidden1 = 128  
n_hidden2 = 64   
n_output = 10   

# Initialize weights and biases
initializer = tf.initializers.GlorotUniform()
W1 = tf.Variable(initializer([n_input, n_hidden1]))
b1 = tf.Variable(tf.zeros([n_hidden1]))

W2 = tf.Variable(initializer([n_hidden1, n_hidden2]))
b2 = tf.Variable(tf.zeros([n_hidden2]))

W3 = tf.Variable(initializer([n_hidden2, n_output]))
b3 = tf.Variable(tf.zeros([n_output]))

# Feed-forward functionA
def forward_pass(x):
    z1 = tf.add(tf.matmul(x, W1), b1)
    a1 = tf.nn.relu(z1)

    z2 = tf.add(tf.matmul(a1, W2), b2)
    a2 = tf.nn.relu(z2)

    logits = tf.add(tf.matmul(a2, W3), b3)
    return logits

# Loss function
def compute_loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

# Accuracy metric
def compute_accuracy(logits, labels):
    correct_preds = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
    return tf.reduce_mean(tf.cast(correct_preds, tf.float32))

# Backpropagation using GradientTape
optimizer = tf.optimizers.SGD(learning_rate)

for epoch in range(num_epochs):
    for i in range(0, x_train.shape[0], batch_size):
        x_batch = x_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        with tf.GradientTape() as tape:
            logits = forward_pass(x_batch)
            loss = compute_loss(logits, y_batch)

        gradients = tape.gradient(loss, [W1, b1, W2, b2, W3, b3])
        optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2, W3, b3]))

    # Evaluate after each epoch
    test_logits = forward_pass(x_test)
    test_accuracy = compute_accuracy(test_logits, y_test)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.numpy():.4f}, Test Accuracy: {test_accuracy.numpy():.4f}")

# Final Evaluation
final_accuracy = compute_accuracy(forward_pass(x_test), y_test)
print(f"Final Test Accuracy: {final_accuracy.numpy():.4f}")

output :
Epoch 1/20, Loss: 0.5234, Test Accuracy: 0.8893
Epoch 2/20, Loss: 0.3121, Test Accuracy: 0.9174
...
Epoch 20/20, Loss: 0.1478, Test Accuracy: 0.9531
Final Test Accuracy: 0.9531

Limitations :
No Regularization – Lacks techniques like dropout or L2 regularization, making it prone to overfitting.
SGD Optimizer – Uses basic SGD instead of advanced optimizers like Adam or RMSprop, which might lead to slower convergence.
No Batch Normalization – Doesn't normalize activations, which could improve training speed and stability.
Limited Architecture – Only two hidden layers; deeper networks generally perform better on complex datasets.
No Data Augmentation – Doesn't apply transformations to training images, which could improve generalization.

Explanation :
Dataset Loading: The MNIST dataset is loaded and normalized (scaled between 0 and 1).
One-hot Encoding: Labels are converted into a one-hot vector format.
Weight Initialization: Uses the Glorot Uniform initializer for better weight distribution.
Feedforward Pass: Computes outputs layer-by-layer using matrix multiplications and activation functions.
Loss Computation: Uses softmax cross-entropy to measure classification error.
Optimization: Backpropagation is implemented using TensorFlow’s GradientTape, and SGD updates the weights.
Epoch-wise Training: The model is trained over 20 epochs with mini-batch processing.
Accuracy Calculation: After each epoch, test accuracy is evaluated.
