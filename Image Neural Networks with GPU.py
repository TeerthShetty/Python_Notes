# My system details
# Torch: 2.11.0+cu128
# CUDA: 12.8
# CUDA Available: True
# GPU Count: 1
# GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU



# Link from where I learnt: https://www.youtube.com/watch?v=w8yWXqWQYmU&t=12s
# Link for the dataset: https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras
#############################################################
#############################################################
# Problem Statement
#############################################################
#############################################################

# To make a NN using basic math and numpy to detect handwritten numbers

# We are using Fully Connected Neural Network not a Convoluted Neural Network
#############################################################
# FULLY CONNECTED NEURAL NETWORK
#############################################################

# Structure : Each neuron in one layer is connected to every neuron in the next layer.
# Data Handling: Treats input as a flat vector, losing spatial or sequential structure
# Use case: Best for tabular data, classification task or as the final layers in CNNs
# Simple to make
# Effective for small datasets or non-spatial data
# If number of parameters are high then it faces overfitting
# Does not work well with high dimensional inputs







#############################################################
#############################################################
# The Math
#############################################################
#############################################################


# 28 x 28 pixels training data which is 784 pixels
# Each pixel is a value between 0 and 255(both inclusive) (2^8 = 265 = 255 - 0 + 1)
# 255 -> White
# 0   -> Black

# Each row will have 784 columns long because each of them is going to correspond to one pixel in that image
# 784 = 28 x 28; -> Entering the data as a single row instead of a 2d array
# Eg
# [[p00, p01, p02, p03........p27],
#  [p28, p29, p30,................]];
# Converted to
# [p00, p01, p02, p03, .............];
# This compression is done because the NN only accept 1D Array





#############################################################
#############################################################
# The Structure
#############################################################
#############################################################


# 784 (input) → 10 (hidden) → 10 (output)



# X = data.T

# Image data : [p00, p01, p02..........] this will be transposed

# This will be a 2 layered Neural Network
# The First layer (Zeroth Layer) has 784 nodes -> Input layer (not counted as a layer though
# Second layer (Frist Layer) will have 10 nodes -> First Hidden Layer
# The Third layer (Second Layer) will have 10 nodes -> Output Layer

# Forward Propagation: Inputting the value
# A0 = X (784 x m), There is no calculations that happen here
# First layer is the unactivated first layer
# Z[1] = W[1]A[0] + b[1] -> Linear combination (dot product is used)
# Z[1] = Raw output of the neuron
# W[1] = Weight for matrix level 1
# b[1] = bias vector for layer 1
# A[0] = input to the layer

# An Activation function is need because the output needs to be non linear
# Without the Activation function it will just be a rally fancy linear regression

# A[1] = ReLU(Z[1])
# This bend the straight line into a curve
# Then in layer 2 the bend increases
# This helps make a better fitting curve

# ReLU or the Softmax function will convert the output layer value into probability values thus they lie between 0 and 1

# So now we need good weights and biases, for this we will be using back propagation
# dz[2] = A[2] - Y
# dW[2] = (1/m).dZ[2].A[1]
# db[2] = (1/m).{sigma}(dZ[2]
# dZ[1] = W[2][T].dZ[2].g'(Z[2])
# dW[1] = (1/m).dZ[1].X[T]
# db[2] = (1/m).{sigma}(dZ[1])

# W[1] = W[1] - {alpha}.dW[1]
# b[1] = b[1] - {alpha}.db[1]
# W[2] = W[2] - {alpha}.dW[2]
# b[2] = W[2] - {alpha}.dW[2]

#{alpha} is the learning rate


# Layer 1 (Input Layer)
# Input : 784 x m
# output: 10 x m
# So each of the 10 neurons in the hidden layer must receive 784 inputs.

# Why is W1 shaped (10, 784)?

# You have 10 neurons in hidden layer 1
# Each neuron needs 784 weights (one per input pixel)
# W1 looks like this

# Neuron 1:  [w1, w2, w3, w4........w784]
# Neuron 2:  [w1, w2, w3, w4........w784]
# Neuron 3:  [w1, w2, w3, w4........w784]
# Neuron 4:  [w1, w2, w3, w4........w784]
# Neuron 5:  [w1, w2, w3, w4........w784]
# Neuron 6:  [w1, w2, w3, w4........w784]
# Neuron 7:  [w1, w2, w3, w4........w784]
# Neuron 8:  [w1, w2, w3, w4........w784]
# Neuron 9:  [w1, w2, w3, w4........w784]
# Neuron 10: [w1, w2, w3, w4........w784]

# W1 = (number of neurons in layer 1) × (number of inputs)
# W1 = 10 × 784


# Why is b1 shaped (10, 1)?

# Bias is one number per neuron.
# You have 10 neurons → you need 10 biases.

# | Parameter | Shape     | Meaning                                            |
# | --------- | --------- | -------------------------------------------------- |
# | **W1**    | (10, 784) | Weights from 784 inputs → 10 hidden neurons        |
# | **b1**    | (10, 1)   | Bias for each of the 10 hidden neurons             |
# | **W2**    | (10, 10)  | Weights from 10 hidden neurons → 10 output neurons |
# | **b2**    | (10, 1)   | Bias for each of the 10 output neurons             |

#############################################################
#############################################################
# The Code
#############################################################
#############################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
import math

np.show_config()
print(torch.cuda.is_available())
print(torch.cuda.device_count())
import sys
print(sys.executable)
print(sys.version)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
start = time.time()

import torch

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())
print("GPU Count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



# Train data and test data is already split
# Here we load the train data
data = pd.read_csv('train.csv')
print(data.head(5))

data = np.array(data)
m,n = data.shape # Dimension of the data
np.random.shuffle(data) # Shuffling is done inorder to remove any order biasing


# We take the first 1000 tuples for development as developer test
# Used to check how well the network is learning (tuning, debugging, accuracy checks)
data_dev = data[0:1000].T # developer data set
Y_dev = torch.tensor(
    data_dev[0],
    dtype=torch.long,
    device=device
)

X_dev = torch.tensor(
    data_dev[1:n] / 255.,
    dtype=torch.float32,
    device=device
)
# Dividing the values by 255 puts all the values between 0 and 1 since black to white will have values 0 to 255 => 256
# This improves numerical stability, speeds up convergence of gradient descent, and prevents large input values from \
# dominating the learning process.


# The tuples after exluding the developer dataset is the new training dataset
data_train = data[1000:m].T # Training dataset
Y_train = torch.tensor(
    data_train[0],
    dtype=torch.long,
    device=device
)

X_train = torch.tensor(
    data_train[1:n] / 255.,
    dtype=torch.float32,
    device=device
)
_,m_train = X_train.shape

# print(X_train[:, 0].shape)
# print(_,m_train)
# print(Y_train.shape)
# print(Y_train)
print(type(X_train))
print(type(Y_train))

print(X_train.device)
print(Y_train.device)

print(X_train.dtype)
print(Y_train.dtype)
# Z[2]= W[2]A[1]+b[2]
# A[2] = softmax(Z[2])

def init_params():
    # Initialize weights for the first layer
    # Shape: (10, 784)
    # 10  → number of neurons in the hidden layer
    # 784 → number of input features (28×28 pixels)
    # np.random.rand() generates values in [0,1), so subtracting 0.5 shifts them to [-0.5, 0.5]
    # This keeps weights small and centered around zero, helping stable learning.

    W1 = torch.randn((10, 784), device=device) * math.sqrt(2 / 784)
    # Initialize bias for the first hidden layer
    # Shape: (10, 1)
    # One bias value for each of the 10 hidden neurons
    # Also centered around zero for symmetry and stable gradients
    b1 = torch.rand((10, 1), dtype=torch.float32, device=device) - 0.5

    # Initialize weights for the second layer
    # Shape: (10, 10)
    # 10 hidden neurons → 10 output neurons (digits 0–9)
    W2 = torch.randn((10, 10), device=device) * math.sqrt(2 / 10)
    # Initialize weights for the second layer
    # Shape: (10, 10)
    # 10 hidden neurons → 10 output neurons (digits 0–9)
    b2 = torch.rand((10, 1), dtype=torch.float32, device=device) - 0.5
    print(type(W1))
    print(W1.device)
    print(W1.dtype)


    return W1, b1, W2, b2


def ReLU(Z):
    # ReLU (Rectified Linear Unit) activation function
    # It replaces all negative values in Z with 0
    # and keeps positive values unchanged.
    # Mathematical form:
    # ReLU(Z) = max(0, Z)
    return  torch.relu(Z)


def softmax(Z):
    # Softmax activation function
    # Converts raw scores (Z) into probabilities.
    # Each column will sum to 1 and represent the probability
    # distribution over the 10 output classes (digits 0–9).

    # Subtracting the maximum value from each column of Z
    # is done for numerical stability.
    # It prevents very large exponent values which can cause overflow.

    expZ = torch.exp(Z - torch.max(Z, dim=0, keepdim=True).values)

    # Divide each exponentiated value by the sum of all exponentiated
    # values in the same column so that probabilities sum to 1.
    return expZ / torch.sum(expZ, dim=0, keepdim=True)

def cross_entropy_loss(A2, Y):
    m = Y.shape[0]
    # Small value to avoid log(0)
    epsilon = 1e-12
    # Convert labels to one-hot
    one_hot_Y = one_hot(Y).T
    # Clamp probabilities
    A2 = torch.clamp(A2, epsilon, 1.0 - epsilon)
    # Cross Entropy
    loss = -(one_hot_Y * torch.log(A2.T)).sum() / m
    return loss.item()

def forward_prop(W1, b1, W2, b2, X):
    # Forward propagation through the neural network.
    # This function computes the output of each layer step-by-step,
    # starting from the input and ending with the final prediction probabilities.

    # Z1 = W1·X + b1
    # Linear combination for the hidden layer
    # W1 shape: (10, 784)
    # X  shape: (784, m)
    # b1 shape: (10, 1)
    # Result Z1 shape: (10, m)
    Z1  = torch.matmul(W1, X) + b1

    # A1 = ReLU(Z1)
    # Apply ReLU activation to introduce non-linearity
    # Keeps positive values, sets negative values to zero
    # A1 shape: (10, m)
    A1 = ReLU(Z1)

    # Z2 = W2·A1 + b2
    # Linear combination for the output layer
    # W2 shape: (10, 10)
    # A1 shape: (10, m)
    # b2 shape: (10, 1)
    # Result Z2 shape: (10, m)
    Z2 = torch.matmul(W2, A1) + b2

    # A2 = softmax(Z2)
    # Apply softmax activation to convert raw scores into probabilities
    # Each column in A2 represents probability distribution over digits 0–9
    # A2 shape: (10, m)
    A2 = softmax(Z2)

    # Return all intermediate values for use in backpropagation
    return Z1, A1, Z2, A2

def ReLU_deriv(Z): # Slope of ReLU is one since the slope is 45 degrees
    # Derivative of the ReLU activation function
    # This function computes the gradient of ReLU with respect to Z.
    # ReLU(Z) = max(0, Z)
    #
    # Its derivative is:
    # 1, if Z > 0
    # 0, if Z ≤ 0
    #
    # So this returns a boolean mask:
    # True  (1) where Z is positive
    # False (0) where Z is zero or negative
    return Z > 0


# def one_hot(Y):
#     # Convert labels to integer type (safety step)
#     # Ensures indexing works correctly
#     Y = Y.astype(int)
#
#     # Create a matrix of zeros with shape:
#     # (number of samples, number of classes)
#     # Y.size        → total number of labels
#     # Y.max() + 1   → number of classes (for MNIST: 0 to 9 → 10 classes)
#     one_hot_Y = np.zeros((Y.size, Y.max() + 1)) # a matrix of zeros is created, Y.max() is no of classes -> 0 to 9 then +1 for 10
#
#     # Set the correct class index to 1 for each sample
#     # np.arange(Y.size) → row indices [0, 1, 2, ..., Y.size-1]
#     # Y                → column indices (actual class labels)
#     # This marks the correct class position with 1
#     one_hot_Y[np.arange(Y.size), Y] = 1
#
#     # Transpose so that shape becomes:
#     # (number of classes, number of samples)
#     # Required format for matrix operations in backpropagation
#     one_hot_Y = one_hot_Y.T
#
#     return one_hot_Y

def one_hot(Y):

    # Ensure integer labels
    Y = Y.long()

    # Number of classes
    # num_classes = int(torch.max(Y).item()) + 10
    num_classes = 10


    # Create one-hot matrix on the same device as Y
    one_hot_Y = torch.zeros(
        (Y.size(0), num_classes),
        dtype=torch.float32,
        device=Y.device
    )

    # Set correct class positions to 1
    one_hot_Y[torch.arange(Y.size(0), device=Y.device), Y] = 1

    return one_hot_Y.T

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    # Perform backpropagation to compute gradients of the loss
    # with respect to all weights and biases.

    # Convert labels Y into one-hot encoded form
    # Shape: (10, m)
    # Number of samples in the CURRENT mini-batch
    m = X.shape[1]
    one_hot_Y = one_hot(Y)

    # dZ2 = A2 - Y
    # Gradient of loss w.r.t Z2 when using softmax + cross entropy loss
    # A2 : predicted probabilities (10, m)
    # one_hot_Y : true labels in one-hot form (10, m)
    dZ2 = A2 - one_hot_Y

    # dW2 = (1/m) * dZ2 · A1ᵀ
    # Gradient of loss w.r.t W2
    # dZ2 shape: (10, m)
    # A1.T shape: (m, 10)
    # Result dW2 shape: (10, 10)
    dW2 = 1 / m * torch.matmul(dZ2, A1.T)

    # db2 = (1/m) * sum(dZ2)
    # Gradient of loss w.r.t b2
    # Summed over all samples
    # Ideally should keep dimensions: axis=1, keepdims=True
    db2 = (1 / m) * torch.sum(dZ2, dim=1, keepdim=True)

    # dZ1 = W2ᵀ · dZ2 ⊙ ReLU'(Z1)
    # Backpropagate error to hidden layer
    # W2.T shape: (10, 10)
    # dZ2 shape: (10, m)
    # ReLU_deriv(Z1) blocks gradients where Z1 ≤ 0
    dZ1 = torch.matmul(W2.T, dZ2) * ReLU_deriv(Z1)

    # dW1 = (1/m) * dZ1 · Xᵀ
    # Gradient of loss w.r.t W1
    # dZ1 shape: (10, m)
    # X.T shape: (m, 784)
    # Result dW1 shape: (10, 784)
    dW1 = (1 / m) * torch.matmul(dZ1, X.T)

    # db1 = (1/m) * sum(dZ1)
    # Gradient of loss w.r.t b1
    # Ideally should be: np.sum(dZ1, axis=1, keepdims=True)
    db1 = (1 / m) * torch.sum(dZ1, dim=1, keepdim=True)

    # Return all gradients
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions (A2):
    # A2 contains the output probabilities from the softmax layer.
    # Shape of A2: (10, m)
    # Each column corresponds to one input image,
    # and each row corresponds to a digit class (0 to 9).

    # np.argmax(A2, 0) returns the index of the maximum value
    # along axis = 0 (column-wise).
    # This index represents the predicted class for each sample.
    return torch.argmax(A2, dim=0)

def get_accuracy(predictions, Y):
    # Print the predicted labels and true labels (mainly for debugging and observation)
    # print(predictions, Y)

    # Compare predictions with true labels element-wise.
    # (predictions == Y) gives a boolean array:
    # True  where prediction is correct
    # False where prediction is wrong

    # np.sum(predictions == Y) counts how many predictions are correct
    # Y.size gives the total number of samples

    # Accuracy = (Number of correct predictions) / (Total number of samples)
    return (predictions == Y).float().mean().item()

def gradient_descent(X, Y, alpha, epochs, batch_size):
    # Initialize weights and biases for both layers
    # W1, b1 → parameters of hidden layer
    # W2, b2 → parameters of output layer
    W1, b1, W2, b2 = init_params()
    m = X.shape[1]

    print(f"Training Samples : {m}")
    print(f"Batch Size       : {batch_size}")
    print(f"Batches / Epoch  : {(m + batch_size - 1) // batch_size}")

    # Loop for a fixed number of training iterations (epochs)
    for epoch in range(epochs):
        perm = torch.randperm(m, device=device)

        X = X[:, perm]
        Y = Y[perm]
        # -------- Forward Propagation --------
        # Compute predictions using current weights and biases
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)

            X_batch = X[:, start:end]
            Y_batch = Y[start:end]

            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_batch)

            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X_batch, Y_batch)

            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)

        # -----------------------------
        # Training Accuracy
        # -----------------------------
        _, _, _, A2_train = forward_prop(W1, b1, W2, b2, X_train)
        train_loss = cross_entropy_loss(A2_train, Y_train)
        train_predictions = get_predictions(A2_train)

        train_accuracy = get_accuracy(
            train_predictions,
            Y_train
        )

        # -----------------------------
        # Validation Accuracy
        # -----------------------------
        _, _, _, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
        dev_predictions = get_predictions(A2_dev)
        dev_accuracy = get_accuracy(dev_predictions, Y_dev)
        print(
            f"Epoch {epoch + 1:3d}/{epochs}"
            f" | Loss: {train_loss:.4f}"
            f" | Train: {train_accuracy:.4f}"
            f" | Validation: {dev_accuracy:.4f}"
        )
    return W1, b1, W2, b2

# Train the neural network using gradient descent
# X_train → input training data (784 × number_of_samples)
# Y_train → true labels for training data
# 0.10    → learning rate (alpha)
# 500     → number of training iterations (epochs)
# The function returns the optimized weights and biases after training
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 10,  64)


def make_predictions(X, W1, b1, W2, b2):
    # Perform a forward pass through the trained network
    # We only care about A2 (the final output probabilities),
    # so the intermediate values Z1, A1, Z2 are ignored using '_'
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)

    # Convert the probability outputs into actual class labels
    # by selecting the index with the highest probability
    _, _, _, A2_full = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2_full)
    # Return the predicted digit(s)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    # Select a single image from the training set using the given index.
    # X_train has shape (784, m), where:
    # 784 → number of pixels in one image (28×28)
    # m   → number of training samples
    #
    # X_train[:, index] extracts one image as a vector of shape (784,)
    # Adding None reshapes it to (784, 1), which is required for matrix operations
    # in forward propagation.
    current_image = X_train[:, index, None]

    # Use the trained model to predict the digit for this single image.
    # make_predictions performs forward propagation and selects the class
    # with the highest probability.
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)


    # Get the true label of this image from the training labels
    label = Y_train[index]

    # Print the predicted digit and the actual digit
    print("Prediction: ", prediction)
    print("Label: ", label)

    # Reshape the flattened image vector (784,) back into its original
    # 2D image form (28, 28) so that it can be displayed.
    # Multiply by 255 to reverse the normalization and restore pixel values
    # to the original grayscale range [0, 255].
    current_image = current_image.reshape((28, 28)) * 255

    # Set the color map to grayscale
    plt.gray()

    # Display the image
    plt.imshow(current_image, interpolation='nearest')

    # Show the plot window
    plt.show()

print("Time:", time.time() - start)
print(data.shape)
