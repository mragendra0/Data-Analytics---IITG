import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

# Find the values
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

# Reshape the training and test data sets
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))


train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.



# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize the parameters
def initialize_parameters(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b

# Compute the cost
def compute_cost(a, y):
    m = y.shape[1]
    cost = -1/m * np.sum(y * np.log(a) + (1-y) * np.log(1-a))
    return cost

# Perform gradient descent to learn parameters
def optimize(w, b, X, Y, num_iterations, learning_rate):
    costs = []

    for i in range(num_iterations):
        m = X.shape[1]

        # Forward propagation
        z = np.dot(w.T, X) + b
        a = sigmoid(z)

        # Compute cost
        cost = compute_cost(a, Y)

        # Backward propagation
        dw = 1/m * np.dot(X, (a - Y).T)
        db = 1/m * np.sum(a - Y)

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the cost
        if i % 100 == 0:
            costs.append(cost)

    return w, b, costs

# Make predictions
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))

    # Compute the activation
    z = np.dot(w.T, X) + b
    a = sigmoid(z)

    # Convert probabilities to binary predictions
    Y_prediction = np.round(a)

    return Y_prediction

# Define the model
def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate):
    # Initialize parameters
    w, b = initialize_parameters(X_train.shape[0])

    # Gradient descent
    w, b, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    # Make predictions
    Y_train_prediction = predict(w, b, X_train)
    Y_test_prediction = predict(w, b, X_test)

    # Calculate accuracy
    train_accuracy = 100 - np.mean(np.abs(Y_train_prediction - Y_train)) * 100
    test_accuracy = 100 - np.mean(np.abs(Y_test_prediction - Y_test)) * 100

    # Print accuracy
    print("Train accuracy: " + str(train_accuracy) + "%")
    print("Test accuracy: " + str(test_accuracy) + "%")

    return train_accuracy, test_accuracy

# Set hyperparameters
num_iterations = 2000
learning_rate = 0.005

# Train the model and evaluate
train_accuracy, test_accuracy = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations, learning_rate)



print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))

# Initialize the parameters
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b

dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))

def propagate(w, b, X, Y):
    m = X.shape[1]  # number of examples

    # Forward propagation
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # Backward propagation
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    # Reshape dw to have the same shape as w
    dw = dw.reshape(w.shape)

    return cost, dw, db
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):
        # Forward and Backward propagation
        cost, dw, db = propagate(w, b, X, Y)

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the cost
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}

    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)


    # Calculate predictions
    A = sigmoid(np.dot(w.T, X) + b)
    Y_prediction = np.where(A > 0.5, 1, 0)

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost=False):
    # Initialize parameters
    w, b = initialize_parameters(X_train.shape[0])

    # Optimize parameters
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters
    w = params["w"]
    b = params["b"]

    # Make predictions
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # Print train/test accuracy
    train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
    print("Train accuracy: " + str(train_accuracy) + "%")
    print("Test accuracy: " + str(test_accuracy) + "%")

    # Package results
    d = {
        "w": w,
        "b": b,
        "costs": costs,
        "Y_prediction_train": Y_prediction_train,
        "Y_prediction_test": Y_prediction_test,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    }

    return d

# Example of a picture that was wrongly classified.
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations, learning_rate, print_cost=True)
train_accuracy = d["train_accuracy"]
test_accuracy = d["test_accuracy"]

print("Training Accuracy: " + str(train_accuracy) + "%")
print("Test Accuracy: " + str(test_accuracy) + "%")
index = 1
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") +  "\" picture.")

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.show()













