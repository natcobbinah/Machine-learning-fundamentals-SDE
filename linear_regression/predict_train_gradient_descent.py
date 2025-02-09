import numpy as np 
import matplotlib.pyplot as plt

# X = input(reservations), w= weight, b=bias
def predict(X, w, b):
    return X * w + b 

# square error after randomly choosing a weight value
# X = input(reservations), Y = (label)->Number of pizzas, w = weight, b = bias
def loss(X,Y, w, b):
    return np.average((predict(X,w,b)-Y) ** 2)

# L = 1/m (Sum[i=1, m]) * (wx1 + b - y1) ** 2
# Change_L / Change_W = 2/m(sum[i=1, m] * x(wx1 + b) - y1)
def gradient(X,Y,w):
    return 2 * np.average(X * (predict(X,w,0) - Y))


# X = input(reservations), Y(label)->Number of pizzas, iterations, lr=learning rate
# used to adjust the weight value, so as to have a better predicting weight value
def train(X, Y, iterations, lr):
    w = 0
    for i in range(iterations):
        print("Iterations %4d => Loss: %.10f" % (i, loss(X,Y,w, 0)))
        w -= gradient(X,Y,w) * lr 
    return w

# Import the dataset 
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

# Train the system 
w = train(X, Y, iterations=100, lr=0.001)
print("\nw=%.10f" % (w))



