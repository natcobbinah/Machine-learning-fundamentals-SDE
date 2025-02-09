import numpy as np

#calling the predict() function
def predict(X,w,b):
    return X * w + b 

# Calculating the loss 
def loss(X,Y,w,b):
    return np.average((predict(X,w,b)-Y) ** 2)

# computing the derivative
def gradient(X,Y,w,b):
    w_gradient = 2 * np.average(X * (predict(X,w,b) - Y))
    b_gradient = 2 * np.average(predict(X,w,b) - Y)
    return (w_gradient, b_gradient)

# calling the training function for 20,000 iterations
def train(X,Y, iterations, lr):
    w = b = 0 
    for i in range(iterations):
        if (i % 5000 == 0):
            print("Iterations %4d => Loss: %.10f" % (i,  loss(X,Y,w,b)))
        
        w_gradient, b_gradient = gradient(X,Y,w,b)
        w -= w_gradient * lr 
        b -= b_gradient * lr 
    return w,b

# loading the data and then calling the desired functions
X,Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
w,b = train(X,Y, iterations=20_000, lr= 0.003)
print("\nw=%.10f, b=%.10f" % (w,b))
print("Prediction: x=%d => y=%.2f" % (20, predict(20,w,b)))