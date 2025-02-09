import numpy as np

# X = input(reservations), w= weight
def predict(X, w):
    return X * w

# square error after randomly choosing a weight value
# X = input(reservations), Y = (label)->Number of pizzas, w = weight
def loss(X,Y,w):
    return np.average((predict(X,w) - Y) ** 2)

# X = input(reservations), Y(label)->Number of pizzas, iterations, lr=learning rate
# used to adjust the weight value, so as to have a better predicting weight value
def train(X, Y, iterations, lr):
    w = 0 
    for i in range(iterations):
        current_loss = loss(X,Y,w)
        print("Iteration %4f  => loss: %.6f" % (i, current_loss))

        if loss(X,Y, w + lr) < current_loss:
            w += lr 
        elif loss(X,Y, w - lr) < current_loss:
            w -= lr 
        else:
            return w 
    
    raise Exception("Couldn't converge within %d iterations" % iterations)

""" # testing examples
# assuming w=1.5
print(predict(14, 1.5)) # => 21

# average loss computed for all reseravtions against pizzas with weight 1.5
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
print(loss(X, Y,1.5)) """

# import the dataset
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

# train the system
w = train(X,Y, iterations=10000, lr=0.01) 
print("\nw=%.3f" % w)

# predict the number of pizzas 
print("Prediction: x=%d => y=%.2f" % (20, predict(20,w))) 