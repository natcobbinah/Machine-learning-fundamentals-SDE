import numpy as np 
import matplotlib.pyplot as plt

# X = input(reservations), w= weight, b=bias
def predict(X, w, b):
    return X * w + b 

# square error after randomly choosing a weight value
# X = input(reservations), Y = (label)->Number of pizzas, w = weight, b = bias
def loss(X,Y, w, b):
    return np.average((predict(X,w,b)-Y) ** 2)

def train(X,Y, iterations, lr):
    w = b = 0 

    for i in range(iterations):
        current_loss = loss(X,Y,w,b)

        if i % 300 == 0:
            print("Iterations %4d => Loss: %.6f" % (i, current_loss))
        
        if loss(X,Y, w + lr, b) < current_loss: # updating weight 
            w += lr 
        elif loss(X,Y, w - lr, b) < current_loss: #updating weight 
            w -= lr 
        elif loss(X,Y,w, b + lr)  < current_loss: #updating bias
            b += lr 
        elif loss(X,Y,w, b - lr) < current_loss: #updating bias 
            b -= lr 
        else: 
            return w,b
    raise Exception("Couldn't converge within %d iterations" % iterations)

# Import the dataset 
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

# Train the system 
w, b = train(X, Y, iterations=10000, lr=0.01)
print("\nw=%.3f, b=%.3f" % (w,b))

# predict the number of pizzas 
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w,b)))

plt.plot(X,Y, "bo")
plt.xlabel("Reservations")
plt.ylabel("Pizzas")
x_edge , y_edge = 50, 50
plt.axis([0, x_edge, 0, y_edge])
plt.plot([0, x_edge], [b, predict(x_edge, w, b)], linewidth=1.0, color='g')
plt.show()