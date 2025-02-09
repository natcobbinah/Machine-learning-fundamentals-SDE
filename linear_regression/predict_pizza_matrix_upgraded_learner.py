import numpy as np
x1, x2, x3, y = np.loadtxt("pizza_3_vars.txt", skiprows=1, unpack=True)
#print(x1.shape, x2.shape, x3.shape)

#joining the first three arrays together
X = np.column_stack((x1, x2, x3))
print(X.shape)

# first two rows
print(X[:2])

# convert 1-dimensional array into a matrix
Y = y.reshape(-1,1)
print(Y.shape)

# set weight to zeros per input variable
w = np.zeros((X.shape[1],1))
print(w.shape)

# upgrading the predict function
def predict(X,w):
    return np.matmul(X,w)

# X.T = transpose of X
def gradient(X,Y,w):
    return 2 * np.matmul(X.T, (predict(X,w) - Y))/ X.shape[0]