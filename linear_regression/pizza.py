import numpy as np 
import matplotlib.pyplot as plt

plt.axis([0, 50, 0, 50])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Reservations", fontsize=14)
plt.ylabel("Pizzas", fontsize=14)
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
plt.plot(X,Y, "bo")
plt.show()
#print(X[0:5])
#print(Y[0:5])