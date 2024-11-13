import numpy as np
import matplotlib.pyplot as plt

weight = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
height = np.array([[ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
# Building Xbar 
one = np.ones((weight.shape[0], 1))
Xbar = np.concatenate((one, weight), axis = 1)

# Calculating weights of the fitting line 
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, height)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)
# Preparing the fitting line 
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1*x0

plt.plot(weight.T,height.T,marker='o',linestyle='',color='red')
plt.plot(x0,y0,linestyle='-',color='blue')
plt.xlabel('weight')
plt.ylabel('height')
plt.title("Linear regression of prediction weight or height")
plt.show()