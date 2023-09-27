
# importing the required modules
import matplotlib.pyplot as plt
import numpy as np

def function(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))
  
# setting the x - coordinates
x = np.arange(-10, 10, 0.1)
# setting the corresponding y - coordinates
y = function(x)
  
# plotting the points
plt.plot(x, y)
  
# function to show the plot
plt.show()