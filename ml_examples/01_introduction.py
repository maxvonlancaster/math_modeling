from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Linear regression is an algorithm that provides a linear relationship between an 
# independent variable and a dependent variable to predict the outcome of future events.

# Input data (for example - temperature)
array_of_data = [0,5,10,15]
X = np.array(array_of_data).reshape(len(array_of_data), 1)

# Power consumption for given temperature:
y = [10.0,7.5,8.5,4]

# Give data to the model:
model = LinearRegression(fit_intercept=True)
model.fit(X, y)

# predict on new data 
array_to_predict = []
a = min(array_of_data)
steps = 1000
increase = (max(array_of_data) - min(array_of_data)) / steps
for i in range(steps):
    a += increase
    array_to_predict.append(a)

Xnew = np.array(array_to_predict).reshape(len(array_to_predict), 1)
ynew = model.predict(Xnew)

plt.style.use('dark_background')
plt.figure(figsize=(10, 10))

# Lets take a look at the points on graph:
plt.plot(X, y, '^', color='white', alpha=0.4, markersize = 10.0)

# And on the predicted location of the new points:
plt.plot(Xnew, ynew, '^', color='blue', alpha=0.4, markersize = 1.0)

plt.axis('on')
plt.show()

