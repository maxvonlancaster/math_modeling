from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Input data (for example - temperature)
array_of_data = [0,5,10,15,20,25,30,35,40]
X = np.array(array_of_data).reshape(len(array_of_data), 1)

# Power consumption for given temperature:
y = [10.0,7.5,8.5,4.0,5.0,6.0,7.0,7.5,10.5]


# Give data to the linear model:
model = LinearRegression(fit_intercept=True)
model.fit(X, y)

# predict on new data 
array_to_predict_l = []
a = min(array_of_data)
steps = 100
increase = (max(array_of_data) - min(array_of_data)) / steps
for i in range(steps):
    a += increase
    array_to_predict_l.append(a)

Xlinear = np.array(array_to_predict_l).reshape(len(array_to_predict_l), 1)
y_linear = model.predict(Xlinear)


# Give data to the non-linear model: (degree - is the degree of the polynomial)
svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=2, epsilon=0.1, coef0=1)
svr_poly.fit(X, y)

# predict on new data 
array_to_predict_poly = []
a = min(array_of_data)
steps = 100
increase = (max(array_of_data) - min(array_of_data)) / steps
for i in range(steps):
    a += increase
    array_to_predict_poly.append(a)

Xpoly = np.array(array_to_predict_poly).reshape(len(array_to_predict_poly), 1)
y_poly = svr_poly.predict(Xpoly)


plt.style.use('dark_background')
plt.figure(figsize=(10, 10))

# Lets take a look at the points on graph:
plt.plot(X, y, '^', color='white', alpha=0.4, markersize = 10.0)

# And on the predicted location of the new points for linear model:
plt.plot(Xlinear, y_linear, '^', color='blue', alpha=0.4, markersize = 2.0)

# And on the predicted location of the new points for the non-linear model:
plt.plot(Xpoly, y_poly, '^', color='red', alpha=0.4, markersize = 2.0)

plt.axis('on')
plt.show()
