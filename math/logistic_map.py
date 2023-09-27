# importing the required modules
import matplotlib.pyplot as plt
import numpy as np
import math

def bern_f(x):
    return x**2

def comb(n, k):
    return math.factorial(n)/(math.factorial(k) * math.factorial(n-k))

def bern_p_sec(x):
    return bern_f(0)*((1-x)**2) + 2*bern_f(0.5)*(1-x)*x + bern_f(1)*(x**2)

def bern_p_four(x):
    return bern_f(0)*((1-x)**4) + 4*bern_f(0.25)*((1-x)**3) * x + 6* bern_f(0.5)* ((1-x)**2) *(x**2) + 4*bern_f(0.75)*(1-x) * (x**3) + bern_f(1)*(x**4)

def bern_p_five(x):
    return bern_f(0)*((1-x)**5) + 2*bern_f(0.25)*((1-x)**3) * x + bern_f(0.5)* ((1-x)**2) *(x**2) + 2*bern_f(0.75)*(1-x) * (x**3) + bern_f(1)*(x**4)


def logistic_map(x, y):
	y_next = y * x * (1 - y)
	x_next = x + 0.000001
	yield x_next, y_next

def logistic_map(x, y):
	y_next = y * x * (1 - y)
	x_next = x + 0.000001
	yield x_next, y_next

def logistic_map_b(x, y):
	y_next = x * bern_p_four(y)
	x_next = x + 0.000001
	yield x_next, y_next


steps = 3000000

Y = np.zeros(steps + 1)
X = np.zeros(steps + 1)

X[0], Y[0] = 1, 0.5

# map the equation to array step by step using the logistic_map function above
for i in range(steps):
	x_next, y_next = next(logistic_map_b(X[i], Y[i])) # calls the logistic_map function on X[i] as x and Y[i] as y
	X[i+1] = x_next
	Y[i+1] = y_next

plt.style.use('dark_background')
plt.figure(figsize=(10, 10))
plt.plot(X, Y, '^', color='white', alpha=0.4, markersize = 0.013)
plt.axis('on')
plt.show()



#steps = 50
#x = np.zeros(steps + 1)
#y = np.zeros(steps + 1)
#x[0], y[0] = 0, 0.4

#r = 2

# loop over the steps and replace array values with calculations
#for i in range(steps):
    #y[i+1] = r * y[i] * (1 - y[i])
    #y[i+1] = bern_p_sec(y[i])
    #x[i+1] = x[i] + 1

# plot the figure!
#fig, ax = plt.subplots()
#ax.plot(x, y, alpha=0.5)
#ax.set(xlabel='Time (years)', ylabel='Population (fraction of max)')
#plt.show()

