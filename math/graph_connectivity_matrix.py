import numpy as np
import matplotlib.pyplot as plt
import math

# return matrix a to the power of i
def matrix_power(a, i):
    b = a
    for j in range(1,i):
        b = np.dot(b,a)
    print(f'a to the power: {i}')
    print(b)
    return b


# calculate the connectivity matrix for graph, given its adjacency matrix a
a=np.array([[0,1,1,0,1],[1,0,0,1,1],[1,0,0,0,1],[0,1,0,0,0],[1,1,1,0,0]])
I=np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])

connectivity_matrix = I + a
n = int(math.sqrt(a.size))
for i in range(2,n):
    connectivity_matrix += matrix_power(a,i)

print('connectivity matrix for graph: ')
print(connectivity_matrix)

# print(np.linalg.det(a))

Y = np.zeros(n)
X = np.zeros(n)
for i in range(n):
    X[i] = math.cos(2 * math.pi * i / n)
    Y[i] = math.sin(2 * math.pi * i / n)

plt.style.use('dark_background')
plt.figure(figsize=(10, 10))
plt.plot(X, Y, '^', color='white', alpha=0.4, markersize = 10.0)
plt.axis('on')
plt.show()
