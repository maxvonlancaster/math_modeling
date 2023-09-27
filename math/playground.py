# import random

def _func(i, j):
    return i*i*i + j*j*j*j

for i in range(0,-10000,-1):
    for j in range(10000):
        if _func(i, j) == 7:
            print("i: " + i + "; j: " + j)

