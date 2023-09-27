import math
import copy

# calculate the determinant of the matrix
def determinant_calc(a):
    n = len(a)
    if n == 2:
        return a[0][0]*a[1][1] - a[1][0]*a[0][1]
    sum = 0
    for i in range(n):
        b = []
        for j in range(n):
            if j != i:
                b.append(a[j][1:])
        sum += a[i][0]*determinant_calc(b)*math.pow(-1,i)
    return sum


# p(x) = A_{n-1}x^{n-1}+...+A_{1}x+A_{0}
# p = [p(1),p(2),...,p(n)] - given
# Find A_{0} - ?

p = [10,62,242,682,1562]     
n = len(p)
a = []
for i in range(1,n+1):
    b = []
    for j in range(1,n+1):
        b.append(math.pow(i,n-j))
    a.append(b)

solutions = []
d = determinant_calc(a)
for i in range(n):
    di = []
    di = copy.deepcopy(a)
    for j in range(n):
        di[j][i] = p[j]
    did = determinant_calc(di)
    solutions.append(did/d)

print(f'all coeficients : {solutions}, the last one (A_0): {solutions[n-1]}')

# c = [[0,10,0,0,1],[1,0,1,0,0],[0,1,9425,1,0],[0,0,1,0,1],[1,0,0,100,0]]
# print(determinant_calc(c))