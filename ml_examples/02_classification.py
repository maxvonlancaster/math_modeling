from sklearn.svm import LinearSVC, SVC
import matplotlib.pyplot as plt

# A linear classifier is a model that makes a decision to categories a set of data points to a 
# discrete class based on a linear combination of its explanatory variables.

clf = SVC(C=1e4, kernel='linear') # or
clf = LinearSVC(C=1e4, loss='hinge', max_iter=100000)

# Array of point on graph (for example - pixel area of cells and amount of concave points of a cell)
X = [[10,0],[10,1],[20,0],[20,1],[20,4],[20,3],[30,1],[40,2],[30,3]]

# Information on whether the cell is cancerous or not - -1 is not cancerous and 1 is cancerous: 
y = [-1,-1,-1,-1,-1,1,1,1,1]

# Give data to the model
clf.fit(X, y) 

# Get the points into separate arrays - for better visual representation
x_0 = []
u_0, u_1, v_0, v_1 = [], [], [], []
for i in range(len(X)):
    x_0.append(X[i][0])
    if y[i] == -1:
        u_0.append(X[i][0])
        v_0.append(X[i][1])
    else:
        u_1.append(X[i][0])
        v_1.append(X[i][1])

# Get the coefficients of linear separation 
W=clf.coef_[0]
I=clf.intercept_
a = -W[0]/W[1]
b = I[0]/W[1]

lx, ly = [], []
steps = 100
t = min(x_0)
d = max(x_0) - min(x_0)
for i in range(steps):
    lx.append(t)
    ly.append(a * t - b)
    t += d/steps


plt.style.use('dark_background')
plt.figure(figsize=(10, 10))

# Here we will show our points on graph, that represent non-cancerous cells:
plt.plot(u_0, v_0, '^', color='blue', alpha=0.4, markersize = 10.0)

# here we wll have cancerous cells:
plt.plot(u_1, v_1, '^', color='red', alpha=0.4, markersize = 10.0)

# And a predicted line of separation:
plt.plot(lx, ly, '^', color='white', alpha=0.4, markersize = 1.0)
plt.axis('on')
plt.show()    
