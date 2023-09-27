import numpy as np
import pylab as pl
from matplotlib import collections  as mc
import math

graph = {
  '0' : ['1','2'],
  '1' : ['3', '4'],
  '2' : ['5'],
  '3' : [],
  '4' : ['5'],
  '5' : []
}

graph_disc = {
  '0' : ['1'],
  '1' : [],
  '2' : [],
  '3' : [],
  '4' : [],
  '5' : []
}

visited = [] # List for visited nodes.
queue = []     #Initialize a queue

# Breadth First Search in Python
def bfs(visited, graph, node): #function for BFS
  visited.append(node)
  queue.append(node)

  while queue:          # Creating loop to visit each node
    m = queue.pop(0) 
    print (m, end = " ") 

    for neighbour in graph[m]:
      if neighbour not in visited:
        visited.append(neighbour)
        queue.append(neighbour)


# Depth First Search
visiteddfs = set() # Set to keep track of visited nodes of graph.

def dfs(visited, graph, node):  #function for dfs 
    if node not in visited:
        print (node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)


# Driver Code
print("Following is the Breadth-First Search")
bfs(visited, graph, '5')    # function calling

n = len(graph)
Y = np.zeros(n)
X = np.zeros(n)
for i in range(n):
    X[i] = math.cos(2 * math.pi * i / n)
    Y[i] = math.sin(2 * math.pi * i / n)

pl.style.use('dark_background')
pl.figure(figsize=(10, 10))
pl.plot(X, Y, '^', color='white', alpha=0.4, markersize = 10.0)

lines = []
for key, value in graph.items():
    for item in value:
      a = [X[int(key)],Y[int(key)]]
      pl.plot([X[int(key)],X[int(item)]], [Y[int(key)],Y[int(item)]], color='white', alpha=0.4, lw=5.0)
      # break
      # lines.append(( (X[int(key)],Y[(int(key))]), ((X[int(item)]),Y[int(item)]) ))

c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])
lc = mc.LineCollection(lines, colors=c, linewidths=5.0)
# pl.plot(lc)



# fig, ax = pl.subplots()
# ax.add_collection(lc)

# pl.axes().add_collection(lc)

pl.axis('on')
pl.show()