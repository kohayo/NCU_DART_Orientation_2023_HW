import numpy as np

from sklearn import datasets

# Q1
iris = datasets.load_iris()

for i in range(len(iris.feature_names)):
    print(iris.feature_names[i])
    x = iris.data[:, i : i + 1]

    std_x = (x-np.mean(x)) / np.std(x)
    min_max_x = (x-min(x)) / (max(x)-min(x))

    print('std: ', std_x)
    print('min_max: ', min_max_x)

# Q2
A = np.array([[2, 1, 0],
             [1, 1, 2],
             [-1, 2, 1]])

B = np.array([[3, 1, -2],
             [3, -2, 4],
             [-3, 5, 1]])

print(np.linalg.inv(A))
print(np.linalg.inv(B))
print(A*B - B*A)
C = np.array([[-21],
              [0],
              [27]])
print(np.linalg.solve(A, C))

# Q3
p = np.array([1, 4, 3])
q = np.array([3, 2, 4])
print(p*q)
print(np.linalg.norm(p-q))
print((np.dot(p, q) / np.dot(q, q)) * q)
print((np.dot(p, q) / (np.linalg.norm(p)*np.linalg.norm(q))))
