from sklearn.datasets import make_classification, make_moons, make_circles
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits



"""SOME DATA GENERATORS"""

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=3)
X, y = make_moons(noise=0.1, n_samples=(100, 50))
X, y = make_circles(noise=0.1, n_samples=(100, 50))
print(X)
print(y)


plt.scatter(X[y==0][:, 0], X[y==0][:, 1], s=100, edgecolors='k')
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], s=100, edgecolors='k', marker='^')
plt.show()

