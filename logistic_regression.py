from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()


# print(list(iris.keys()))
# print(iris.data)
# print(iris.data.shape)
# print(iris.target)
# print(iris.DESCR)


X = iris.data[:,3:]
y = (iris.target ==2).astype(np.int8)

print(X)
print(y)
# Train a logistic regression classifier

clf = LogisticRegression()
clf.fit(X,y)

example = clf.predict([[1.660413]])
print(example)

# Using matplotlib to plot the visualization
X_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba(X_new)

plt.plot(X_new, y_prob[:,1],"g-")
plt.show()
