from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

# print(iris.keys())
# print(iris.data)
# print(iris.target)

features = iris.data
labels = iris.target

clf = KNeighborsClassifier()
clf.fit(features, labels)

pred = clf.predict([[9,9,3,3],[0,0,0,0]])
print(pred)