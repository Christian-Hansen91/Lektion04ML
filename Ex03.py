from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn import tree

X, y = make_moons(n_samples=1000, noise=0.2, random_state=10)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
clf = tree.DecisionTreeClassifier().fit(train_X, train_y)
print("test size:", test_y.size)
test_score = clf.score(test_X, test_y)
print('test score: ' + str(test_score))
