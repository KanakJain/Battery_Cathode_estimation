import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


X = pd.read_csv('Features1.csv')
y = pd.read_csv('Res.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
tree = DecisionTreeClassifier(max_depth=8)
tree.fit(X_train, y_train)
tree.fit(X_train, y_train.values.ravel())
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
plt.barh(range(5), tree.feature_importances_, align='center')
plt.show()

