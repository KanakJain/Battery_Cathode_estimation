import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

X = pd.read_csv('Features1.csv')
y = pd.read_csv('Res.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
Gb = GradientBoostingClassifier(learning_rate=0.09, max_depth=2)
Gb.fit(X_train, y_train.values.ravel())
print("Accuracy on training set: {:.3f}".format(Gb.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(Gb.score(X_test, y_test)))