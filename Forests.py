from sklearn.ensemble import RandomForestClassifier
from Preprocess import *

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
forests = RandomForestClassifier(n_estimators=5, random_state=2)
forests.fit(X_train, y_train.values.ravel())
print("Accuracy on training set: {:.3f}".format(forests.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forests.score(X_test, y_test)))
