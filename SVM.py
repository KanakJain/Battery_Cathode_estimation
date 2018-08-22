from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from Preprocess import *

svm = SVC(kernel='rbf', C=10, gamma=0.1)
k = KFold(n_splits=10, shuffle=True, random_state=7)
scores = cross_val_score(svm, X, y.values.ravel(), cv=k)
print("Cross-validation scores: {}".format(scores))
svm.fit(X_train_scaled, y_train)
p = svm.predict(X_test_scaled)
print(accuracy_score(y_test, p))
