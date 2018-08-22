import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


X = pd.read_csv('Features1.csv')
y = pd.read_csv('Res.csv')
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y.values.ravel(), random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=0)
best_score = 0
print("Size of training set: {} size of validation set: {} size of test set:"" {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_valid, y_valid)
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}

svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)
print("Best score on validation set: {:.2f}".format(best_score))
print("Best parameters: ", best_parameters)
print("Test set score with best parameters: {:.2f}".format(test_score))

