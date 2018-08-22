from sklearn.neighbors import KNeighborsClassifier
from Preprocess import *
import matplotlib.pyplot as plt

train_accu = []
test_accu = []
neighbors = range(1, 11)
for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train.values.ravel())
    train_accu.append(knn.score(X_train, y_train))
    test_accu.append(knn.score(X_test, y_test))
plt.plot(neighbors, train_accu, label="Training accuracy")
plt.plot(neighbors, test_accu, label="Test accuracy")
plt.show()
# p = knn.predict(X_test)
# print(accuracy_score(y_test.values.ravel(), p))
