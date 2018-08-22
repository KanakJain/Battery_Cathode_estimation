from sklearn.neural_network import MLPClassifier
from Preprocess import *
from sklearn.metrics import accuracy_score

mlp = MLPClassifier(random_state=0, hidden_layer_sizes=15).fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score(X_train_scaled, y_train)))
p = mlp.predict(X_test_scaled)
print(accuracy_score(y_test, p))


