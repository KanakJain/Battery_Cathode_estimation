from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

X = pd.read_csv('Features1.csv')
y = pd.read_csv('Res.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), random_state=0)
Scalar = MinMaxScaler()
Scalar.fit(X_train)
X_train_scaled = Scalar.transform(X_train)
Scalar.fit(X_test)
X_test_scaled = Scalar.transform(X_test)
print(type(X_train))
