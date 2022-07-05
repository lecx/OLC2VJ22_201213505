import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

le = preprocessing.LabelEncoder()

df = pd.read_csv('vgsales.csv')

lines  = []

PF = le.fit_transform(df['Platform'])
PS = le.fit_transform(df['Publisher'])
GS = le.fit_transform(df['Global_Sales'])

lines.append(PF)
lines.append(PS)
lines.append(GS)

X = list(zip(*lines))
y = np.asarray(df['Genre'])

X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500, alpha=0.0001,
                    solver='adam', random_state=21, tol=0.000000001)
#mlp = MLPClassifier(hidden_layer_sizes=(6,6,6,6),solver='lbfgs',max_iter=6000)
mlp.fit(X_train,y_train)
prediction = mlp.predict(X_test)

print(classification_report(y_test,prediction))