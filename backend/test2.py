import matplotlib.pyplot as plt
import pandas as pd;
from sklearn import preprocessing;
from sklearn.naive_bayes import GaussianNB;

df = pd.read_csv("pred.csv");

A = df['A']
B = df['B']
C = df['C']
D = df['D']
P = df['E']

print(A);
print(B);
print(C);
print(D);
print(P);


features = list(zip(A, B, C, D));
print(features);

model = GaussianNB();
model.fit(features, P);

predict = model.predict([[5, 500, 200, False]])
print("Predict value: ", predict);