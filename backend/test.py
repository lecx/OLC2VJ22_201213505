from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from audioop import rms
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('nac.csv')

X = np.asarray(df['Ano']).reshape(-1, 1)
y = df['Republica']


degree = 2
poly_reg = PolynomialFeatures(degree=degree)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# para rmse
Y_new = lin_reg2.predict(X_poly)

rmse = mean_squared_error(y, Y_new)
r2 = r2_score(y, Y_new)

valMinX = str(min(X))
valMaxX = str(max(X))

x_new_min = float(valMinX[1:(len(valMinX)-1)])
x_new_max = float(valMaxX[1:(len(valMaxX)-1)])

x_new = np.linspace(x_new_min, x_new_max, 50)
x_new = x_new[:, np.newaxis]

x_new_transfer = poly_reg.fit_transform(x_new)

y_new = lin_reg2.predict(x_new_transfer)

plt.plot(x_new, y_new, color='coral', linewidth=3)
plt.grid()
plt.xlim(min(X), max(X))
plt.ylim(min(y), max(y))

pred = lin_reg2.predict(poly_reg.fit_transform([[2030]]))


title = 'GRADO={}; RMSE={};  R2={}; PRED={}'.format(
    degree, np.round(rmse, 3), np.round(r2, 3), np.round(pred, 3))
plt.title("Regresion Polinomial \n" + title)
plt.xlabel('Ano')
plt.ylabel('Republica')
plt.savefig("polinomial.png", format='png')
plt.show()
plt.close()
