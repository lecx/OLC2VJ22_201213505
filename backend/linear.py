from audioop import rms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

df = pd.read_csv("nac.csv")

x = np.asarray(df['Ano']).reshape(-1,1)
y = df['Republica']

regr = linear_model.LinearRegression()
regr.fit(x,y)
y_pred = regr.predict(x)

rmse = mean_squared_error(x, y_pred)
coef = regr.coef_
r2 = regr.score(x, y)
pred = regr.predict([[2030]])
title = 'RMSE = {}; COEF = {}; R2= {}; PRED={}'.format(np.round(rmse,2),np.round(coef,2),np.round(r2,2),np.round(pred,2))

plt.title('Regresion Lineal \n' + title)
plt.scatter(x,y, color='red')
plt.plot(x,y_pred, color='blue',linewidth=3)
#plt.ylim(0,20)

print("RMSE: ",rmse)
print("coef: ",coef)
print("R2: ",r2)
print("Prediccion: ",pred)
plt.show()                   # Display the plotp
#plt.savefig("linear.png" , format='png')
#plt.close()