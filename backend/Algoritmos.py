from PIL import Image, ImageFont, ImageDraw
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model, preprocessing
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn.naive_bayes import GaussianNB
matplotlib.use('Agg')


class Algoritmo():

    # Lineal
    def graf_linear(self, df, valx, valy, val, op):

        x = np.asarray(df[valx]).reshape(-1, 1)
        y = df[valy]

        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        y_pred = regr.predict(x)

        rmse = mean_squared_error(x, y_pred)
        coef = regr.coef_
        r2 = regr.score(x, y)

        info = []
        if op == 'Predicción de la tendencia' and val > 0:
            pred = regr.predict([[val]])            
            info.append("RMSE = {}".format(np.round(rmse, 3)))
            info.append(" COEF = {}".format(np.round(coef, 3)))
            info.append("R2 = {}".format(np.round(r2, 3)))
            info.append("PRED = {}".format(np.round(pred, 3)))
        else:
            info.append("RMSE = {}".format(np.round(rmse, 3)))
            info.append(" COEF = {}".format(np.round(coef, 3)))
            info.append("R2 = {}".format(np.round(r2, 3)))

        plt.title('Regresion Lineal \n')
        plt.grid()
        plt.xlim(min(x), max(x))
        plt.ylim(min(y), max(y))
        plt.xlabel(valx)
        plt.ylabel(valy)

        if op == 'Graficar puntos':
            plt.scatter(x, y, color='red')
        elif op == 'Función de tendencia' or op == 'Predicción de la tendencia':
            plt.scatter(x, y, color='red')
            plt.plot(x, y_pred, color='blue', linewidth=3)

        plt.savefig("linear.png", format='png')
        plt.close()
        return info

    # polinomial
    def graf_polinomial(self, df, valx, valy, val, degree, op):
        X = np.asarray(df[valx]).reshape(-1, 1)
        y = df[valy]

        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures

        poly_reg = PolynomialFeatures(degree=degree)
        X_poly = poly_reg.fit_transform(X)

        lin_reg2 = LinearRegression()
        lin_reg2.fit(X_poly, y)

        # coeficiente
        coef = lin_reg2.coef_
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

        info = []
        if op == 'Predicción de la tendencia' and val > 0:
            pred = lin_reg2.predict(poly_reg.fit_transform([[val]]))
            info.append("GRADO = {}".format(degree))
            info.append("RMSE = {}".format(np.round(rmse, 3)))
            info.append(" COEF = {}".format(coef))
            info.append("R2 = {}".format(np.round(r2, 3)))
            info.append("PRED = {}".format(np.round(pred, 3)))
        else:
            info.append("GRADO = {}".format(degree))
            info.append("RMSE = {}".format(np.round(rmse, 3)))
            info.append(" COEF = {}".format(coef))
            info.append("R2 = {}".format(np.round(r2, 3)))

        plt.title("Regresion Polinomial \n")
        plt.grid()
        plt.xlim(min(X), max(X))
        plt.ylim(min(y), max(y))
        plt.xlabel(valx)
        plt.ylabel(valy)

        if op == 'Graficar puntos':
            plt.scatter(X, y, color='red')
        elif op == 'Función de tendencia' or op == 'Predicción de la tendencia':
            plt.scatter(X, y, color='red')
            plt.plot(x_new, y_new, color='blue', linewidth=3)

        plt.savefig("polinomial.png", format='png')
        plt.close()
        return info

    def graf_gaus(self, data, cols, val):

        le = preprocessing.LabelEncoder()
        sizeH = len(cols)-1

        lines = []

        for i in range(sizeH):
            print(np.asarray(data[cols[i]]))
            lines.append(np.asarray(data[cols[i]]))

        print(np.asarray(data[cols[sizeH]]))
        play = np.asarray(data[cols[sizeH]])

        valS = val.split(',')
        valorPred = le.fit_transform(valS)

        features = list(lines)
        model = GaussianNB()
        model.fit(features, play)
        predicted = model.predict([valorPred])  # sunny, hot, high, false
        print("PREDICT VALUE: ", predicted)

        img = Image.new('RGB', (700, 500), color='white')
        d = ImageDraw.Draw(img)
        myFont = ImageFont.truetype('arial.ttf', 30)

        str1 = "Clasificador Gaussiano\n\n"
        for i in lines:
            str1 += str(i) + '\n'

        str1 += "\nValor de Prediccion: [" + val + "]\n"
        d.text((10, 10), str1, font=myFont, fill='black')
        img.save('gaus.png')

        info = []
        info.append("Prediccion = {}".format(str(predicted)))
        return info

    def graf_tree(self, data, cols, val):

        le = preprocessing.LabelEncoder()
        sizeH = len(cols)-1

        lines = []

        for i in range(sizeH):
            print(cols[i],np.asarray(data[cols[i]]))
            lines.append(np.asarray(data[cols[i]]))

        print(cols[sizeH],np.asarray(data[cols[sizeH]]))
        play = np.asarray(data[cols[sizeH]])

        valS = val.split(',')
        valorPred = le.fit_transform(valS)
        #valorPred = [int(i, base=16) for i in valS]

        features = list(lines)
        clf = DecisionTreeClassifier(max_depth=4,random_state=0)
        clf = clf.fit(features, play)
        predicted = clf.predict([valorPred])
        print("PREDICT VALUE: ", predicted)

        plt.figure()
        plot_tree(clf,filled=True,fontsize=12)
        plt.savefig("arbol.png", format='png')                
        plt.close()

        #img = Image.new('RGB', (700, 500), color='white')
        #d = ImageDraw.Draw(img)
        #myFont = ImageFont.truetype('arial.ttf', 30)

        str1 = ""        
        for i in lines:
            str1 += str(i) + '\n'

        #str1 += "\nValor de Prediccion: [" + val + "]\n"        
        #d.text((10, 10), str1, font=myFont, fill='black')
        
        info = []
        info.append(str1)
        info.append("Prediccion = {}".format(str(predicted)))
        return info