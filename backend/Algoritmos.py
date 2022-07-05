from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
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
            info.append("PRED = {}".format(np.round(pred, 3)))

        info.append("RMSE = {}".format(np.round(rmse, 3)))
        info.append(" COEF = {}".format(np.round(coef, 3)))
        info.append("R2 = {}".format(np.round(r2, 3)))
        cons = np.round(regr.intercept_,3)
        info.append("Y = " + str(np.round(coef[0],3)) + "*X "+str(cons if cons < 0 else ("+" + str(cons))))

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
            info.append("PRED = {}".format(np.round(pred, 3)))

        info.append("GRADO = {}".format(degree))
        info.append("RMSE = {}".format(np.round(rmse, 3)))
        info.append(" COEF = {}".format(coef))
        info.append("R2 = {}".format(np.round(r2, 3)))

        strFuncs = ""
        for i in range(1, len(coef)):            
            var = ("*X^" + str(i)) if i > 1 else "*X "            
            cons = np.round(coef[i], 3) 
            strFuncs = str(cons if cons < 0 else ("+" + str(cons))) + var + " " + strFuncs

        cons = np.round(lin_reg2.intercept_,3)
        strFuncs = "Y = " + strFuncs + " " + str(cons if cons < 0 else (" +" + str(cons))) + '\n'

        info.append(strFuncs)

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

    # gaussiana
    def graf_gaus(self, data, cols, val):

        le = preprocessing.LabelEncoder()
        sizeH = len(cols)-1

        lines = []

        for i in range(sizeH):
            print(cols[i], le.fit_transform(np.asarray(data[cols[i]])))
            lines.append(le.fit_transform(np.asarray(data[cols[i]])))

        print(cols[sizeH], np.asarray(data[cols[sizeH]]))
        play = np.asarray(data[cols[sizeH]])

        features = list(zip(*lines))
        print("features", features)
        model = GaussianNB()
        model.fit(features, play)

        info = []

        str1 = ""
        for i in lines:
            str1 += str(i) + '\n'

        info.append(str1)

        if val != '':
            valS = val.split(',')
            valorPred = le.fit_transform(valS)

            predicted = model.predict([valorPred])  # sunny, hot, high, false
            print("PREDICT VALUE: ", predicted)
            info.append("Prediccion = {}".format(str(predicted)))

        return info

    # arbol de decision
    def graf_tree(self, data, cols, val):

        le = preprocessing.LabelEncoder()
        sizeH = len(cols)-1

        lines = []

        for i in range(sizeH):
            print(cols[i], le.fit_transform(np.asarray(data[cols[i]])))
            lines.append(le.fit_transform(np.asarray(data[cols[i]])))

        print(cols[sizeH], np.asarray(data[cols[sizeH]]))
        play = np.asarray(data[cols[sizeH]])

        features = list(zip(*lines))
        print("features", features)

        clf = DecisionTreeClassifier(max_depth=4, random_state=0)
        clf = clf.fit(features, play)

        info = []

        str1 = ""
        for i in lines:
            str1 += str(i) + '\n'

        info.append(str1)

        if val != '':
            valS = val.split(',')
            valorPred = le.fit_transform(valS)

            predicted = clf.predict([valorPred])
            print("PREDICT VALUE: ", predicted)
            info.append("Prediccion = {}".format(str(predicted)))

        plt.figure(figsize=(9, 9))
        plot_tree(clf, filled=True, fontsize=12)
        plt.savefig("arbol.png", format='png')
        plt.close()

        return info

    # redes neuronales
    def graf_neu_net(self, data, cols):

        le = preprocessing.LabelEncoder()
        sizeH = len(cols)-1

        lines = []

        for i in range(sizeH):
            print(cols[i], le.fit_transform(np.asarray(data[cols[i]])))
            lines.append(le.fit_transform(np.asarray(data[cols[i]])))

        print(cols[sizeH], np.asarray(data[cols[sizeH]]))
        y = np.asarray(data[cols[sizeH]])
        X = list(zip(*lines))

        X_train, X_test, y_train, y_test = train_test_split(X, y)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500, alpha=0.0001,
                            solver='adam', random_state=21, tol=0.000000001)
        #mlp = MLPClassifier(hidden_layer_sizes=(6,6,6,6),solver='lbfgs',max_iter=6000)
        mlp.fit(X_train, y_train)
        prediction = mlp.predict(X_test)

        classF = classification_report(y_test, prediction)

        info = []
        info.append(classF)

        return info
