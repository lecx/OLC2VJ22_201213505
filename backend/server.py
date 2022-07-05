import base64
from flask import Flask, request, jsonify
import pandas as pd

from Algoritmos import Algoritmo
ope_algo = Algoritmo()

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route('/api/')
def index():
    return jsonify("estamos vivos!!!")


@app.route('/api/algoritmo', methods=["POST"])
def do_calculate():
  #  try:
        params = request.get_json(force=True)

        if not params:
            return jsonify({'code': '99', 'error': "Parametros incompletos, intente nuevamente."}), 200

        algo = params['algo']

        if algo is None:
            return jsonify({'code': '99', 'error': "Algoritmo no definido, intente nuevamente."}), 200

        op = params['op']
        val = params['val']

        data = params['data']
        if data is None:
            return jsonify({'code': '99', 'error': "Datos no enviados, intente nuevamente."}), 200

        resp = "No fue posible procesar la solicitud."
        info = []

        if algo == 'Regresión lineal':

            x = params['x']
            if x is None:
                return jsonify({'code': '99', 'error': "Columna X no definida, intente nuevamente."}), 200

            y = params['y']
            if y is None:
                return jsonify({'code': '99', 'error': "Columna Y no definida, intente nuevamente."}), 200

            if op == 'Predicción de la tendencia' and val == 0:
                return jsonify({'code': '99', 'error': "Operacion de prediccion necesita un valor valido, intente nuevamente."}), 200

            df = pd.read_json(data, orient='columns')
            info = ope_algo.graf_linear(df, x, y, val, op)

            with open("linear.png", "rb") as img_file:
                benc = base64.b64encode(img_file.read()).decode('utf-8')
                #b = "data:image/png;base64,"+benc
                resp = benc

        elif algo == 'Regresión polinomial':
            x = params['x']
            if x is None:
                return jsonify({'code': '99', 'error': "Columna X no definida, intente nuevamente."}), 200

            y = params['y']
            if y is None:
                return jsonify({'code': '99', 'error': "Columna Y no definida, intente nuevamente."}), 200

            degree = params['degree']

            if degree is None:
                return jsonify({'code': '99', 'error': "Datos no enviados, intente nuevamente."}), 200

            if op == 'Predicción de la tendencia' and val == 0:
                return jsonify({'code': '99', 'error': "Operacion de prediccion necesita un valor valido, intente nuevamente."}), 200

            df = pd.read_json(data, orient='columns')
            info = ope_algo.graf_polinomial(df, x, y, val, degree, op)

            with open("polinomial.png", "rb") as img_file:
                benc = base64.b64encode(img_file.read()).decode('utf-8')
                #b = "data:image/png;base64,"+benc
                resp = benc

        elif algo == 'Clasificador Gaussiano':
            if op != 'Clasificacion':
                return jsonify({'code': '99', 'error': "Operacion no valida, intente nuevamente."}), 200

            columns = params['columns']
            if columns is None or not columns:
                return jsonify({'code': '99', 'error': "No se seleccionaron columnas, intente nuevamente."}), 200

            val = params['valC']
            #if val is None or val == '':
             #   return jsonify({'code': '99', 'error': "No se envio valor a clasificar, intente nuevamente."}), 200

            df = pd.read_json(data, orient='columns')
            info = ope_algo.graf_gaus(df, columns, val)

            resp = ""
            #with open("gaus.png", "rb") as img_file:
            #    benc = base64.b64encode(img_file.read()).decode('utf-8')
            #    #b = "data:image/png;base64,"+benc
            #   resp = benc

        elif algo == 'Clasificador de árboles de decisión':
            if op != 'Clasificacion':
                return jsonify({'code': '99', 'error': "Operacion no valida, intente nuevamente."}), 200

            columns = params['columns']
            if columns is None or not columns:
                return jsonify({'code': '99', 'error': "No se seleccionaron columnas, intente nuevamente."}), 200

            val = params['valC']
            #if val is None or val == '':
             #   return jsonify({'code': '99', 'error': "No se envio valor a clasificar, intente nuevamente."}), 200

            df = pd.read_json(data, orient='columns')
            info = ope_algo.graf_tree(df, columns, val)

            with open("arbol.png", "rb") as img_file:
                benc = base64.b64encode(img_file.read()).decode('utf-8')
                #b = "data:image/png;base64,"+benc
                resp = benc

        elif algo == 'Redes neuronales':
            if op != 'Clasificacion':
                return jsonify({'code': '99', 'error': "Operacion no valida, intente nuevamente."}), 200

            columns = params['columns']
            if columns is None or not columns:
                return jsonify({'code': '99', 'error': "No se seleccionaron columnas, intente nuevamente."}), 200

            resp = ""
            df = pd.read_json(data, orient='columns')
            info = ope_algo.graf_neu_net(df, columns)

        return jsonify({'code': '00', 'img': resp, 'info': info}), 200
#    except Exception:
 #       return jsonify({'code': '99', 'error': "Error interno al realizar operacion, no bajar puntos Please!!"}), 200


# Main
if __name__ == "__main__":
    app.run("localhost", port=5000)
