import base64
import io
from itsdangerous import exc
import streamlit as st
import requests
import pandas as pd
from PIL import Image

def fetch(session, url):
    try:
        result = session.get(url)
        return result.json()
    except Exception:
        return {}

def post(session,url,params):
    try:
        result = session.post(url,json=params)
        return result.json()
    except Exception:
        return None

def main():
    try:

        st.set_page_config(page_title="OLC2_VJ22", page_icon="❤🤖❤",layout="wide")
        st.title("Machine Learning")
        session = requests.Session()
        dataResponse = None

        col1, col2 = st.columns(2)

        with col1:
            spectra = st.file_uploader("Cargar Archivo", type={"csv", "xls","xlsx","json"})
            data = None
            header = None
            if spectra is not None:
                ext = spectra.name.split('.')[1]
                
                if ext == 'csv':
                    data = pd.read_csv(spectra)
                elif ext == 'xls' or ext == 'xlsx':
                    data = pd.read_excel(spectra)
                elif ext == 'json':
                    data = pd.read_json(spectra)

                header = data.columns.values.tolist()            
                x = st.selectbox('col X ' ,header)
                y = st.selectbox('col Y ' ,header)
                val = st.number_input('Valor Prediccion ', min_value=0, value=0 )
                valDegree = st.number_input('Grado ', min_value=1, value=1 )

            with st.container():

                options = st.multiselect('Columnas (Gausiana y Arbol de Desición)', header if header else [],[])
                valPredC = st.text_input('Valor Calculo (Gausiana y Arbol de Desición)',value="", autocomplete=None, placeholder='Valor sin corchetes')

            with st.form("form1"):
                algo = st.selectbox('Algoritmo', [ 'Regresión lineal','Regresión polinomial','Clasificador Gaussiano'
                                        ,'Clasificador de árboles de decisión','Redes neuronales'], key=1)  
                
                op = st.selectbox('Operaciones', ['Graficar puntos','Función de tendencia'
                                        ,'Predicción de la tendencia','Clasificacion'], key=2)  
                
                submitted = st.form_submit_button("generar grafica")

                if submitted:
                    json = {'data':data.to_json(orient = 'columns'),'x':x,'y':y, 'val':val,'algo':algo,'op':op,'degree':valDegree,'columns':options,'valC':valPredC}
                    #dataResponse = post(session,"http://lecx.pythonanywhere.com/api/algoritmo",json) 
                    dataResponse = post(session,"http://localhost:5000/api/algoritmo",json) 
    
        with col2:
            st.subheader("Resultados")

            if data is not None:                
                st.write(data)

            if dataResponse is not None:
                if dataResponse['code'] == '00':
                        st.success("Operacion procesada.")
                                            
                        if dataResponse['img'] is not None and dataResponse['img'] != "":
                            im = Image.open(io.BytesIO(base64.b64decode(dataResponse['img'])))
                            st.image(im)

                        for tmp in dataResponse['info']:                         
                            st.code(tmp,language='python')
                else:
                    st.error(dataResponse['error'])
    except Exception:
        st.error("Favor ingrese los valores correctos.")

if __name__ == '__main__':
    main()
