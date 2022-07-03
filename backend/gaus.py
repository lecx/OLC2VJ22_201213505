#Es utilizada como Gaussiana o distribución Normal
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

from PIL import Image, ImageFont, ImageDraw 

df = pd.read_excel('ent.xlsx')

header = ['A','B','C','D','E','F','G','H','I']

sizeH = len(header)-1

print("lenCol",sizeH)

lines = []

for i in range(sizeH):    
    print(np.asarray(df[header[i]]))
    lines.append(np.asarray(df[header[i]]))

print(np.asarray(df[header[sizeH]]))
play = np.asarray(df[header[sizeH]])

val = '8,5,1,4,6,3,0,0'.split(',')
valorPred = [int(i, base=16) for i in val]
print(valorPred)

features = list(lines)
print(features)
model = GaussianNB()
model.fit(features, play)
predicted = model.predict([valorPred]) #sunny, hot, high, false
print("PREDICT VALUE: ", predicted)

img = Image.new('RGB', (700, 500), color = 'white')
d = ImageDraw.Draw(img)
myFont = ImageFont.truetype('arial.ttf', 30)

str1 = "Clasificador Gaussiano\n\n"
for i in lines:
    str1 += str(i) + '\n'

str1 += "\nValor de Prediccion: " + str(valorPred) + "\n"
str1 += "\n\tPrediccion: " + str(predicted)
d.text((10,10), str1, font=myFont,fill='black')
img.save('gaus.png')