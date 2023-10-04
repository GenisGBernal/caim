import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

datos = {}

claves_a_eliminar = []

with open("salida.txt", "r") as archivo:
    print(archivo)
    for linea in archivo:
        cantidad, palabra = linea.strip().split(", ")
        cantidad = int(cantidad)
        datos[palabra] = cantidad

for clave, valor in datos.items():
    if not re.match('^[a-zA-Z]+$', clave) or clave in stopwords.words('english'):
        claves_a_eliminar.append(clave)

for clave in claves_a_eliminar:
    del datos[clave]


for clave, valor in dict(sorted(datos.items(), key=lambda x: x[1], reverse=True)).items():
            print(f'{valor}, {clave}')
