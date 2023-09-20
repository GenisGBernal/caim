import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

datos = {}

with open("C:/Users/thema/Desktop/proyectos/caim/lab/session1ESZipfHeaps/salida.txt", "r") as archivo:
    print(archivo)
    for linea in archivo:
        cantidad, palabra = linea.strip().split(", ")
        cantidad = int(cantidad)
        datos[palabra] = cantidad


print(len(datos))

datos_ordenados = dict(sorted(datos.items(), key=lambda x: x[1], reverse=True))

palabras = list(datos_ordenados.keys())
frecuencias = list(datos_ordenados.values())
posiciones = list(range(1, len(palabras) + 1))

plt.figure(figsize=(12, 6))  # Ajusta el tamaño del gráfico
plt.plot(posiciones, frecuencias, marker='o', linestyle='-')

plt.xlabel('Palabras')
plt.ylabel('Frecuencia')
plt.title('Frecuencia de Palabras')

plt.xticks(rotation=90)

plt.savefig('grafico_frecuencia_palabras.png')