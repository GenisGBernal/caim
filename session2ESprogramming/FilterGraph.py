import matplotlib.pyplot as plt
import numpy as np

# Datos de ejemplo: vector con el número de palabras y vector con tipos de tokenizadores
num_palabras = [167063, 59569, 61825, 54455]
tipos_tokenizador = ['Whitespace', 'Classic', 'Standard', 'Letter']

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))  # Tamaño de la figura

# Crear las barras del gráfico
plt.bar(tipos_tokenizador, num_palabras, color=plt.cm.Blues(np.linspace(0.5, 1, len(num_palabras))))

# Agregar etiquetas y título
plt.xlabel('Tipus de Tokenizador')
plt.ylabel('Número de Paraules')

# Agregar líneas horizontales discontinuas
plt.axhline(60000, linestyle='--', color='gray', alpha=0.5)

# Mostrar el gráfico
plt.box(False)
plt.tight_layout()  # Ajustar el diseño para que las etiquetas se vean bien
plt.show()
