import matplotlib.pyplot as plt
import numpy as np

num_palabras = [54455, 67106, 67099, 44872, 45523, 51685]
tipos_filtros = ['Lowercase', 'Asciifolding', 'Stop', 'Snowball', 'Porter Stem', 'Kstem']

plt.figure(figsize=(10, 6))
plt.bar(tipos_filtros, num_palabras, color=plt.cm.Greens(np.linspace(0.5, 1, len(num_palabras))))
plt.xlabel('Tipus de Filtre')
plt.ylabel('Número de Paraules')
plt.axhline(60000, linestyle='--', color='gray', alpha=0.5)
plt.axhline(50000, linestyle='--', color='gray', alpha=0.5)

plt.box(False)
plt.tight_layout()
plt.show()
