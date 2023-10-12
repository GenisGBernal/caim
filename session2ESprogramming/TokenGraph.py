import matplotlib.pyplot as plt
import numpy as np

num_palabras = [167063, 59569, 61825, 54455]
tipos_tokenizador = ['Whitespace', 'Classic', 'Standard', 'Letter']

plt.figure(figsize=(10, 6))
plt.bar(tipos_tokenizador, num_palabras, color=plt.cm.Blues(np.linspace(0.5, 1, len(num_palabras))))
plt.xlabel('Tipus de Tokenizador')
plt.ylabel('Número de Paraules')
plt.axhline(60000, linestyle='--', color='gray', alpha=0.5)

plt.box(False)
plt.tight_layout()
plt.show()
