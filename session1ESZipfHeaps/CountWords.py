"""
.. module:: CountWords

CountWords
*************

:Description: CountWords

    Generates a list with the counts and the words in the 'text' field of the documents in an index

:Authors: bejar
    

:Version: 

:Created on: 04/07/2017 11:58 

"""
import re
from matplotlib import pyplot as plt
import nltk
from nltk.corpus import stopwords
import numpy as np
from scipy.optimize import curve_fit
nltk.download('stopwords')
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from elasticsearch.exceptions import NotFoundError, TransportError

import argparse

__author__ = 'bejar'

a = 0.5

def func(x, b, c):
    return c/(x+b)**a

def zip(frecuencia_palabras, frecuencia_palabras_logaritimico, num_palabras, log_num_palabras):
    popt, pcov = curve_fit(func, num_palabras, frecuencia_palabras)
    print('b = %d, c = %d' % (popt[0],popt[1]))
    fitArray = []
    logFitArray = []
    for num in frecuencia_palabras:
        fitArray.append(func(num, *popt))
        logFitArray.append(np.log(func(num, *popt)))

    print("Calculo zips hecho")
    
    zip_lineal(num_palabras, frecuencia_palabras, fitArray)

def zip_lineal(num_palabras, frecuencia_palabras, fitArray):
    plt.plot(num_palabras, frecuencia_palabras, 'b-', label='Actual frequencies')
    plt.plot(num_palabras, fitArray,'r-',label='Zipf\'s fit')
    plt.legend()
    plt.xlabel('x = Rank of the word (sorted by most frequent)')
    plt.ylabel('y = Frequency of the word')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=None, required=True, help='Index to search')
    parser.add_argument('--alpha', action='store_true', default=False, help='Sort words alphabetically')
    args = parser.parse_args()

    index = args.index

    try:
        client = Elasticsearch(hosts='http://localhost:9200')
        voc = {}
        sc = scan(client, index=index, query={"query" : {"match_all": {}}})
        for s in sc:
            try:
                tv = client.termvectors(index=index, id=s['_id'], fields=['text'])
                if 'text' in tv['term_vectors']:
                    for t in tv['term_vectors']['text']['terms']:
                        if t in voc:
                            voc[t] += tv['term_vectors']['text']['terms'][t]['term_freq']
                        else:
                            voc[t] = tv['term_vectors']['text']['terms'][t]['term_freq']
            except TransportError:
                pass
        lpal = []

        for v in voc:
            lpal.append((v.encode("utf-8", "ignore"), voc[v]))

        # print("Datos obtenidos")

        # datos = {}

        # for clave, valor in reversed(sorted(lpal, key=lambda x: x[0 if args.alpha else 1])):
        #     datos[clave] = valor

        # claves_a_eliminar = []

        # for clave, valor in datos.items():
        #     if not re.match('^[a-zA-Z]+$', clave) or clave in stopwords.words('english'):
        #         claves_a_eliminar.append(clave)

        # for clave in claves_a_eliminar:
        #     del datos[clave]

        # print("Datos limpios")


        # print("Datos ordenados")

        # frecuencia_palabras = []
        # frecuencia_palabras_logaritimico = []
        # num_palabras = range(1, len(datos)+1)
        # log_num_palabras = np.log(num_palabras)

        # for clave, valor in datos.items():
        #     frecuencia_palabras.append(valor)
        #     frecuencia_palabras_logaritimico.append(np.log(valor))

        wordCount = 60000
        cont = 0
        wordFreqArray = reversed(sorted(lpal, key=lambda x: x[0 if args.alpha else 1]))
        
        freqArray = [] #Contains the frequency of the words
        logFreqArray = [] #Contains the log of the frequencies
        numArray = range(1,wordCount+1) #Ranges from 1 to wordCount
        logNumArray = np.log(numArray) #Ranges from log(1) to log(wordCount)
    
        for pal, cnt in wordFreqArray:
            palabra = pal.decode('utf-8')
            if re.match('^[a-zA-Z]+$', palabra) and palabra not in stopwords.words('english'):
                # print('%d. %d, %s' % (cont, cnt, pal))
                cont += 1
                freqArray.append(cnt)
                logFreqArray.append(np.log(cnt))
            if cont >= wordCount:
                break
        print('%s Words' % cont)

        print("Metadata obtenido")

        zip(freqArray, logFreqArray, numArray, logFreqArray)

        # zip(frecuencia_palabras, frecuencia_palabras_logaritimico, num_palabras, log_num_palabras)
        
    except NotFoundError:
        print(f'Index {index} does not exists')