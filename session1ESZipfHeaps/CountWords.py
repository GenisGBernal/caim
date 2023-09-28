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
    return np.abs(c)/((x+np.abs(b))**a)

def zip(frecuencia_palabras, frecuencia_palabras_logaritimico, log):
    x = range(1,len(frecuencia_palabras)+1)
    posiciones = []
    posiciones_log = []
    for pos in x:
        posiciones.append(pos)
        posiciones_log.append(np.log(pos))
    
    popt, pcov = curve_fit(func, posiciones, frecuencia_palabras)
    print('b = %d, c = %d' % (popt[0],popt[1]))
    frecuencia_ideal = []
    frecuencia_ideal_log = []
    for pos in posiciones:
        frecuencia_ideal.append(func(pos, *popt))
        frecuencia_ideal_log.append(np.log(func(pos, *popt)))

    print("Calculo zips hecho")
    if (log):
        zip_log(posiciones_log, frecuencia_palabras_logaritimico, frecuencia_ideal_log)
    else:
        zip_lineal(posiciones, frecuencia_palabras, frecuencia_ideal)

def zip_lineal(posiciones, frecuencia_palabras, frecuencia_ideal):
    plt.plot(posiciones, frecuencia_palabras, 'b-', label='Freqüència de les paraules')
    plt.plot(posiciones, frecuencia_ideal,'r-',label='Zipf ideal')
    plt.legend()
    plt.xlabel('x = Paraules ordenades de major a menor freqüència')
    plt.ylabel('y = Freqüència de les paraules')
    plt.show()

def zip_log(posiciones,logFreqArray,logFitArray):
    plt.plot(posiciones, logFreqArray, 'b-', label='Freqüència de les paraules logarítmic')
    plt.plot(posiciones, logFitArray,'r-',label='Zipf logarítmic')
    plt.legend()
    plt.xlabel('x = Paraules ordenades de major a menor freqüència logarítmic')
    plt.ylabel('y = Freqüència de les paraules logarítmic')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default='novels', help='Index to search')
    parser.add_argument('--alpha', action='store_true', default=False, help='Sort words alphabetically')
    parser.add_argument('--log', default=False, help='User log plot')
    args = parser.parse_args()

    index = args.index
    log = args.log
    log = log == "True"

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

        wordFreqArray = reversed(sorted(lpal, key=lambda x: x[0 if args.alpha else 1]))
        
        frecuencia_palabras = []
        frecuencia_palabras_log = [] 

        count = 0

        for pal, cnt in wordFreqArray:
            palabra = pal.decode('utf-8')
            if re.match('^[a-zA-Z]+$', palabra) and palabra not in stopwords.words('english'):
                frecuencia_palabras.append(cnt)
                frecuencia_palabras_log.append(np.log(cnt))
                count = count + 1
            # if (count >= 1000):
            #     break

        print('%s Words' % len(frecuencia_palabras))

        print("Metadata obtenido")

        zip(frecuencia_palabras, frecuencia_palabras_log, log)
        
    except NotFoundError:
        print(f'Index {index} does not exists')