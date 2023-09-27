from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from elasticsearch.exceptions import NotFoundError, TransportError

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

import argparse

def func(N, k, beta):
    return k * (N ** beta)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', action='store_true', default=False, help='Sort words alphabetically')
    args = parser.parse_args()

    indexes = ["novels1", "novels2", "novels3", "novels4", "novels5"]

    try:
        client = Elasticsearch(hosts='http://localhost:9200')
        
        totalWords = []
        differentWords = []

        for index in indexes:
            voc = {}
            totalCount = 0
            differentCount = 0
    
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

            for pal, cnt in sorted(lpal, key=lambda x: x[0 if args.alpha else 1]):
                if re.match('^[a-zA-Z]+$', pal.decode('utf-8')) and pal.decode('utf-8') not in set(stopwords.words('english')):
                    totalCount += cnt
                    differentCount += 1

            differentWords.append(differentCount)
            totalWords.append(totalCount)
        
        print(f'Total words: {totalWords}')
        print(f'Different words: {differentWords}')

        # Curve fitting
        popt, pcov = curve_fit(func,totalWords,differentWords)
        k = popt[0]
        beta = popt[1]
        print('------------------------')
        print('Heaps Optimal Parameters')
        print(f'K Optimal Value: {k}')
        print(f'Beta Optimal Value: {beta}')

        # Creating Heaps plot
        HeapsArray = []
        for num in totalWords:
            HeapsArray.append(func(num,*popt))
        
        # Real and Heaps plot
        plt.plot(totalWords, differentWords, label='Valors Reals')
        plt.plot(totalWords, HeapsArray, ls='--', label='Llei de Heaps')
        plt.legend()
        plt.xlabel('Número de Paraules Totals')
        plt.ylabel('Número de Paraules Diferents')
        plt.show()

        # Real and Heaps logaritmic plot
        plt.plot(np.log(totalWords), np.log(differentWords), label='Log dels Valors Reals')
        plt.plot(np.log(totalWords), np.log(HeapsArray), ls='--', label='Log de la Llei de Heaps')
        plt.legend()
        plt.xlabel('Número de Paraules Totals')
        plt.ylabel('Número de Paraules Diferents')
        plt.show()

    except NotFoundError:
        print(f'Index {index} does not exists')
