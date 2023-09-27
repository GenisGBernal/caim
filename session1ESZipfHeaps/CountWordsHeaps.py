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

def func(n, k, beta):
    return k * (n ** beta)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', action='store_true', default=False, help='Sort words alphabetically')
    args = parser.parse_args()

    indexes = ["novels_1", "novels_2", "novels_3", "novels_4", "novels_5", "novels_6", "novels_7"]

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
                    print(pal.decode('utf-8'))
                    totalCount += cnt
                    differentCount += 1
            differentWords.append(differentCount)
            totalWords.append(totalCount)
        
        print(f'Total words: {totalWords}')
        print(f'Different words: {differentWords}')

        totalWords = [10944, 17967, 24764, 36784, 66831, 95244]
        differentWords = [6213, 10944, 17967, 20738, 24764, 36784]

        # Curve fitting
        popt, pcov = curve_fit(func,totalWords,differentWords)
        k = popt[0]
        beta = popt[1]
        print('Heaps Optimal Parameters')
        print('K Optimal Value: %d', k)
        print('Beta Optimal Value: %d', beta)

        # Creating Heaps plot
        fitArray = []
        logFitArray = []
        for num in totalWords:
            fitArray.append(func(num,*popt))
            logFitArray.append(np.log(func(num,*popt)))
        
        # Real and Heaps plot
        plt.plot(totalWords, differentWords, 'b-', label='Real values')
        plt.plot(totalWords, fitArray,'r-', ls='--', label='Heap\'s law')
        plt.legend()
        plt.xlabel('x = Number of total words')
        plt.ylabel('y = Number of different words')
        plt.show()

        plt.plot(np.log(totalWords), np.log(differentWords), 'b-', label='Log of real values')
        plt.plot(np.log(totalWords), logFitArray,'r-',ls='--', label='Log of Heap\'s law')
        plt.legend()
        plt.xlabel('x = Number of total words')
        plt.ylabel('y = Number of different words')
        plt.show()

    except NotFoundError:
        print(f'Index {index} does not exists')
