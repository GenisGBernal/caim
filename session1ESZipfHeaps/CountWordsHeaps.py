from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from elasticsearch.exceptions import NotFoundError, TransportError

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import argparse

def func(n, k, beta):
    return k * (n ** beta)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=None, required=True, help='Index to search')
    parser.add_argument('--alpha', action='store_true', default=False, help='Sort words alphabetically')
    args = parser.parse_args()

    index = args.index

    try:
        client = Elasticsearch(hosts='http://localhost:9200')
        voc = {}
        
        totalCount = 0
        differentCount = 0

        totalWords = []
        differentWords = []
    
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
        print('Heaps Optimal Parameters')
        print('K Optimal Value: %d',k)
        print('Beta Optimal Value: %d', beta)

    except NotFoundError:
        print(f'Index {index} does not exists')
