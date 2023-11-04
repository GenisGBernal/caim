"""
.. module:: SearchIndexWeight

SearchIndex
*************

:Description: SearchIndexWeight

    Performs a AND query for a list of words (--query) in the documents of an index (--index)
    You can use word^number to change the importance of a word in the match

    --nhits changes the number of documents to retrieve

:Authors: bejar
    

:Version: 

:Created on: 04/07/2017 10:56 

"""

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch.client import CatClient

import argparse
import operator

from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q

import numpy as np

__author__ = 'bejar'

nrounds = 3 # Iteracions
k = 10 # K documents més importants
R = 4 # Nombre de termes nous a guardar de cada nova query
a = 5 # alfa a la regla de Rocchio
B = 4 # beta a la regla de Rocchio, on a > B

def document_term_vector(client, index, id):
    """
    Returns the term vector of a document and its statistics a two sorted list of pairs (word, count)
    The first one is the frequency of the term in the document, the second one is the number of documents
    that contain the term

    :param client:
    :param index:
    :param id:
    :return:
    """
    termvector = client.termvectors(index=index, id=id, fields=['text'],
                                    positions=False, term_statistics=True)

    file_td = {}
    file_df = {}

    if 'text' in termvector['term_vectors']:
        for t in termvector['term_vectors']['text']['terms']:
            file_td[t] = termvector['term_vectors']['text']['terms'][t]['term_freq']
            file_df[t] = termvector['term_vectors']['text']['terms'][t]['doc_freq']
    return sorted(file_td.items()), sorted(file_df.items())

def doc_count(client, index):
    """
    Returns the number of documents in an index

    :param client:
    :param index:
    :return:
    """
    return int(CatClient(client).count(index=[index], format='json')[0]['count'])

def toTFIDF(client, index, file_id):
    """
    Returns the term weights of a document

    :param file:
    :return:
    """

    # Get the frequency of the term in the document, and the number of documents
    # that contain the term
    file_tv, file_df = document_term_vector(client, index, file_id)

    max_freq = max([f for _, f in file_tv])

    dcount = doc_count(client, index)

    tfidfw = []
    for (t, w),(_, df) in zip(file_tv, file_df):
        tf = w/max_freq
        idf = np.log2(dcount/df)
        wt = tf*idf
        tfidfw.append((t, wt))

    tfidfwNormalized = normalize(tfidfw)
    return {term: value for term, value in tfidfwNormalized}

def normalize(tw):
    """
    Normalizes the weights in t so that they form a unit-length vector
    It is assumed that not all weights are 0
    :param tw:
    :return:
    """
    w = [w for _, w in tw]
    magnitude = np.linalg.norm(w)
    tw_normalized = [(t, w/ magnitude) for t, w in tw];

    return tw_normalized

def queryToDic(query):
    queryDic = {}
    for elem in query:
        if '^' in elem:
            key, value = elem.split('^')
            value = float(value)
        elif '~' in elem:
            key, value = elem.split('~')
            value = 1.0/float(value)
        else:
            key = elem
            value = 1.0
        queryDic[key] = value
    queryVector = []
    for term, value in queryDic.items():
        queryVector.append((term,value))
    queryVectorNormalized = normalize(queryVector)
    return {term: value for term, value in queryVectorNormalized}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=None, help='Index to search')
    parser.add_argument('--query', default=None, nargs=argparse.REMAINDER, help='List of words to search')

    args = parser.parse_args()

    index = args.index
    query = args.query

    try:
        client = Elasticsearch(hosts='http://localhost:9200')
        s = Search(using=client, index=index)

        if query is not None:
            for j in range(0, nrounds):
                q = Q('query_string',query=query[0])
                for i in range(1, len(query)):
                    q &= Q('query_string',query=query[i])
                s = s.query(q)
                kDocuments = s[0:k].execute()

                print("Query:")
                print(query)

                queryDic= queryToDic(query)

                sumDocuments = {}

                for document in kDocuments:  # only returns a specific number of results
                    document_tfidf = toTFIDF(client, index, document.meta.id)
                    sumDocuments = {term: sumDocuments.get(term, 0) + document_tfidf.get(term, 0) for term in set(sumDocuments) | set(document_tfidf)} 
                    print(f'ID= {document.meta.id} SCORE={document.meta.score}')
                    print(f'PATH= {document.path}')
                    print(f'TEXT: {document.text[:50]}')
                    print('-----------------------------------------------------------------')

                # sumDocuments = {term: weight*B/k  for term, weight in sumDocuments.items()} # B * (d..dn)/k
                # originalQuery = {term: weight*a for term, weight in queryDic.items()} # a * query

                newQuery = {term: (queryDic.get(term, 0)*a) + (sumDocuments.get(term, 0)*B/k )  for term in set(sumDocuments) | set(queryDic)} # a*Q + B * (d..dn)/k
                newQueryOrdered = sorted(newQuery.items(), key=operator.itemgetter(1), reverse = True)[:R] # Sort terms and get R most important -> [{term, value}, ...]
                newQueryOrderedNormalized = normalize(newQueryOrdered)
                query = []
                for term, value in newQueryOrderedNormalized:
                    query.append(term + '^' + str(value))


        else:
            print('No query parameters passed')

        print (f"{kDocuments.hits.total['value']} Documents")

    except NotFoundError:
        print(f'Index {index} does not exists')

