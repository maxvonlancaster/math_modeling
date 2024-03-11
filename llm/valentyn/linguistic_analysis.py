from bs4 import BeautifulSoup
from gensim.test.utils import lee_corpus_list
from gensim.models import Word2Vec
import numpy as np
import math
from numpy import dot
from numpy.linalg import norm

def analyse_n_grams():
    return 0

def analyse_tfidf(rows, data_to_analyse):
    words = [doc.lower().split() for doc in rows]
    # words_to_analyse = [doc.lower().split() for doc in data_to_analyse]
    vocab = sorted(set(sum(words, [])))
    vocab_dict = {k:i for i,k in enumerate(vocab)}

    # term frequencies:
    X_tf = np.zeros((len(data_to_analyse), len(vocab)), dtype=int)
    for i,elem in enumerate(data_to_analyse):
        for word in elem.split():
            if word in vocab_dict.keys():
                X_tf[i, vocab_dict[word]] += 1

    idf = np.zeros((len(vocab)), dtype=float)
    for i in range(len(idf)):
        count = 0
        for j in range(len(X_tf)):
            if X_tf[j, i] > 0:
                count += 1
        if count == 0:
            idf[i] = 1
        else:
            idf[i] = math.log(len(X_tf)/ count)
    
    # TFIDF
    X_tfidf = X_tf * idf

    norms = np.zeros((len(data_to_analyse)), dtype=float)
    for i in range(len(norms)):
        norm = np.linalg.norm(X_tfidf[i])
        norms[i] = norm
    
    return norms



def analyse_tfidf_cosine(rows, data_to_analyse):
    words = [doc.lower().split() for doc in rows]
    # words_to_analyse = [doc.lower().split() for doc in data_to_analyse]
    vocab = sorted(set(sum(words, [])))
    vocab_dict = {k:i for i,k in enumerate(vocab)}

    # term frequencies:
    X_tf = np.zeros((len(data_to_analyse), len(vocab)), dtype=int)
    for i,elem in enumerate(data_to_analyse):
        for word in elem.split():
            if word in vocab_dict.keys():
                X_tf[i, vocab_dict[word]] += 1

    idf = np.zeros((len(vocab)), dtype=float)
    for i in range(len(idf)):
        count = 0
        for j in range(len(X_tf)):
            if X_tf[j, i] > 0:
                count += 1
        if count == 0:
            idf[i] = 1
        else:
            idf[i] = math.log(len(X_tf)/ count)
    
    # TFIDF
    X_tfidf = X_tf * idf

    vector_one = np.ones((len(vocab)), dtype=float)

    cosinuses = np.zeros((len(data_to_analyse)), dtype=float)

    for i in range(len(data_to_analyse)):
        cosinuses[i] = dot(X_tfidf[i], vector_one) / (norm(X_tfidf[i]) * norm(vector_one))
    
    return cosinuses