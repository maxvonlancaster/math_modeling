from bs4 import BeautifulSoup
from gensim.test.utils import lee_corpus_list
from gensim.models import Word2Vec
import numpy as np
import math

def analyse_n_grams():
    return 0

def analyse_tfidf(rows, data_to_analyse):
    words = [doc.lower().split() for doc in rows]
    words_to_analyse = [doc.lower().split() for doc in data_to_analyse]
    vocab = sorted(set(sum(words, [])))
    vocab_dict = {k:i for i,k in enumerate(vocab)}

    # term frequencies:
    X_tf = np.zeros((len(data_to_analyse), len(vocab)), dtype=int)
    for i,elem in enumerate(data_to_analyse):
        for word in elem.split():
            if word in vocab_dict.keys():
                X_tf[i, vocab_dict[word]] += 1
    
    # for i,doc in enumerate(words):
    #     for word in doc:
    #         X_tf[i, vocab_dict[word]] += 1

    # inverse document frequency
    # idf = np.log(X_tf.shape[0]/X_tf.astype(bool).sum(axis=0))

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

    # cosine similiarity:
    # X_tfidf_norm = X_tfidf / np.linalg.norm(X_tfidf, axis=1)[:,None]
    # M = X_tfidf_norm @ X_tfidf_norm.T
    # return M

    norms = np.zeros((len(data_to_analyse)), dtype=float)
    for i in range(len(norms)):
        norm = np.linalg.norm(X_tfidf[i])
        norms[i] = norm
    
    return norms
