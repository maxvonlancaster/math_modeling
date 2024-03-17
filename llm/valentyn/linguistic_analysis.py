from bs4 import BeautifulSoup
from gensim.test.utils import lee_corpus_list
from gensim.models import Word2Vec
import numpy as np
import math
from numpy import dot
from numpy.linalg import norm

common_words = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with",
    "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if",
    "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just",
    "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back",
    "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us"
]

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



def analyse_tfidf_cosine(rows, data_to_analyse, remove_common):
    words = [doc.lower().split() for doc in rows]
    # words_to_analyse = [doc.lower().split() for doc in data_to_analyse]
    vocab = set(sum(words, []))
    if(remove_common):
        vocab = vocab - set(common_words)
    vocab_list = sorted(vocab)
    vocab_dict = {k:i for i,k in enumerate(vocab)}

    # term frequencies:
    X_tf = np.zeros((len(data_to_analyse), len(vocab_list)), dtype=int)
    for i,elem in enumerate(data_to_analyse):
        for word in elem.lower().split():
            if word in vocab_dict.keys():
                X_tf[i, vocab_dict[word]] += 1

    idf = np.zeros((len(vocab_list)), dtype=float)
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

    vector_one = np.ones((len(vocab_list)), dtype=float)

    cosinuses = np.zeros((len(data_to_analyse)), dtype=float)

    for i in range(len(data_to_analyse)):
        cosinuses[i] = dot(X_tfidf[i], vector_one) / (norm(X_tfidf[i]) * norm(vector_one))
    
    return cosinuses