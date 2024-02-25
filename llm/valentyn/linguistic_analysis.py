from bs4 import BeautifulSoup
import requests
import re
from gensim.test.utils import lee_corpus_list
from gensim.models import Word2Vec
import csv
import os
import sys
import numpy as np

def parse_file(filename, position):
    csv.field_size_limit(500000)
    script_dir = os.path.dirname(os.path.abspath(__file__)) + '\\data'
    file_path = os.path.join(script_dir, filename)
    text_list = []

    with open(file_path, 'r', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            text_list.append(row[position])

    return text_list

def analyse_n_grams():
    return 0

def analyse_ittf(rows, data_to_analyse):
    words = [doc.split() for doc in rows]
    words_to_analyse = [doc.split() for doc in data_to_analyse]
    vocab = sorted(set(sum(words, [])))
    vocab_dict = {k:i for i,k in enumerate(vocab)}

    # term frequencies:
    X_tf = np.zeros((len(data_to_analyse), len(vocab)), dtype=int)
    for i,doc in enumerate(words):
        for word in doc:
            X_tf[i, vocab_dict[word]] += 1

    # inverse document frequency
    idf = np.log(X_tf.shape[0]/X_tf.astype(bool).sum(axis=0))

    # TFIDF
    X_tfidf = X_tf * idf

    # cosine similiarity:
    X_tfidf_norm = X_tfidf / np.linalg.norm(X_tfidf, axis=1)[:,None]
    M = X_tfidf_norm @ X_tfidf_norm.T
    return M



# model = Word2Vec(lee_corpus_list, vector_size=24, epochs=100)
# word_vectors = model.wv
# print(word_vectors["has"])
# print(model.wv.most_similar("has"))

rows = parse_file("DataSet_Misinfo_FAKE.csv", 1)



