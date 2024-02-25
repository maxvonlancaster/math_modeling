from bs4 import BeautifulSoup
import requests
import re
from gensim.test.utils import lee_corpus_list
from gensim.models import Word2Vec
import csv
import os
import sys
import numpy as np
import math
import time

def parse_file(filename, position, take):
    csv.field_size_limit(500000)
    script_dir = os.path.dirname(os.path.abspath(__file__)) + '\\data'
    file_path = os.path.join(script_dir, filename)
    text_list = []

    with open(file_path, 'r', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file)
        i = 0
        for row in csv_reader:
            if i < take:
                text_list.append(row[position])
                i += 1
            else: 
                break

    return text_list

def analyse_n_grams():
    return 0

def analyse_ittf(rows, data_to_analyse):
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
    
    # print(X_tf)
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
    
    # print(idf)

    # TFIDF
    X_tfidf = X_tf * idf

    # print(X_tfidf)

    # cosine similiarity:
    # X_tfidf_norm = X_tfidf / np.linalg.norm(X_tfidf, axis=1)[:,None]
    # M = X_tfidf_norm @ X_tfidf_norm.T
    # return M

    norms = np.zeros((len(data_to_analyse)), dtype=float)
    for i in range(len(norms)):
        norm = np.linalg.norm(X_tfidf[i])
        norms[i] = norm
    
    return norms



# model = Word2Vec(lee_corpus_list, vector_size=24, epochs=100)
# word_vectors = model.wv
# print(word_vectors["has"])
# print(model.wv.most_similar("has"))

start_time = time.time()

rows = parse_file("DataSet_Misinfo_FAKE.csv", 1, 1005)
# rows = ["minus plus", "cosine sin", "one two three"]

data_to_analyse = ["hello world", "hello Valentyn", "trump vaccine clinton public plus minus one ctng pew"]
result = analyse_ittf(rows, data_to_analyse)

end_time = time.time()
execution_time = end_time - start_time

print(result)
print(f"Час виконання: {execution_time} секунд")
