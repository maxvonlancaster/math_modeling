# Bag of words models and TFIDF

from bs4 import BeautifulSoup
import requests
import re

response = requests.get("https://github.com/maxvonlancaster")
root = BeautifulSoup(response.content, "lxml")
txt = root.text
from wordcloud import WordCloud
wc = WordCloud(width=800,height=400).generate(re.sub(r"\s+"," ", root.text))
wc.to_image()
wc.to_file('wc.png')

# ####################################################################################################


# Term frequency

documents = ["the goal of this lecture is to explain the basics of free text processing",
             "the bag of words model is one such approach",
             "text processing via bag of words"]

# documents = ["the goal of this lecture is to explain the basics of free text processing",
#              "the bag of words model is one such approach",
#              "tralalalal text text tralall"]


document_words = [doc.split() for doc in documents]
vocab = sorted(set(sum(document_words, [])))
vocab_dict = {k:i for i,k in enumerate(vocab)}
print(vocab, "\n")
print(vocab_dict, "\n")

# # Now let’s construct a matrix that contains word counts (term frequencies) for all the documents

import numpy as np
X_tf = np.zeros((len(documents), len(vocab)), dtype=int)
for i,doc in enumerate(document_words):
    for word in doc:
        X_tf[i, vocab_dict[word]] += 1
print(X_tf)

# ####################################################################################################


# Inverse document frequency
# idf = log(documents / documents with word)

idf = np.log(X_tf.shape[0]/X_tf.astype(bool).sum(axis=0))
print(idf)

# ####################################################################################################


# # TFIDF

X_tfidf = X_tf * idf
print("TFIDF ", X_tfidf)

# ####################################################################################################


# # Cosine similarity

X_tfidf_norm = X_tfidf / np.linalg.norm(X_tfidf, axis=1)[:,None]
M = X_tfidf_norm @ X_tfidf_norm.T
print(M)


# # Word embeddings and word2vec

# documents = [
#     "pittsburgh has some excellent new restaurants",
#     "boston is a city with great cuisine",
#     "postgresql is a relational database management system"
# ]

# import gensim as gs
# import gensim.downloader as api
# import numpy as np
# model = gs.models.KeyedVectors.load_word2vec_format('lee_background.cor', binary=True)
# # model = api.load('word2vec-google-news-300')
# print(model.wv["pittsburgh"][:10])

from gensim.test.utils import lee_corpus_list
from gensim.models import Word2Vec

model = Word2Vec(lee_corpus_list, vector_size=24, epochs=100)
word_vectors = model.wv
print(word_vectors["has"])

print(model.wv.most_similar("has"))




