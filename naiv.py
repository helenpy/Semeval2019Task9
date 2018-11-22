import numpy as np
import itertools
from tqdm import tqdm
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.util import ngrams  


def word_ngram(corpus, n):
    corpus_grams = [list(ngrams(word_tokenize(text), n)) for text in corpus]
    index_ref = list({gram for text in corpus_grams for gram in text})
    out = np.zeros((len(corpus_grams), len(index_ref)))
    for i, el in enumerate(corpus_grams):
        for gram in el:
            out[i][index_ref.index(gram)] += 1

    import pdb
    pdb.set_trace()

    return out, index_ref



min_len = 500
max_len = 10000

o = pandas.read_csv('./Subtask-A/Training_Full_V1.2.csv', names=['id','x','y'])

X = o['x']
Y = o['y']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(x_train)
x = tfidf_vect.transform(x_train)

model = SVC(C=10000)
model.fit(x, y_train)

pred = model.predict(tfidf_vect.transform(x_test))

from sklearn.metrics import accuracy_score 
print(accuracy_score(y_test, pred))
