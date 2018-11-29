import numpy as np
import itertools
from tqdm import tqdm
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.util import ngrams  
import csv

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

def write_csv(sent_list, label_list, out_path):
    filewriter = csv.writer(open(out_path, "w+"))
    count = 0
    for ((id, sent), label) in zip(sent_list, label_list):
        filewriter.writerow([id, sent, label])

def read_csv(data_path,corpus):
    file_reader = csv.reader(open(data_path,"rt", errors="ignore",encoding="utf-8"), delimiter=',')
    sent_list = []

    for row in file_reader:
        id = row[0]
        sent = row[1]
        label = row[2]
        if corpus=='train':
            sent_list.append((id,sent,label))
        else:
            sent_list.append((id,sent))
    return sent_list
    
min_len = 500
max_len = 10000

o = read_csv('./Subtask-A/Full_V1.3_Training.csv','train')#pandas.read_csv('./Subtask-A/Full_V1.3_Training.csv', names=['id','x','y'])

X = [x[1] for x in o]
Y = [int(y[2]) for y in o]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(x_train)
x = tfidf_vect.transform(x_train)

model = SVC(C=1,probability=True)
model.fit(x, y_train)

pred = model.predict(tfidf_vect.transform(x_test))

from sklearn.metrics import f1_score 
print(f1_score(y_test, pred))

test_path='./Subtask-A/TrialData_SubtaskA_Test.csv'
test_path_output=test_path[:-4] + "_predictions.csv"
oo = read_csv(test_path,'test')

XZ = [x[1] for x in o]

trial_predictions=model.predict(tfidf_vect.transform(XZ))
write_csv(oo,trial_predictions,test_path_output)
