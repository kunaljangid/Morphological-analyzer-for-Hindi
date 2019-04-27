from sklearn.cluster import KMeans as km
from gensim.models import Word2Vec as w2v
import numpy as np
import pickle as cp
file = open('sing_plu.txt')
words = []
line = file.readline().rstrip('\n')
while line != '':
    word = line.split(' ')[0].rstrip('\t0').rstrip('\t1').rstrip('\t')
    words.append(word)
    line = file.readline().rstrip('\n')
# print(words)
model = w2v.load('hi.bin')
word_vecs = []
error = 0
for word in words:
    try:
        word_vecs.append(model.wv[word])
        #print(word + ' successfully parsed!')
    except:
        error+=1
        #print(word + ' is not in vocabulary!')

word_vecs = np.asarray(word_vecs)
#print(word_vecs.shape)

km_model = km(n_clusters = 2, init='k-means++', max_iter = 100, n_init = 1)
km_model.fit(word_vecs)



# pred = km_model.predict(np.asarray([model.wv['पत्ते']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['आँखें']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['छात्र']]))
# print(pred)



with open('model.pkl', 'wb') as m:
    cp.dump(km_model, m)
