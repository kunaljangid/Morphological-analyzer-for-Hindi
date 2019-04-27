from sklearn.cluster import KMeans as km
from gensim.models import Word2Vec as w2v
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
import numpy as np
import pickle as cp
file = open('sing_plu.txt')
datafile = open('datakmalgo.txt')
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
data =[]
word_vecs = np.asarray(word_vecs)
#print(word_vecs.shape)
dataline = datafile.readline().rstrip('\n')
while dataline != '':
    tem = dataline.split(' ')[0].rstrip('\t0').rstrip('\t1').rstrip('\t')
    data.append(tem)
    dataline = datafile.readline().rstrip('\n')

km_model = km(n_clusters = 2, init='k-means++', max_iter = 100, n_init = 1)
km_model.fit(word_vecs)
total=0
for word in data:
    try:
        t = model.wv[word]
        predicted= km_model.predict(np.asarray([model.wv[word]]))
        print(word,predicted)
        # print(predicted)

    except:
        total+=1




# labels = km_model.labels_
# metrics.silhouette_score(word_vecs, labels, metric='manhattan')
# pred = km_model.predict(np.asarray([model.wv['पत्ते']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['आँखें']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['छात्र']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['पत्ते']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['चिड़ियाँ']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['आँखें']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['छात्र']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['कहानी']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['बुढ़िया']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['प्रतियाँ']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['बात']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['पुस्तक']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['रुपया']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['भेड़']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['घोड़ा']]))
# print(pred)
# print("ceck1")
# pred = km_model.predict(np.asarray([model.wv['पत्ता']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['बेटा']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['लड़का']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['आँख']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['किताब']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['बहन']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['तस्वीर']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['ऋतु']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['बच्चा']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['कपड़ा']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['बात']]))
# print(pred)
# print("ceck2")
# pred = km_model.predict(np.asarray([model.wv['लड़के']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['आँखें']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['किताबें']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['बहनें']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['तस्वीरें']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['बच्चे']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['कपड़े']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['बातें']]))
# print(pred)
# pred = km_model.predict(np.asarray([model.wv['पुस्तकें']]))
#

with open('model.pkl', 'wb') as m:
    cp.dump(km_model, m)
