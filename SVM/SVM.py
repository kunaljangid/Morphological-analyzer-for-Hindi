from sklearn.cluster import KMeans as km
from gensim.models import Word2Vec as w2v
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn import datasets
import numpy as np
import pickle as cp
from sklearn.externals import joblib
file = open('train.txt')
words = []
test =[]
testfile = open('test.txt')
fline = testfile.readline().rstrip('\n')
while fline != '':
    testword = fline.split(' ')[0].rstrip('\t0').rstrip('\t1').rstrip('\t')
    test.append(testword)
    fline = testfile.readline().rstrip('\n')


line = file.readline().rstrip('\n')
while line != '':
    word = line.split(' ')[0].rstrip('\t0').rstrip('\t1').rstrip('\t')
    words.append(word)
    line = file.readline().rstrip('\n')
# print(words)

model = w2v.load('hi.bin')
word_vecs = []
labeltemp=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,]
error = 0
labels = []
i=0

for word in words:
    try:
        word_vecs.append(model.wv[word])
        labels.append(labeltemp[i])
        i+=1
        #print(word + ' successfully parsed!')
    except:
        error+=1
        i+=1
        #print(word + ' is not in vocabulary!')

word_vecs = np.asarray(word_vecs)


classifier = svm.SVC(kernel = 'linear' , C = 1.0,gamma=0.2)
tempp = classifier.fit(word_vecs,labels)
# print(tempp)
# # get the separating hyperplane
# w = classifier.coef_[0]
# a = -w[0] / w[1]
# xx = np.linspace(-5, 5)
# yy = a * xx - (classifier.intercept_[0]) / w[1]
#
# # plot the parallels to the separating hyperplane that pass through the
# # support vectors
# b = classifier.support_vectors_[0]
# yy_down = a * xx + (b[1] - a * b[0])
# b = classifier.support_vectors_[-1]
# yy_up = a * xx + (b[1] - a * b[0])
#
# # plot the line, the points, and the nearest vectors to the plane
# # plt.plot(xx, yy, 'k-')
# # plt.plot(xx, yy_down, 'k--')
# # plt.plot(xx, yy_up, 'k--')

# plt.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1],
            # s=80, facecolors='none')
# plt.scatter(word_vecs[:, 0], word_vecs[:, 1], c=labels, cmap=plt.cm.Paired)
#
# plt.axis('tight')
# plt.show()
# classifier.save(sys.argv[1]+'_weights.hd5')
# predicted= classifier.predict(np.asarray([model.wv['पत्ते']]))
# print(predicted)
#

total=0
for word in test:
    try:
        t=model.wv[word]
        predicted= classifier.predict(np.asarray([model.wv[word]]))
        print(word,predicted)
        # print(predicted)

    except:
        total+=1


# predicted= classifier.predict(np.asarray([model.wv['आँखें']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['छात्र']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['पत्ते']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['चिड़ियाँ']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['आँखें']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['छात्र']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['कहानी']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['बुढ़िया']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['प्रतियाँ']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['बात']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['पुस्तक']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['रुपया']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['भेड़']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['घोड़ा']]))
# print(predicted)
# print("ceck1")
# predicted= classifier.predict(np.asarray([model.wv['पत्ता']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['बेटा']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['लड़का']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['आँख']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['किताब']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['बहन']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['तस्वीर']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['ऋतु']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['बच्चा']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['कपड़ा']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['बात']]))
# print(predicted)
# print("ceck2")
# predicted= classifier.predict(np.asarray([model.wv['लड़के']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['आँखें']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['किताबें']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['बहनें']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['तस्वीरें']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['बच्चे']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['कपड़े']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['बातें']]))
# print(predicted)
# predicted= classifier.predict(np.asarray([model.wv['पुस्तकें']]))
#
#
with open('model.pkl', 'wb') as m:
    joblib.dump(classifier, 'model.pkl')
