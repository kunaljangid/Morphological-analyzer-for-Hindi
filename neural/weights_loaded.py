from gensim.models import Word2Vec as w2v
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation, Dropout
import numpy as np
import pickle as cp
import sys
import keras.models as km
model = w2v.load('hi.bin')
testfile = open('test.txt')
tline = testfile.readline().rstrip('\n')
test_temp=[]
test_vecs = []
vectorset = []
while tline != '':
    tword = tline.split(' ')
    # print(tword)
    test_temp.append(tword)
    tline = testfile.readline().rstrip('\n')
n=0
testtotal=0
for tword in test_temp:
     try:
        # print('0')
        # print(tword[0])
        # print(tword[1])
        test_vecs.append(model.wv[tword[0]])
        test_vecs.append(model.wv[tword[1]])
        # print('1')
        testtotal = testtotal+1
        vectorset.append(tword[0])
        vectorset.append(tword[1])
        # print("2")
     except:
         n = n+1
print(testtotal)
# print(vectorset)
test_vecs = np.asarray(test_vecs)
#print(test_vecs)
testtempor = []
for i in range(0,testtotal):
    if i%2==0:
        c = np.array(test_vecs[i])
        d = np.array(test_vecs[i+1])
        testtempor.append(np.concatenate((c,d), axis = 0))
#print(testtempor)
test_vecs = np.asarray(testtempor)
size = len(testtempor)
modelt = km.load_model('5_weights.hd5')
for i in range(size):
    x = modelt.predict(np.asarray([test_vecs[i]]))
    if x>0.7:
        z='YES'
    else :
        z='NO'
    print(i , vectorset[2*i] , vectorset[2*i+1] , z)
    # print(vectorset[2*i-1])
    # print(vectorset[2*i])
    # print()
