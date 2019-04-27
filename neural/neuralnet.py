from gensim.models import Word2Vec as w2v
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation, Dropout
import numpy as np
import pickle as cp
import sys
model = w2v.load('hi.bin')
word_vecs = []
labels = []
label_temp = []
test_temp=[]
test_vecs = []
# training data set reading
file = open('train.txt')
words = []

line = file.readline()
while line != '':
    word = line.split()
    words.append(word)
    line = file.readline()

# testing data loading
testfile = open('test.txt')
tline = testfile.readline()
while tline != '':
    tword = tline.split()
    test_temp.append(tword)
    tline = testfile.readline()
n=0
testtotal=0
for tword in test_temp:
     try:
        test_vecs.append(model.wv[tword[0]])
        test_vecs.append(model.wv[tword[1]])
        testtotal = testtotal+1
     except:
         n = n+1
test_vecs = np.asarray(test_vecs)
testtempor = []
for i in range(0,testtotal):
    if i%2==0:
        c = np.array(test_vecs[i])
        d = np.array(test_vecs[i+1])
        testtempor.append(np.concatenate((c,d), axis = 0))
test_vecs = np.asarray(testtempor)
#print(len(test_vecs))
####################################################################################
#label dataset reading
file_2 = open('label.txt')
line_2 = file_2.readline().rstrip('\n')
while line_2 != '':
    lab = line_2.split()
    label_temp.append([int(lab[0])])
    line_2 = file_2.readline().rstrip('\n')
#print(label_temp)
yes = 0
no = 0
count = 0
total = 0
for word in words:
    try:
    #    print(word)
        word_vecs.append(model.wv[word[0]])
        word_vecs.append(model.wv[word[1]])
        labels.append(label_temp[count])
        yes = yes+1
        count = count+1
        total = total+1
    except:
        no = no+1
        count = count+1
        #total = total+1
#print(len(labels), len(words))
word_vecs = np.asarray(word_vecs)
tempor = []
count = 0
for i in range(0,total):
    a = word_vecs[i]
    b= word_vecs[i+1]
    tempor.append(np.concatenate((a,b), axis = 0))
    i = i+1
    count = count+1
    #print(count)
word_vecs = np.asarray(tempor)
#print(labels)
########################################################################
#print(word_vecs.shape)
#print(np.asarray([labels]).shape)
modelt = Sequential()
modelt.add(Dense(256))
modelt.add(Activation('relu'))
modelt.add(Dropout(0.6))
modelt.add(Dense(128))
modelt.add(Activation('relu'))
modelt.add(Dropout(0.6))
modelt.add(Dense(32))
modelt.add(Activation('relu'))
modelt.add(Dense(4))
modelt.add(Activation('sigmoid'))
modelt.add(Dense(1))
modelt.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
modelt.fit(word_vecs,np.asarray(labels), epochs=20, batch_size = 4, shuffle=True, validation_split = 0.2)
modelt.save(sys.argv[1]+'_weights.hd5')
for i in range(0,50):
    print(modelt.predict(np.asarray([test_vecs[i]])))
