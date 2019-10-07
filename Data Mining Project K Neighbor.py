import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import time
import datetime

import matplotlib.pyplot as plt

def import_data_train():
    train = pd.read_csv('../input/fraud-detection-project/train_entire.csv')
    time = []
    for i in range(len(train)):
        time.append(int(train.iloc[i,5][-8:-6]))
    time_df = pd.DataFrame(time)
    time_df.columns = ['time']
    train['clicktime'] = time_df['time']
    train = train.drop('attributed_time', 1)
    return train

trainSet = import_data_train()
bbb = random.sample(trainSet.index.tolist(), 10000)
trainSet = trainSet.loc[bbb]
trainSet.head()


def import_data_test():
    test = pd.read_csv('../input/fraud-detection-project/csv_test.csv')
    time = []
    for i in range(len(test)):
        time.append(int(test.iloc[i,5][-8:-6]))
    time_df = pd.DataFrame(time)
    time_df.columns = ['time']
    test['click_time'] = time_df['time']
    test = test.drop('attributed_time', 1)
    return test

testSet = import_data_test()
aaa = random.sample(testSet.index.tolist(), 5000)
testSet = testSet.loc[aaa]
testSet.head()



def crossVal(i):
    kf = KFold(n_splits=10)
    accuSum = 0
    for train, test in kf.split(trainSet):
        training = trainSet.iloc[train,:]
        testing = trainSet.iloc[test,:]
        knn = KNeighborsClassifier(n_neighbors = i, metric='hamming')
#         knn = KNeighborsClassifier(n_neighbors = i, metric=mydist)
        knn.fit(training.iloc[:,:-1].values, training.iloc[:,-1].values)
        predicted = knn.predict(testing.iloc[:,:-1].values)
        accuracy = accuracy_score(testing.iloc[:,-1].values, predicted)
        accuSum += accuracy
    return accuSum / 10.0

 length = range(1,101)

accuList = []
def selectK(i):
    accu = 0
    kk = 0
    for k in i:
        accuracy = crossVal(k)
        accuList.append(accuracy)
        if accuracy > accu:
            accu = accuracy
            kk = k
    return kk

bestK = selectK(length)
print (accuList)
print (bestK)




plt.plot(length, accuList, 'o-')


model = KNeighborsClassifier(n_neighbors = bestK, metric='hamming')
model.fit(trainSet.iloc[:,:-1], trainSet.iloc[:,-1])
predicted = model.predict(testSet.iloc[:,:-1])
accuracy = accuracy_score(testSet.iloc[:,-1], predicted)
print (confusion_matrix(testSet.iloc[:,-1], predicted))
print (accuracy)

