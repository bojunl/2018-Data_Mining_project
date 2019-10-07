
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import os
import time

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


# In[ ]:


def import_data_train():
    train = pd.read_csv('../input/fraud-detection-project/train_entire.csv',skiprows = 1, header=None,names =['ip', 'app', 'device', 'os', 'channel','click_time','attributed_time','is_attributed'])
    time = []
    for i in range(len(train)):
        time.append(int(train.iloc[i,5][-8:-6]))
    time_df = pd.DataFrame(time)
    time_df.columns = ['time']
    train['click_time'] = time_df['time']
    train = train.drop('attributed_time', 1)
    return train

train = import_data_train()
bbb = random.sample(train.index.tolist(), 5000)
trainSet = train.loc[bbb]

trainSet.head()


# In[ ]:


def import_data_test():
    test = pd.read_csv('../input/fraud-detection-project/test_set.csv',skiprows = 1, header=None,names =['ip', 'app', 'device', 'os', 'channel','click_time','attributed_time','is_attributed'])
    time = []
    for i in range(len(test)):
        time.append(int(test.iloc[i,5][-8:-6]))
    time_df = pd.DataFrame(time)
    time_df.columns = ['time']
    test['click_time'] = time_df['time']
    test = test.drop('attributed_time', 1)
    return test

test = import_data_test()
aaa = random.sample(test.index.tolist(), 5000)
testSet = test.loc[aaa]

testSet.head()


# In[ ]:


def crossVal(i,j,k):
    kf = KFold(n_splits=2)
    accuSum = 0
    for train, test in kf.split(trainSet):
        train = trainSet.iloc[train,:]
        test = trainSet.iloc[test,:]
        mlp = MLPClassifier(hidden_layer_sizes=(i,j,k),max_iter=500)
        mlp.fit(train.iloc[:,:-1], train.iloc[:,-1])
        predicted = mlp.predict(test.iloc[:,:-1])
        accuracy = accuracy_score(test.iloc[:,-1], predicted)
        accuSum += accuracy
    return accuSum / 2


# In[ ]:


best_accuracy = 0
accuracy_list = []
best_i, best_j, best_k = 0,0,0

for i in [7,8,9,10,11,12,13]:
    for j in [7,8,9,10,11,12,13]:
        for k in [7,8,9,10,11,12,13]:
            current_accuracy = crossVal(i,j,k)
            accuracy_list.append(current_accuracy)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy;
                best_i, best_j, best_k = i,j,k

print(best_i)
print(best_j)
print(best_k)
print(best_accuracy)


# In[ ]:


X_test = test.drop(columns = ['is_attributed'])
y_test = test[['is_attributed']]


# In[ ]:


mlp = MLPClassifier(hidden_layer_sizes=(9,11,11),max_iter=500)
mlp.fit(train.iloc[:,:-1], train.iloc[:,-1])
predictions = mlp.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))


# In[ ]:


aaa = random.sample(test.index.tolist(), 500000)
trainhalf1 = test.loc[aaa]
testhalf2 = test.drop(aaa)
X_test1 = testhalf2.drop(columns = ['is_attributed'])
y_test1 = testhalf2[['is_attributed']]

mlp = MLPClassifier(hidden_layer_sizes=(11,13,11),max_iter=500)
mlp.fit(trainhalf1.iloc[:,:-1], trainhalf1.iloc[:,-1])
predictions = mlp.predict(X_test1)
print(accuracy_score(y_test1, predictions))
print(confusion_matrix(y_test1,predictions))


# In[ ]:


best_accuracy = 0
accuracy_list = []
j = 11
k = 11
for i in [7,8,9,10,11,12,13]:
    current_accuracy = crossVal(i,j,k)
    accuracy_list.append(current_accuracy)

plt.plot([7,8,9,10,11,12,13], accuracy_list, 'o-')
plt.title("performance with different number of nodes in the first layer")



# In[ ]:


accuracy_list = []
i = 9
k = 11
for j in [7,8,9,10,11,12,13]:
    current_accuracy = crossVal(i,j,k)
    accuracy_list.append(current_accuracy)

plt.plot([7,8,9,10,11,12,13], accuracy_list, 'o-')
plt.title("performance with different number of nodes in the second layer")


# In[ ]:


accuracy_list = []
i = 9
j = 11
for k in [7,8,9,10,11,12,13]:
    current_accuracy = crossVal(i,j,k)
    accuracy_list.append(current_accuracy)

plt.plot([7,8,9,10,11,12,13], accuracy_list, 'o-')
plt.title("performance with different number of nodes in the third layer")
