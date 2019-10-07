
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import gc 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt

# Load subset of the training data
#train = pd.read_csv('../input/data-mining-project/csv_train.csv')

#positive = train.loc[train['is_attributed'] == 1]
#negative = train.loc[train['is_attributed'] == 0]

#negative1 = negative.sample(n=positive.shape[0])



#frames = [positive, negative1]

#result = pd.concat(frames)
    

#negative1.describe()
#result.head() 

#result.to_csv('balanced.csv', index = False)

#print (positive.shape[0])
# Any results you write to the current directory are saved as output.


# In[ ]:


train_sample = pd.read_csv('../input/fraud-detection-project/balanced.csv',skiprows = 1, header=None,names =['ip', 'app', 'device', 'os', 'channel','click_time','attributed_time','is_attributed'])

train_sample.head()


# In[ ]:


train_sample_categorized = train_sample

variables = ['ip', 'app', 'device', 'os', 'channel']
for v in variables:
    train_sample[v] = train_sample[v].astype('category')

#set click_time and attributed_time as timeseries
train_sample_categorized['click_time'] = pd.to_datetime(train_sample['click_time'])
train_sample_categorized['attributed_time'] = pd.to_datetime(train_sample['attributed_time'])
train_sample_categorized['hour'] = train_sample['click_time'].dt.hour 
               
train_sample_categorized.head()             
                                  
                                  


# In[ ]:


ac = pd.DataFrame([        
    train_sample_categorized['ip'], train_sample_categorized['app'], train_sample_categorized['device'], 
    train_sample_categorized['os'], train_sample_categorized['channel'],train_sample_categorized['hour']
])

ac = ac.T
ac.head()
    


# In[ ]:


from sklearn.naive_bayes import GaussianNB   
clf = GaussianNB()
clf = clf.fit(ac, train_sample_categorized ['is_attributed'])
GaussianNB(priors=None, var_smoothing=1e-09)


# In[ ]:


from sklearn.metrics import confusion_matrix
    


# In[ ]:


labels = clf.predict(ac)
mat = confusion_matrix(train_sample_categorized ['is_attributed'], labels)
print(mat) 


# In[ ]:


train = pd.read_csv('../input/fraud-detection-project/train_entire.csv',skiprows = 1, header=None,names =['ip', 'app', 'device', 'os', 'channel','click_time','attributed_time','is_attributed'])
test = pd.read_csv('../input/fraud-detection-project/test_set.csv',skiprows = 1, header=None,names =['ip', 'app', 'device', 'os', 'channel','click_time','attributed_time','is_attributed'])


# In[ ]:



variables = ['ip', 'app', 'device', 'os', 'channel']
for v in variables:
    train[v] = train[v].astype('category')
    test[v]=test[v].astype('category')

#set click_time and attributed_time as timeseries
train['click_time'] = pd.to_datetime(train['click_time'])
train['attributed_time'] = pd.to_datetime(train['attributed_time'])
train['hour'] = train['click_time'].dt.hour#.astype('uint8')  

test['click_time'] = pd.to_datetime(test['click_time'])
test['attributed_time'] = pd.to_datetime(test['attributed_time'])
test['hour'] = test['click_time'].dt.hour#.astype('uint8')  

#set as_attributed in train as a categorical
train['is_attributed']=train['is_attributed'].astype('category')
test['is_attributed']=test['is_attributed'].astype('category')
          


# In[ ]:


acwhole = pd.DataFrame([        
    train['ip'], train['app'], train['device'], 
    train['os'], train['channel'], train['hour']
])

acwhole = acwhole.T
acwhole.head()
    


# In[ ]:


actest = pd.DataFrame([        
    test['ip'], test['app'], test['device'], 
    test['os'], test['channel'], test['hour']
])

actest = actest.T
actest.head()


# In[ ]:


from sklearn.naive_bayes import GaussianNB  
from sklearn.metrics import accuracy_score
clfw = GaussianNB()
clfw = clfw.fit(acwhole, train['is_attributed'])
GaussianNB(priors=None, var_smoothing=1e-09)

labels = clfw.predict(actest)
mattest = confusion_matrix(test['is_attributed'], labels)
accG = accuracy_score(test['is_attributed'], labels, normalize=True, sample_weight=None)
print(mattest) 
print ("accuracy is "+ accG.astype(str))


# In[ ]:


from sklearn.model_selection import KFold, cross_val_score
k_fold = KFold(n_splits=10)
scores = cross_val_score(clfw, actest, labels, cv=k_fold, n_jobs=1)


# In[ ]:


print (scores)


# In[ ]:


from sklearn.naive_bayes import ComplementNB   
from sklearn.metrics import accuracy_score
clfc = ComplementNB()
clfc = clfc.fit(acwhole, train['is_attributed'])
ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)

labelsc = clfc.predict(actest)
mattest = confusion_matrix(test['is_attributed'], labelsc)

accC = accuracy_score(test['is_attributed'], labelsc, normalize=True, sample_weight=None)
print(mattest) 
print ("accuracy is "+ accC.astype(str))


# In[ ]:


trainC = pd.read_csv('../input/fraud-detection-project/csv_train.csv',skiprows = 1, header=None,names =['ip', 'app', 'device', 'os', 'channel','click_time','attributed_time','is_attributed'])


# In[ ]:


variablesC = ['ip', 'app', 'device', 'os', 'channel']
for v in variables:
    trainC[v] = trainC[v].astype('category')

#set click_time and attributed_time as timeseries
trainC['click_time'] = pd.to_datetime(trainC['click_time'])
trainC['attributed_time'] = pd.to_datetime(trainC['attributed_time'])


trainC['hour'] = trainC['click_time'].dt.hour
trainC['is_attributed']=trainC['is_attributed'].astype('category')


# In[ ]:


acC = pd.DataFrame([        
    trainC['ip'], trainC['app'], trainC['device'], 
    trainC['os'], trainC['channel'], trainC['hour']
])

acC = acC.T
acC.head()
    


# In[ ]:


from sklearn.naive_bayes import ComplementNB   
from sklearn.metrics import accuracy_score
clfc = ComplementNB()
clfc = clfc.fit(acC, trainC['is_attributed'])
ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)

labelsc = clfc.predict(actest)
mattest = confusion_matrix(test['is_attributed'], labelsc)

accC = accuracy_score(test['is_attributed'], labelsc, normalize=True, sample_weight=None)
print(mattest) 
print ("accuracy is "+ accC.astype(str))

