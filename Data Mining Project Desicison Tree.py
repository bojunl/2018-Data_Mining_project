import numpy as np 
import pandas as pd 
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def import_data_DT():
    train = pd.read_csv('../input/fraud-detection-project/train_entire.csv')
    time = []
    for i in range(len(train)):
        time.append(int(train.iloc[i,5][-8:-6]))
    time_df = pd.DataFrame(time)
    time_df.columns = ['time']
    train['clicktime'] = time_df['time']
    train = train.drop('attributed_time', 1)
    return train
    
def training_DT(depth, leaf, X_train, Y_train, clf):
    classifier = DecisionTreeClassifier(criterion = clf, max_depth = depth, min_samples_leaf = leaf)
    classifier.fit(X_train, Y_train)
    return classifier

def accuracy_DT(classifier, X_test, Y_test):
    Y_pred = classifier.predict(X_test)
    M = confusion_matrix(Y_test, Y_pred)
    A = accuracy_score(Y_test,Y_pred)
    return M, A
    
def cross_validation_DT(X, Y, depth, leaf, clf):
    kf = KFold(n_splits = 4)
    accu = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        classifier = training_DT(depth, leaf, X_train, Y_train, clf)
        Mat, Acc = accuracy_DT(classifier, X_test, Y_test)
        accu.append(Acc)
    return np.mean(accu)

def visualization_DT(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_wireframe(X, Y, Z, rstride = 1, cstride = 1)
    ax.set_xlabel('Max Tree Depth')
    ax.set_ylabel('Min to Form Leaf')
    ax.set_zlabel('Accuracy')
    plt.show()

# testing the classification accuracy for different choices of tree depth and least number of elements in a leaf
def main_DT():
    dataset_DT = import_data_DT()
    X = dataset_DT.values[:,0:6]
    Y = dataset_DT.values[:,6]
    accuracy_matrix_gini = []
    accuracy_matrix_entropy = []
    depth_choice = []
    leaf_choice = []
    for i in range(1, 101):
        depth_choice.append(i)
        leaf_choice.append(i)
    for i in depth_choice:
        temp1 = []
        temp2 = []
        for j in leaf_choice:
            temp1.append(cross_validation_DT(X, Y, i, j, "gini"))
            temp2.append(cross_validation_DT(X, Y, i, j, "entropy"))
        accuracy_matrix_gini.append(temp1)
        accuracy_matrix_entropy.append(temp2)
        print(i)
    x_axis = []
    y_axis = []
    for i in range(len(depth_choice)):
        temp = []
        for j in range(len(depth_choice)):
            temp.append(depth_choice[i]) 
        x_axis.append(temp)
        y_axis.append(leaf_choice)
    return np.array(x_axis), np.array(y_axis), np.array(accuracy_matrix_gini), np.array(accuracy_matrix_entropy)



# visualization of choice of depth and min to form leaf versus corresponding classification accuracy
X_axis, Y_axis, gini_matrix, entropy_matrix = main_DT()
visualization_DT(X_axis, Y_axis, gini_matrix)

visualization_DT(X_axis, Y_axis, entropy_matrix)











import numpy as np 
import pandas as pd 
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def import_data_DT():
    train = pd.read_csv('../input/fraud-detection-project/train_entire.csv')
    time = []
    for i in range(len(train)):
        time.append(int(train.iloc[i,5][-8:-6]))
    time_df = pd.DataFrame(time)
    time_df.columns = ['time']
    train['clicktime'] = time_df['time']
    train = train.drop('attributed_time', 1)
    return train
    
def import_test_set_DT():
    test = pd.read_csv('../input/fraud-detection-project/test_set.csv')
    time = []
    for i in range(len(test)):
        time.append(int(test.iloc[i, 5][-8: -6]))
    time_df = pd.DataFrame(time)
    time_df.columns = ['time']
    test['clicktime'] = time_df['time']
    test = test.drop('attributed_time', 1)
    return test
    
def training_DT(depth, leaf, X_train, Y_train, clf):
    classifier = DecisionTreeClassifier(criterion = clf, max_depth = depth, min_samples_leaf = leaf)
    classifier.fit(X_train, Y_train)
    return classifier

def accuracy_DT(classifier, X_test, Y_test):
    Y_pred = classifier.predict(X_test)
    M = confusion_matrix(Y_test, Y_pred)
    A = accuracy_score(Y_test,Y_pred)
    return M, A



def main_DT():
    dataset_DT = import_data_DT()
    testset_DT = import_test_set_DT()
    X = dataset_DT.values[:,0:6]
    Y = dataset_DT.values[:,6]
    X_t = testset_DT.values[:, 0:6]
    Y_t = testset_DT.values[:, 6]
    classifier = training_DT(21, 27, X, Y, "entropy")
    M, A = accuracy_DT(classifier, X_t, Y_t)
    print(A)
    print(M)
    


main_DT()