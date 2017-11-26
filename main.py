# Credit Card Fraud Detection
# Dataset Link: https://www.kaggle.com/dalpozz/creditcardfraud
# Dataset Size: 68 MB

import os, sys
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

# Fix seed for reproducibility 
seed = 24
np.random.seed(seed)
print("*MESSAGE* Random seed: {}".format(seed))

# Load data from csv into pandas dataframe
def load_data(csv_data,ratio=0.1):
    # Read the csv nd extract all the required info
    dataframe = pd.read_csv(csv_data)
    num_pos = dataframe[dataframe['Class']==1].shape[0]
    num_neg = dataframe.shape[0]-num_pos

    # Print the statics of Data
    print("Data statitics:")
    print(". # of samples        : {}".format(dataframe.shape[0]))
    print(". # of features       : {}".format(dataframe.shape[1]-1))
    print(". # of +ve/-ve samples: {}/{}".format(num_pos,num_neg))

    message = ("*MESSAGE* As you can see, out dataset is highly skewed: "
               "only {0:.3f}% of data is +ve".format(num_pos*100/(num_pos+num_neg)))
    print(message)

    return dataframe

"""
Data set has a column 'Amount', which is not normalised
So, we normalise the 'Amount' column and drop the original 'Amount' Column
Further, since the data is highly skewed, we undersample it 
"""
def preprocess_FollowUnderSampling(data):
    data['NormalisedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    data = data.drop(['Time','Amount'],axis=1)

    # Undersampling Steps
    fraud_count = len(data[data.Class==1])
    fraud_indices = data[data.Class==1].index
    notFraud_indices = data[data.Class==0].index
    rand_notFraud_indices = notFraud_indices[np.random.permutation(len(notFraud_indices))]
    rand_notFraud_indices_undersampled = rand_notFraud_indices[0:3*fraud_count]
    undersampled_indices = np.concatenate([fraud_indices,rand_notFraud_indices_undersampled])
    undersampled_data = data.iloc[undersampled_indices,:]
    X_undersampled = undersampled_data.iloc[:,undersampled_data.columns!='Class']
    Y_undersampled = undersampled_data.iloc[:,undersampled_data.columns=='Class']
    return X_undersampled,Y_undersampled

"""
X,Y are pandas dataframes
"""
def get_tr_tst(X,Y,ratio=0.10):
    X = X.as_matrix()
    Y = Y.as_matrix()
    indices = np.random.permutation(X.shape[0])
    split_index = int(ratio*indices.size)
    tst_data = X[indices[0:split_index],:]
    tst_lb = Y[indices[0:split_index]]
    tst_lb.shape = (tst_lb.size,)
    tr_data = X[indices[split_index:],:]
    tr_lb = Y[indices[split_index:]]
    tr_lb.shape = (tr_lb.size,)
    print("Statitics:")
    print(". # of train/test samples: {}/{}".format(tr_lb.size,tst_lb.size))
    return tr_data,tr_lb,tst_data,tst_lb

def svm(tr_data,tr_lb,tst_data,tst_lb,kernel='rbf'):
    mySVM = SVC(kernel='poly',class_weight='balanced')
    C_range = [0.01,0.1,1,10,100,1000,10000]
    param_grid = dict(C=C_range)
    #param_grid
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.1)
    clf = GridSearchCV(mySVM, param_grid=param_grid, cv=cv, n_jobs=-1)
    clf.fit(tr_data,tr_lb)
    message = ("*MESSAGE* Best C: {}, Score: {}".format(clf.best_params_['C'],clf.best_score_))
    print(message)
    pred = clf.predict(tst_data)
    count = 0
    for i in range(pred.size):
        if pred[i]==tst_lb[i]: count += 1
    print("*MESSAGE* Accuracy: {}".format(count/tst_lb.size))
    precision,recall,fscore = precision_recall_fscore_support(tst_lb,pred,average='binary')[0:3]
    print("*MESSAGE* Precision: {}, Recall: {}, F-score: {}".format(precision,recall,fscore))
    #return pred

def main():
    csv_data = "creditcard.csv"
    data = load_data(csv_data,0.1)
    X,Y = preprocess_FollowUnderSampling(data)
    tr_data,tr_lb,tst_data,tst_lb = get_tr_tst(X,Y,0.1)
    svm(tr_data,tr_lb,tst_data,tst_lb,'poly')


if __name__=="__main__":
    main()

