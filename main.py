# Credit Card Fraud Detection
# Dataset Link: https://www.kaggle.com/dalpozz/creditcardfraud
# Dataset Size: 68 MB

import os, sys
import pandas as pd
import numpy as np

def load_data(csv_data):
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

# data is pandas dataframe
# Normal split, no stratified split
# Bound to fail due to skewness of data
def get_tr_tst(data,ratio):
    temp = data.as_matrix()
    indices = np.random.permutation(data.shape[0])
    split_index = int(ratio*indices.size)
    tst_data = temp[indices[0:split_index],0:-1]
    tst_lb = temp[indices[0:split_index],-1]
    tr_data = temp[indices[split_index:],0:-1]
    tr_lb = temp[indices[split_index:],-1]
    print("Statitics:")
    print(". # of train/test samples: {}/{}".format(tr_lb.size,tst_lb.size))
    return tst_data,tst_lb,tr_data,tr_lb

def rbf_svm(tr_feat,tr_lb,tst_feat,tst_lb):
    mySVM = SVC(kernel='rbf')
    C_range = np.logspace(1,4,4)
    Gamma_range = np.logspace(1,4,4)
    param_grid = dict(gamma=Gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=32)
    clf = GridSearchCV(mySVM, param_grid=param_grid, cv=cv, n_jobs=-1)
    clf.fit(tr_feat,tr_lb)

def main():
    csv_data = "creditcard.csv"
    data = load_data(csv_data,0.1)
    tst_feat,tst_lb,tr_feat,tr_lb = get_tr_tst(data,0.1)
    rbf_svm(tst_feat,tst_lb,tr_feat,tr_lb)


if __name__=="__main__":
    main()

