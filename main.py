# Credit Card Fraud Detection
# Dataset Link: https://www.kaggle.com/dalpozz/creditcardfraud
# Dataset Size: 68 MB

import os, sys
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import itertools
from scipy import interp
import pylab
import matplotlib.pyplot as plt

# Fix seed for reproducibility 
seed = 24
np.random.seed(seed)
print("*MESSAGE* Random seed: {}".format(seed))

# Load data from csv into pandas dataframe
# Data set has a column 'Amount', which is not normalised
# So, we normalise the 'Amount' column and drop the original 'Amount' Column
def load_data(csv_data):
    # Read the csv nd extract all the required info
    dataframe = pd.read_csv(csv_data)
    dataframe['NormalisedAmount'] = StandardScaler().fit_transform(dataframe['Amount'].values.reshape(-1, 1))
    dataframe = dataframe.drop(['Time','Amount'],axis=1)
    num_pos = dataframe[dataframe['Class']==1].shape[0]
    num_neg = dataframe.shape[0]-num_pos

    # Print the statics of Data
    print("Data statitics:")
    print(". # of samples        : {}".format(dataframe.shape[0]))
    print(". # of columns        : {}".format(dataframe.shape[1]-1))
    print(". # of +ve/-ve samples: {}/{}".format(num_pos,num_neg))

    message = ("*MESSAGE* As you can see, out dataset is highly skewed: "
               "only {0:.3f}% of data is +ve".format(num_pos*100/(num_pos+num_neg)))
    print(message)

    return dataframe

# Since the data is highly skewed, we undersample it 
def underSampling(data,ratio=4):
    print("*MESSAGE* Ratio of underSampling is {}:1".format(ratio))
    #data['NormalisedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    #data = data.drop(['Time','Amount'],axis=1)

    # Undersampling Steps
    fraud_count = len(data[data.Class==1])
    fraud_indices = data[data.Class==1].index
    notFraud_indices = data[data.Class==0].index
    rand_notFraud_indices = notFraud_indices[np.random.permutation(len(notFraud_indices))]
    rand_notFraud_indices_undersampled = rand_notFraud_indices[0:ratio*fraud_count]
    undersampled_indices = np.concatenate([fraud_indices,rand_notFraud_indices_undersampled])
    undersampled_data = data.iloc[undersampled_indices,:]
    X_undersampled = undersampled_data.iloc[:,undersampled_data.columns!='Class']
    Y_undersampled = undersampled_data.iloc[:,undersampled_data.columns=='Class']
    return X_undersampled,Y_undersampled

# X,Y are pandas dataframes
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

"""
We don't know, how much undersampling of non-fraud transactions will be good enough
May be a ratio of 1:1 .. but too less data
May be a ration of 10:1 .. but too much of unbalance
Let's plot the accuracy, precision, recall and f-score for various ratio values
A ratio of 3 or 4 looks good, we have taken 4
"""
def plot_underSamplingInfo(data):
    mySVM = SVC(C=1,kernel='poly',class_weight='balanced')
    max_ratio = 5
    accuracy = [0]*max_ratio
    precision = [0]*max_ratio
    recall = [0]*max_ratio
    fscore = [0]*max_ratio
    for i in range(max_ratio):
        X,Y = underSampling(data,i+1)
        tr_data,tr_lb,tst_data,tst_lb = get_tr_tst(X,Y)
        pred = mySVM.fit(tr_data,tr_lb).predict(tst_data)
        count = 0
        for j in range(pred.size):
            if pred[j]==tst_lb[j]: count += 1
        accuracy[i] = count/pred.size
        precision[i],recall[i],fscore[i] = precision_recall_fscore_support(tst_lb,pred,average='binary')[0:3]
    x = [i+1 for i in range(max_ratio)]
    plt.plot(x,accuracy,label='Acuracy')
    plt.plot(x,precision,label='Precision')
    plt.plot(x,recall,label='Recall')
    plt.plot(x,fscore,label='F-Score')
    plt.xlabel('Non-Fraud:Fraud')
    plt.legend(loc="lower left")
    plt.show()

"""
Plot confusion matrix
Code taken from here: https://goo.gl/NcUCCt
"""
def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

"""
SVM has some hyper-parameters and is still considered one of the best classifiers
The other thing to look for is what kernel performs best and it totally depends on data
Our options are 'rbf', 'poly', and 'sigmoid'
We have skipped 'linear' because due to skewness of data set
'linear' seemed to trivial
Hyper-parameter Optimization: The below function deals with finding the best hyper-parameters
for a given kernel
We have used grid-search. Other option is bayesian but we couldn't find any known implementation
"""
def svm(tr_data,tr_lb,tst_data,tst_lb,kernel='rbf'):
    print("*MESSAGE* kernel = {}".format(kernel))
    mySVM = SVC(kernel=kernel,class_weight='balanced')
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
    # PLot confusion matrix
    cnf_matrix = confusion_matrix(tst_lb,pred)
    plot_confusion_matrix(cnf_matrix,np.array(['non-fraud','fraud'],dtype='<U10'),title="Confusion matrix: kernel={}".format(kernel))
    plt.show()

def plot_roc(X,y,classifier):
    cv = StratifiedKFold(n_splits=6)
    tprs,aucs = [],[]
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    X,y = X.as_matrix(),y.as_matrix()
    y.shape = (y.size,)
    for train, test in cv.split(X,y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    print('*MESSAGE* Mean AUC: {}'.format(mean_auc))
    plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example using 6-fold cross-validation')
    plt.legend(loc="lower right")
    plt.show()

def main():
    csv_data = "creditcard.csv"
    data = load_data(csv_data)
    
    # Plot undersampling info to get how much undersampling should we do
    plot_underSamplingInfo(data)
    X,y = underSampling(data)
    
    # Find which kernel performs best
    tr_data,tr_lb,tst_data,tst_lb = get_tr_tst(X,y,0.1)
    svm(tr_data,tr_lb,tst_data,tst_lb,'rbf')
    svm(tr_data,tr_lb,tst_data,tst_lb,'poly')
    svm(tr_data,tr_lb,tst_data,tst_lb,'sigmoid')

    # Now we have the best C for SVM as well as best kernel performance for
    # out dataset, lets analyze the full dataset now to know how good it is
    # The below code is taken from here: https://goo.gl/DcMhma 
    classifier = SVC(C=1, kernel='rbf', probability=True)
    plot_roc(X,y,classifier)
    
    # XGBClassifier
    print("*MESSAGE* ==XGBClassifier==")
    X = data.iloc[:,data.columns!='Class']
    y = data.iloc[:,data.columns=='Class']
    test_size = 0.10
    X_train, X_test, y_train, y_test = train_test_split(X.as_matrix(), y.as_matrix(), test_size=test_size,random_state=seed)
    y_train.shape, y_test.shape = (y_train.size,),(y_test.size,)
    print("*MESSAGE* Train Count: {}".format(Counter(y_train)))
    print("*MESSAGE* Test Count: {}".format(Counter(y_test)))
    clf = XGBClassifier()
    clf.fit(X_train, y_train)

    # Plot confusion matrix for XGBClassifier
    pred = clf.predict(X_test)
    count = 0
    for i in range(pred.size):
        if pred[i]==y_test[i]: count += 1
    print("*MESSAGE* Accuracy: {}".format(count/y_test.size))
    precision,recall,fscore = precision_recall_fscore_support(y_test,pred,average='binary')[0:3]
    print("*MESSAGE* Precision: {}, Recall: {}, F-score: {}".format(precision,recall,fscore))
    cm = confusion_matrix(y_test,pred)
    plot_confusion_matrix(cm,np.array(['non-fraud','fraud'],dtype='<U10'),title='Confusion matrix: XGBClassifier')
    plt.show()

    # Plot ROC for XGBClassifier
    classifier = XGBClassifier()
    plot_roc(X,y,classifier)

if __name__=="__main__":
    main()

