# Credit Card Fraud Detection
# Dataset Link: https://www.kaggle.com/dalpozz/creditcardfraud
# Dataset Size: 68 MB

import os, sys
import pandas as pd
import numpy as np

# Read the csv nd extract all the required info
csv_data = "creditcard.csv"
dataframe = pd.read_csv(csv_data)
num_pos = dataframe[dataframe['Class']==1].shape[0]
num_neg = dataframe.shape[0]-num_pos

# Print the statics of Data
print("Data statice:")
print(". # of samples        : {}".format(dataframe.shape[0]))
print(". # of features       : {}".format(dataframe.shape[1]-1))
print(". # of +ve/-ve samples: {}/{}".format(num_pos,num_neg))

message = "*MESSAGE* As you can see, out dataset is highly skewed: only {0:.3f}% of data is +ve".format(num_pos*100/(num_pos+num_neg))
print(message)