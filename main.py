#========================================================
#	IMPORTS
#========================================================

import pandas as pd         # dataset calcualtions
import numpy as np

import seaborn as sns       # data visualization (in EDA and buidling model)
import matplotlib.pyplot as plt

import xgboost as xgb       # building a model
from sklearn.metrics import classification_report, roc_auc_score



#========================================================
#	GLOBALS
#========================================================

credit_features = pd.read_csv('dataset/credit_features_subset.csv')
loan_applications = pd.read_csv('dataset/loan_applications.csv')

merged_data = pd.merge( credit_features, loan_applications, on='UID', how='left' )



#========================================================
#	TESTING
#========================================================

if (__name__ == '__main__'):
    print(f'merged_data.dtypes:\n{merged_data.dtypes}')
    merged_data.head(20)

    sns.countplot( x='Success', data=merged_data )
    plt.title('How many loans were accepted')
    plt.show()