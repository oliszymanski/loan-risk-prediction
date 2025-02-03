#========================================================
#	IMPORTS
#========================================================

import pandas as pd         # dataset calcualtions
import numpy as np

import seaborn as sns       # data visualization (in EDA and buidling model)
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier




#========================================================
#	GLOBALS
#========================================================

credit_features = pd.read_csv('dataset/credit_features_subset.csv')
loan_applications = pd.read_csv('dataset/loan_applications.csv')

merged_data = pd.merge( credit_features, loan_applications, on='UID', how='left' )



#========================================================
#	MODEL BUILD
#========================================================

model_data = merged_data.drop( columns=[ 'ApplicationDate' ], axis=1 )
model_data = pd.get_dummies( model_data, columns=[ 'LoanPurpose', 'EmploymentType' ] )

X = model_data.drop( 'Success', axis=1 )
y = model_data[ 'Success' ]

print(f'X:\n{X}\n\ny:\n{y}')

df_majority = model_data[model_data['Success'] == 0]
df_minority = model_data[model_data['Success'] == 1]

df_majority_undersampled = resample(
    df_majority,
    replace=False,
    n_samples=len(df_minority),
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f'training set shape:\n{X_train.shape}\n{y_train.shape}')
print(f'testing set shape:\n{X_test.shape}\n{y_test.shape}')

model = RandomForestClassifier( random_state=42, verbose=1 )
history = model.fit( X_train, y_train )