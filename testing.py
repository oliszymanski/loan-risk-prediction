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

model_data = merged_data.drop( columns=[ 'ApplicationDate' ], axis=1 )
model_data = pd.get_dummies( model_data, columns=[ 'LoanPurpose', 'EmploymentType' ] )

X = model_data.drop( 'Success', axis=1 )
y = model_data[ 'Success' ]

#========================================================
#	EDA
#========================================================

print(f'merged_data.dtypes:\n{merged_data.dtypes}')

merged_data.head(20)
merged_data.info()

print( merged_data.isnull().sum() )
merged_data.describe()

sns.countplot( x='Success', data=merged_data )
plt.title('How many loans were accepted?')
plt.grid()

sns.countplot(x='EmploymentType', hue='Success', data=merged_data)
plt.title('Which employment type has mostly asked for a loan?')
plt.grid()

plt.figure( figsize=(8, 6) )
sns.scatterplot( x='Amount', y='Term', data=merged_data, hue='Success' )
plt.grid()

sns.histplot( x='Amount', data=merged_data, bins=30, kde=True )

merged_data.hist(figsize=(20, 15))
plt.show()

# Correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(merged_data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()



#========================================================
#	MODEL TESTING, EVALUATION AND HYPERPARAMETER TUNING
#========================================================

y_pred = model.predict( X_test )
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print( f'ROC-AUC score: {roc_auc_score(y_test, y_proba)}' )

cv_scores = cross_val_score( model, X_train, y_train, cv=5, scoring='roc_auc' )
print( f'Mean ROC-AUC: { cv_scores.mean() }' )

cm = confusion_matrix( y_test, y_pred )
sns.heatmap( cm, annot=True, xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'], cmap='Blues' )

param_grid = {
    'n_estimators' : [100, 200, 500],
    'max_depth' : [ 3, 5, 7, None ],
    'min_samples_split' : [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42, verbose=1), param_grid, cv=5, scoring="roc_auc")
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)