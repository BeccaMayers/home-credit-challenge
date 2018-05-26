#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 20:19:59 2018

@author: bexhome
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import os

#Read in some data
PATH = "/Users/bexhome/Documents/home-credit-default-risk"
os.listdir(PATH)
application_train = pd.read_csv(PATH+"/application_train.csv")
application_test = pd.read_csv(PATH+"/application_test.csv")
bureau = pd.read_csv(PATH+"/bureau.csv")
bureau_balance = pd.read_csv(PATH+"/bureau_balance.csv")
credit_card_balance = pd.read_csv(PATH+"/credit_card_balance.csv")
installments_payments = pd.read_csv(PATH+"/installments_payments.csv")
previous_application = pd.read_csv(PATH+"/previous_application.csv")
POS_CASH_balance = pd.read_csv(PATH+"/POS_CASH_balance.csv")


#Check out the data
data_dict = {'application_train': application_train, 
              'application_test': application_test, 
              'bureau': bureau, 
              'bureau_balance': bureau_balance, 
              'credit_card_balance': credit_card_balance, 
              'installments_payments': installments_payments, 
              'previous_application': previous_application, 
              'POS_CASH_balance': POS_CASH_balance
              }

#Data shape
print("Shape of Data Observations and Features\n")
for key, value in data_dict.items():
    print(key, "observations:", value.shape[0], "\n", key, "features:", value.shape[1])

#Data head
print("\nData Head\n")
for key, value in data_dict.items():
    print(key, "head:\n", value.head())

#Data columns
print("\nData Columns\n")
for key, value in data_dict.items():
    print(key, "columns:\n", value.columns.values, "\n")

#Data describe
print("\nData Describe\n")
print(application_train.describe())
#to describe the remaining data:
#for key, value in data_dict.items():
#    print(key, "describe:\n", value.describe(), "\n")

#Missing Data
print("\nMissing Data\n")
for key, value in data_dict.items():
    print(key, "missing data:\n")
    total = value.isnull().sum().sort_values(ascending = False)
    percent = (value.isnull().sum()/value.isnull().count()*100).sort_values(ascending = False)
    missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(20), "\n")

#Non-zero Data
print("\nNon-Zero Data\n")
non_zeros = application_train.apply(np.count_nonzero, axis = 0).head()
print(non_zeros)

#Groupby
print("\nGroupby AMT_INCOME_TOTAL and TARGET\n")
print(application_train[['AMT_INCOME_TOTAL', 'TARGET']].
      groupby(['AMT_INCOME_TOTAL'], as_index = False).mean().
      sort_values(by = 'TARGET', ascending = True))

print("\nGroupby AMT_CREDIT and TARGET\n")
print(application_train[['AMT_CREDIT', 'TARGET']].
      groupby(['AMT_CREDIT'], as_index = False).mean().
      sort_values(by = 'TARGET', ascending = True))

#Check out the statistics of specific features
print("\nGrouped Target Mean\n")
grouped_target_mean = application_train.groupby(['TARGET']).mean()
print(grouped_target_mean)

"""
Notice the big difference in DAYS_EMPLOYED between the mean of
the two target groups.
"""
"""
###
#Visualize This
print("\nVisualization\n")

scatterplot = application_train.plot(kind = 'scatter', 
                                     x = 'AMT_INCOME_TOTAL',
                                     y = 'AMT_CREDIT', s = 64, c = 'blue',
                                    edgecolors = 'white')

#Heatmap for Application_Train

#*Needs nans removed*
#*IndexError: Inconsistent shape between the condition and the input (got (122, 1) and (122,))
x = application_train.columns.values
y = application_train.columns.values
 
plt.title('Application_Train Heatmap')
plt.ylabel('y')
plt.xlabel('x')
#plt.imshow(heatmap)
#plt.imshow(heatmap, extent=extent)
sns.heatmap(x, y)
plt.show()
"""

#Create Categorical Features from application_train for the following:
"""
NAME_CONTRACT_TYPE
CODE_GENDER 
FLAG_OWN_CAR 
FLAG_OWN_REALTY 
CNT_CHILDREN (has children/doesn't have children) 
NAME_TYPE_SUITE (who accompanied client; accompanied or alone) NAME_INCOME_TYPE 
NAME_EDUCATION_TYPE 
NAME_HOUSING_TYPE 
DAYS_EMPLOYED (employed or not) 
OCCUPATION_TYPE*
"""

# use pd.concat to join the new columns with your original dataframe
application_train = pd.concat([application_train,pd.get_dummies(application_train['NAME_CONTRACT_TYPE'], prefix='NAME_CONTRACT_TYPE')],axis=1)

#drop the original column 
application_train.drop(['NAME_CONTRACT_TYPE'], axis = 1, inplace = True)
print(application_train.head())

"""

#Feature Importance Using Random Forest
categorical_feats = [
    f for f in application_train.columns if application_train[f].dtype == 'object'
]

for col in categorical_feats:
    lb = preprocessing.LabelEncoder()
    lb.fit(list(application_train[col].values.astype('str')) + list(application_test[col].values.astype('str')))
    application_train[col] = lb.transform(list(application_train[col].values.astype('str')))
    application_test[col] = lb.transform(list(application_test[col].values.astype('str')))

application_train.fillna(-999, inplace = True)


rf = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=4, max_features=0.5, random_state=2018)
rf.fit(application_train.drop(['SK_ID_CURR', 'TARGET'],axis=1), application_train.TARGET)
features = application_train.drop(['SK_ID_CURR', 'TARGET'],axis=1).columns.values
"""
