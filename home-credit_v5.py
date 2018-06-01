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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import os

#Read in some data
PATH = "/Users/bexhome/Documents/home-credit-default-risk"
os.listdir(PATH)
application_train = pd.read_csv(PATH+"/application_train.csv")
application_test = pd.read_csv(PATH+"/application_test.csv")
combine = [application_train, application_test]

#The other datasets

application_train = pd.read_csv(PATH+"/application_train_short.csv")
application_test = pd.read_csv(PATH+"/application_test_short.csv")
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

#Notice the big difference in DAYS_EMPLOYED between the mean of
#the two target groups.



#Create Categorical Features from application_train for the following:

NAME_CONTRACT_TYPE = application_train['NAME_CONTRACT_TYPE'].unique()
CODE_GENDER = application_train['CODE_GENDER'].unique()
FLAG_OWN_CAR = application_train['FLAG_OWN_CAR'].unique()
FLAG_OWN_REALTY = application_train['FLAG_OWN_REALTY'].unique()
NAME_TYPE_SUITE = application_train['NAME_TYPE_SUITE'].unique()
NAME_INCOME_TYPE = application_train['NAME_INCOME_TYPE'].unique()
NAME_EDUCATION_TYPE = application_train['NAME_EDUCATION_TYPE'].unique()
NAME_FAMILY_STATUS = application_train['NAME_FAMILY_STATUS'].unique()
NAME_HOUSING_TYPE = application_train['NAME_HOUSING_TYPE'].unique()
OCCUPATION_TYPE = application_train['OCCUPATION_TYPE'].unique()
WEEKDAY_APPR_PROCESS_START = application_train['WEEKDAY_APPR_PROCESS_START'].unique()
ORGANIZATION_TYPE = application_train['ORGANIZATION_TYPE'].unique()

categorical_features = {'NAME_CONTRACT_TYPE': NAME_CONTRACT_TYPE,
                         'CODE_GENDER': CODE_GENDER,
                         'FLAG_OWN_CAR': FLAG_OWN_CAR,
                         'FLAG_OWN_REALTY': FLAG_OWN_REALTY,
                         'NAME_TYPE_SUITE': NAME_TYPE_SUITE,
                         'NAME_INCOME_TYPE': NAME_INCOME_TYPE,
                         'NAME_EDUCATION_TYPE': NAME_EDUCATION_TYPE,
                         'NAME_FAMILY_STATUS': NAME_FAMILY_STATUS,
                         'NAME_HOUSING_TYPE': NAME_HOUSING_TYPE,
                         'OCCUPATION_TYPE': OCCUPATION_TYPE,
                         'ORGANIZATION_TYPE': ORGANIZATION_TYPE,
                         'WEEKDAY_APPR_PROCESS_START': WEEKDAY_APPR_PROCESS_START
                         }

#print("One-Hotting Categoricals\n")
for key in categorical_features:
    application_train = pd.concat([application_train, pd.get_dummies(application_train[key], prefix=[key])], axis=1)
    application_train.drop([key], axis = 1, inplace = True)
    combine = [application_train, application_test]

for key in categorical_features:
    application_test = pd.concat([application_test, pd.get_dummies(application_test[key], prefix=[key])], axis=1)
    application_test.drop([key], axis = 1, inplace = True)
    combine = [application_train, application_test]

#create HasChildren
for dataset in combine:
    dataset['Has0Children'] = pd.Series(len(dataset['CNT_CHILDREN']), index = dataset.index)
    dataset['Has0Children'] = 0
    dataset.loc[dataset['CNT_CHILDREN'] == 0, 'Has0Children'] = 1
    combine = [application_train, application_test]
    
for dataset in combine:
    dataset['Has1Child'] = pd.Series(len(dataset['CNT_CHILDREN']), index = dataset.index)
    dataset['Has1Child'] = 0
    dataset.loc[dataset['CNT_CHILDREN'] == 1, 'Has1Child'] = 1
    combine = [application_train, application_test]

for dataset in combine:
    dataset['Has2Children'] = pd.Series(len(dataset['CNT_CHILDREN']), index = dataset.index)
    dataset['Has2Children'] = 0
    dataset.loc[dataset['CNT_CHILDREN'] == 2, 'Has2Children'] = 1
    combine = [application_train, application_test]
    
for dataset in combine:
    dataset['Has3Children'] = pd.Series(len(dataset['CNT_CHILDREN']), index = dataset.index)
    dataset['Has3Children'] = 0
    dataset.loc[dataset['CNT_CHILDREN'] == 3, 'Has3Children'] = 1
    combine = [application_train, application_test]

#create WasEmployed
for dataset in combine:
    dataset['WasNotEmployed'] = pd.Series(len(dataset['DAYS_EMPLOYED']), index = dataset.index)
    dataset['WasNotEmployed'] = 0
    dataset.loc[dataset['DAYS_EMPLOYED'] == 365243, 'WasNotEmployed'] = 1
    combine = [application_train, application_test]


#Now let's deal with all those NaNs, XNAs, NAs, and 365243 DAYS (1000 yrs) variables
#For now, instead of filling them, let's mass drop them and come back later and work through them

#Percentage of Missing values in other data sets
#Credit: from https://www.kaggle.com/pavanraj159/loan-repayers-v-s-loan-defaulters-home-credit

plt.figure(figsize=(15,15))

plt.subplot(231)
sns.heatmap(pd.DataFrame(bureau.isnull().sum()/bureau.shape[0]*100),annot=True,
            cmap=sns.color_palette("cool"),linewidth=1,linecolor="white")
plt.title("bureau")

plt.subplot(232)
sns.heatmap(pd.DataFrame(bureau_balance.isnull().sum()/bureau_balance.shape[0]*100),annot=True,
            cmap=sns.color_palette("cool"),linewidth=1,linecolor="white")
plt.title("bureau_balance")

plt.subplot(233)
sns.heatmap(pd.DataFrame(credit_card_balance.isnull().sum()/credit_card_balance.shape[0]*100),annot=True,
            cmap=sns.color_palette("cool"),linewidth=1,linecolor="white")
plt.title("credit_card_balance")

plt.subplot(234)
sns.heatmap(pd.DataFrame(installments_payments.isnull().sum()/installments_payments.shape[0]*100),annot=True,
            cmap=sns.color_palette("cool"),linewidth=1,linecolor="white")
plt.title("installments_payments")

plt.subplot(235)
sns.heatmap(pd.DataFrame(pos_cash_balance.isnull().sum()/pos_cash_balance.shape[0]*100),annot=True,
            cmap=sns.color_palette("cool"),linewidth=1,linecolor="white")
plt.title("pos_cash_balance")

plt.subplot(236)
sns.heatmap(pd.DataFrame(previous_application.isnull().sum()/previous_application.shape[0]*100),annot=True,
            cmap=sns.color_palette("cool"),linewidth=1,linecolor="white")
plt.title("previous_application")

plt.subplots_adjust(wspace = 1.6)

#Data Droppings
##Delete all columns and return and dive deeper into moulding them out

#missing application_train data
total = application_train.isnull().sum().sort_values(ascending = False)
percent = (application_train.isnull().sum()/application_train.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
application_train = application_train.drop((missing_data[missing_data['Total'] > 279]).index, 1)
combine = [application_train, application_test]

#missing application_test data
total = application_test.isnull().sum().sort_values(ascending = False)
percent = (application_test.isnull().sum()/application_train.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
application_test = application_test.drop((missing_data[missing_data['Total'] > 25]).index, 1)
combine = [application_train, application_test]

#Filling in missing application_train values (4 cols) with the most common occurance

freq_AGP = application_train.AMT_GOODS_PRICE.dropna().mode()[0]
#print("Freq_AGP:", freq_AGP)
application_train['AMT_GOODS_PRICE'] = application_train['AMT_GOODS_PRICE'].fillna(freq_AGP)
combine = [application_train, application_test]
#Check
#print(application_train[['AMT_GOODS_PRICE', 'TARGET']].
#      groupby(['AMT_GOODS_PRICE'], as_index=False).mean().
#      sort_values(by='TARGET', ascending=False))


freq_AA = application_train.AMT_ANNUITY.dropna().mode()[0]
#print("freq_AA:", freq_AA)
application_train['AMT_ANNUITY'] = application_train['AMT_ANNUITY'].fillna(freq_AA)
combine = [application_train, application_test]
#Check
#print(application_train[['AMT_ANNUITY', 'TARGET']].
#      groupby(['AMT_ANNUITY'], as_index=False).mean().
#      sort_values(by='TARGET', ascending=False))

freq_CFM = application_train.CNT_FAM_MEMBERS.dropna().mode()[0]
#print("freq_CFM", freq_CFM)
application_train['CNT_FAM_MEMBERS'] = application_train['CNT_FAM_MEMBERS'].fillna(freq_CFM)
combine = [application_train, application_test]
#Check
#print(application_train[['CNT_FAM_MEMBERS', 'TARGET']].
#      groupby(['CNT_FAM_MEMBERS'], as_index=False).mean().
#      sort_values(by='TARGET', ascending=False))

application_train['DAYS_LAST_PHONE_CHANGE'] = application_train['DAYS_LAST_PHONE_CHANGE'].fillna(0)
combine = [application_train, application_test]

#Filling in for application_test
freq_A_A = application_test.AMT_ANNUITY.dropna().mode()[0]
#print("freq_A_A:", freq_A_A)
application_test['AMT_ANNUITY'] = application_test['AMT_ANNUITY'].fillna(freq_A_A)
combine = [application_train, application_test]
#Check
#print(application_train[['AMT_ANNUITY', 'TARGET']].
#      groupby(['AMT_ANNUITY'], as_index=False).mean().
#      sort_values(by='TARGET', ascending=False))

freq_ES2 = application_test.EXT_SOURCE_2.dropna().mode()[0]
#print("freq_ES2:", freq_ES2)
application_test['EXT_SOURCE_2'] = application_test['EXT_SOURCE_2'].fillna(freq_ES2)
combine = [application_train, application_test]
#Check
#print(application_train[['EXT_SOURCE_2', 'TARGET']].
#      groupby(['EXT_SOURCE_2'], as_index=False).mean().
#      sort_values(by='TARGET', ascending=False))

#Application_train has 3 more features than test does
application_test["['NAME_INCOME_TYPE']_Maternity Leave"] = 0
application_test["['NAME_FAMILY_STATUS']_Unknown"] = 0
application_test['DuumyVar'] = 0

#Convert 365243 to 0
#Check for 365243 in the other columns & confirm these changes properly made
for dataset in combine:
    dataset['DAYS_BIRTH'] = dataset['DAYS_BIRTH'].replace(365243, 0)
    dataset['DAYS_EMPLOYED'] = dataset['DAYS_EMPLOYED'].replace(365243, 0)
    dataset['DAYS_ID_PUBLISH'] = dataset['DAYS_ID_PUBLISH'].replace(365243, 0)
    combine = [application_train, application_test]

#Confirm no missing data left
print("\nRemaining Missing Training Data:\n", application_train.isnull().sum().max()) 
print("\nRemaining Missing Testing Data:\n", application_test.isnull().sum().max(), "\n")

#Shape after wrangling
print("\nShape of Data After Wrangling it:\n", "Train Shape:\n", application_train.shape, 
      "\nTest Shape:\n", application_test.shape, "\nCombine Shapes:\n", combine[0].shape, 
      combine[1].shape)

#vars for calculating MAE
x = application_train.drop(['TARGET'], axis=1)
y = application_train['TARGET']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)

#x_train, y_train, and x_test shapes
print("x_train shape:\n", x_train.shape, "\ny_train shape:\n", y_train.shape,"\nx_test shape:\n",
      x_test.shape)

#Mean Absolute Error
regressor = KNeighborsRegressor()
regressor.fit(x_train, y_train)
y_est = regressor.predict(x_test)
print("MAE=", mean_squared_error(y_test,y_est))
#output: MAE= 0.087362892867

#Reducing the MAE further by normalizing the input features using Z-scores:
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
regressor = KNeighborsRegressor()
regressor.fit(x_train_scaled, y_train)
y_est = regressor.predict(x_test_scaled)
print("MAE =", mean_squared_error(y_test, y_est))


#Reducing MAE further using RobustScaler, which is impervious to outliers
#via using median and IQR instead of mean:
scaler2 = RobustScaler()
x_train_scaled = scaler2.fit_transform(x_train)
x_test_scaled = scaler2.transform(x_test)
regressor = KNeighborsRegressor()
regressor.fit(x_train_scaled, x_test_scaled)
print("MAE =", mean_squared_error(y_test, y_est))

#PCA
#It's just like restructuring the information in the dataset by aggregating as
#much as possible of the information onto the initial vectors produced by the
#PCA.

#PCA on a 2D space
pca_1c = PCA(n_components = 1)
x_pca_1c = pca_1c.fit_transform(application_train)

plt.scatter(x_pca_1c[:, 0], np.zeros(x_pca_1c.shape), c = application_train.TARGET,
           alpha = 0.8, s = 60, marker = 'o', edgecolors = 'white')
plt.show()
print("PCA 2d Explained Variance Ratio:", pca_1c.explained_variance_ratio_.sum())
#PCA 2d Explained Variance Ratio: 0.717023659086

#Randomized PCA
rpca_70c = PCA(svd_solver = 'randomized', n_components = 70)
x_rpca_70c = rpca_70c.fit_transform(application_train)

plt.scatter(x_rpca_70c[:, 0], x_rpca_70c[:, 1], c = application_train.TARGET,
           alpha = 0.8, s = 60, marker = 'o', edgecolors = 'white')
plt.show()
print("Randomized PCA Explained Variance Ratio:", rpca_70c.explained_variance_ratio_.sum())
#Randomized PCA Explained Variance Ratio: 0.99999999992

#Standard PCA
pca_70c = PCA(n_components = 70)
x_pca_70c = pca_70c.fit_transform(application_train)
print("x_pca_70 shape:\n", x_pca_70c.shape)

plt.scatter(x_pca_70c[:, 0], x_pca_70c[:, 1], c = application_train.TARGET,
           alpha = 0.8, s = 60, marker = 'o', edgecolors = 'white')
plt.show()
print("PCA Explained Variance Ratio:", pca_70c.explained_variance_ratio_.sum())
#n = 70 PCA Explained Variance Ratio: 0.999999999999
#n = 60 PCA Explained Variance Ratio: 0.999999999998
#n = 50 PCA Explained Variance Ratio: 0.999999999997
#n = 10 PCA Explained Variance Ratio: 0.999999999969

#PCA with whitening
pca_70cw = PCA(n_components = 70, whiten = True)
x_pca_70cw = pca_70cw.fit_transform(application_train)

plt.scatter(x_pca_70cw[:, 0], x_pca_70cw[:, 1], c = application_train.TARGET,
           alpha = 0.8, s = 60, marker = 'o', edgecolors = 'white')
plt.show()

print("Whitened PCA Explained Variance Ratio:", pca_70cw.explained_variance_ratio_.sum())
#Whitened PCA Explained Variance Ratio: 0.999999999999

#vars for modeling
x_train = application_train.drop('TARGET', axis = 1)
y_train = application_train['TARGET']
x_test = application_test.drop('SK_ID_CURR', axis = 1).copy()

#Now for some modeling!

#Logistic Regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)
print(acc_log) 

#Feature Correlation
coeff_df = pd.DataFrame(application_train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
print(coeff_df.sort_values(by='Correlation', ascending=False))

#Support Vector Machines (SVM)
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_svc = round(svc.score(x_train, y_train) * 100, 2)
print(acc_svc)

#k-NN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_knn = round(knn.score(x_train, y_train) * 100, 2)
print(acc_knn) 

#Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_test)
acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)
print(acc_gaussian) 

#Perceptron
perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_test)
acc_perceptron = round(perceptron.score(x_train, y_train) * 100, 2)
print(acc_perceptron) 

#Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_test)
acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)
print(acc_linear_svc) 

#Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_test)
acc_sgd = round(sgd.score(x_train, y_train) * 100, 2)
print(acc_sgd) 

#Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
print(acc_decision_tree) 

#Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)
print(acc_random_forest) 

#Model Evaluation

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Stochastic Gradient Descent', \
              'Linear SVC', 'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian, 
              acc_perceptron, acc_sgd, acc_linear_svc, acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))

submission = pd.DataFrame({
    "SK_ID_CURR": application_test["SK_ID_CURR"],
    "TARGET": y_pred
    })

submission.to_csv('../Desktop/submission.csv', index = False)


