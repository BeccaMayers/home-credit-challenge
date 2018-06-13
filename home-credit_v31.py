#Home Credit Challenge v11

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, Imputer
#metrics
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.metrics import roc_auc_score
#models
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
#suppress warnings
import warnings
warnings.filterwarnings('ignore')
#misc
import gc

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

##Data Reducing Memory Pattern
#Got this from this kernel: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    #iterate through all the columns of a dataframe and modify 
    #the data type to reduce memory usage.        
    
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def import_data(file):
    #create a dataframe and optimize its memory usage
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df

###
#Get data
###
path = '/Users/bexhome/Documents/home-credit-default-risk/data'
df0 = import_data(path + '/application_train.csv')
df1 = import_data(path + '/application_test.csv')
df2 = import_data(path + '/previous_application.csv')

#Read in remaining data
"""
bureau = pd.read_csv(path + '/bureau.csv')
b_balance = pd.read_csv(path + '/bureau_balance.csv')
cc_balance = pd.read_csv(path + '/credit_card_balance.csv')
installments = pd.read_csv(path + '/installments_payments.csv')
pcb = pd.read_csv(path + '/PS_CASH_Balance.csv')
"""

#Join train/test
df01 = pd.concat((df0, df1)).reset_index(drop=True)

#Examine Distribution of Target Column
target_distribution = df01['TARGET'].value_counts()
print("\nTarget distribution\n", target_distribution)

"""
#Visualize it
df01['TARGET'].astype(int).plot.hist();
plt.show()
"""

#Examine Missing Values
def missing_values_table(df):
    #Total missing values
    mis_val = df.isnull().sum() 
    #Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    #Table with results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis = 1)
    #Rename columns
    mis_val_table_rename = mis_val_table.rename(
        columns = {0 : 'Missing Values',
                   1 : 'Percentage Total Values'})
    #Sort
    mis_val_table_rename = mis_val_table_rename[mis_val_table_rename.iloc[:,1] != 0].sort_values('Percentage Total Values', ascending = False).round(1)
    #Print summary
    print('\nYour selected dataframe has ' + str(df.shape[1]) + ' columns. \n'
          'There are ' + str(mis_val_table_rename.shape[0]) + ' columns that have missing values.')
    #Return
    return mis_val_table_rename

#Missing Values
missing = missing_values_table(df01)
print("\nMissing Values\n", missing)

#Data column types
print("\nDtypes:")
print(df01.dtypes.value_counts())

#Unique classes per column
print("\nUnique classes:")
print(df01.select_dtypes('object').apply(pd.Series.nunique, axis = 0))

#Repairs to Features
#Got this from: https://www.kaggle.com/kingychiu/home-credit-eda-distributions-and-outliers
df01 = df01[df01['AMT_INCOME_TOTAL'] != 1.170000e+08]
df01 = df01[df01['AMT_REQ_CREDIT_BUREAU_QRT'] != 261]
df01 = df01[df01['OBS_30_CNT_SOCIAL_CIRCLE'] < 300]
df01['DAYS_EMPLOYED'] = df01['DAYS_EMPLOYED'].replace(365243, 0)

#Label encoding for <2 unique classes

#create label encoder object
le = LabelEncoder()
le_count = 0

#iterate
for col in df0:
    if df0[col].dtype == 'object':
        #if 2 or fewer unique categories
        if len(list(df0[col].unique())) <= 2:
            #Train
            le.fit(df0[col])
            #Transform
            df0[col] = le.transform(df0[col])
            df1[col] = le.transform(df1[col])
            #keep track of how many columns were label encoded
            le_count += 1
            
print("%d columns were label encoded. \n" % le_count)

#One-hotting
df0 = pd.get_dummies(df0)
df1 = pd.get_dummies(df1)
print("One-Hotted Training Features shape:", df0.shape)
print("One-Hotted Testing Features shape:", df1.shape)

#Check one-hotting
#print("\nDF Head\n", df0.head())

#Align the training and testing data
train_labels = df0['TARGET']

#align
df0, df1 = df0.align(df1, join = 'inner', axis = 1)

print("Aligned Training Features shape:", df0.shape)
print("Aligned Testing Features shape:", df1.shape, "\n")

#Add target back into df
df0['TARGET'] = train_labels

#Find correlations with target and sort
correlations = df0.corr()['TARGET'].sort_values()

#Let's see 'em
print("Most Positive Correlations:\n", correlations.tail(15))
print("\nMost Negative Correlations:\n", correlations.head(15))

#Effect of Age on Repayment

#Find correlation of positive days since birth and target
df0['DAYS_BIRTH'] = abs(df0['DAYS_BIRTH'])
df0['DAYS_BIRTH'].corr(df0['TARGET'])
#Note: -0.07 is not a significant correlation, but clearly age affects
#target, so consider binning ages

#Visualization
#set plot style
plt.style.use('seaborn-whitegrid')

#plot distribution of age in years
plt.hist(df0['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)
plt.title('Age of Client'); plt.xlabel('Age(years)'); 
plt.ylabel('Count');
#Note there are no outliers in age

#Kernel Density Estimation Plot
plt.figure(figsize = (10, 8))

#Plot repaid-on-time loans 
sns.kdeplot(df0.loc[df0['TARGET'] == 0, 'DAYS_BIRTH'] / 365, 
            label = 'Paid (Target 0)')

#Plot not-repaid-on-time loans 
sns.kdeplot(df0.loc[df0['TARGET'] == 1, 'DAYS_BIRTH'] / 365, 
            label = 'Unpaid (Target 1)')

#Labeling  
plt.xlabel('Age (Years)'); plt.ylabel('Density'); 
plt.title('Distribution of Ages');
plt.show()
#Age Binning

#Separate df for age info
age_data = df0[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

#Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], 
                                  bins = np.linspace(20, 70, num = 11))
age_data.head(10)

#Group by bin & calculate averages
age_groups = age_data.groupby('YEARS_BINNED').mean()
print(age_groups)

#Failure to Repay by Age Group
plt.figure(figsize = (8, 8), facecolor = 'white')

#Bar plot - age bins and the target average
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

#Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); 
plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');
plt.show()
#Extract EXT_SOURCE features and show correlations
ext_data = df0[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs = ext_data.corr()
print(ext_data_corrs)

#Heatmap of correlations
plt.figure(figsize = (8, 6), facecolor = 'white')
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, 
            vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');
plt.show()
#Effect of EXT_SOURCE on target 
plt.figure(figsize = (10, 12), facecolor = 'white')
plt.style.use('seaborn-whitegrid')

#Iterate 
for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                            'EXT_SOURCE_3']):
    #Create a new subplot for each source
    plt.subplot(3, 1, i + 1)
    #Plot repaid loans
    sns.kdeplot(df0.loc[df0['TARGET'] == 0, source], label = 'repaid (target 0)')
    #Plot loans that were not repaid
    sns.kdeplot(df0.loc[df0['TARGET'] == 1, source], label = 'unrepaid (target 1)')
    
    #Label the plots
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source); plt.ylabel('Density');   
plt.show()
plt.tight_layout(h_pad = 2.5)

#Pairs Plot

#Copy the data for plotting
plot_data = ext_data.drop(columns = ['DAYS_BIRTH']).copy()

#Add in the age of the client in years
plot_data['YEARS_BIRTH'] = age_data['YEARS_BIRTH']

#Drop na values and limit to first 100000 rows
plot_data = plot_data.dropna().loc[:100000, :]

#Function to calculate correlation coefficient between two columns
def corr_func(x, y, **kwargs):
    r = np.corrcoef(x, y)[0][1]
    ax = plt.gca()
    ax.style.use('seaborn-whitegrid')
    ax.annotate("r = {:.2f}".format(r),
                xy=(.2, .8), xycoords=ax.transAxes,
                size = 20)

#Create the pairgrid object
grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False,
                    hue = 'TARGET', 
                    vars = [x for x in list
                            (plot_data.columns) if x != 'TARGET'])

#Upper is a scatter plot
grid.map_upper(plt.scatter, alpha = 0.2)

#Diagonal is a histogram
grid.map_diag(sns.kdeplot)

#Bottom is density plot
grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r);

plt.suptitle('Ext Source and Age Pairs Plot', size = 32, y = 1.05);
plt.show()
#Polynomials - powers of existing features and their interactions
#Combining them might show a relationship with the target

#Make a new dataframe 
poly_features = df0[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
                    'DAYS_BIRTH', 'TARGET']]
poly_features_test = df1[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
                          'DAYS_BIRTH']]

#Imputer for handling missing values
imputer = Imputer(strategy = 'median')
poly_target = poly_features['TARGET']
poly_features = poly_features.drop(columns = ['TARGET'])

#Impute missing values
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)
                                
#Create polynomials. Degree of 3 helps prevent overfitting
poly_transformer = PolynomialFeatures(degree = 3) 

#Train the polynomials
poly_transformer.fit(poly_features)

#Transform features
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('\nPolynomial Features shape: ', poly_features.shape, "\n")

#Name new features using `get_feature_names` method.

poly_transformer.get_feature_names(input_features = 
                                   ['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                    'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]

#Correlatation of new features with the target

#Create df 
poly_features = pd.DataFrame(poly_features, columns = poly_transformer.
                             get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                'EXT_SOURCE_3', 'DAYS_BIRTH']))

#Add target
poly_features['TARGET'] = poly_target

#Find correlations 
poly_corrs = poly_features.corr()['TARGET'].sort_values()

#Display most negative & most positive
print("Most negative\n", poly_corrs.head(10), "\n")
print("Most positive\n", poly_corrs.tail(5), "\n")

#Some of the new variables have greater correlation with the target.
#Let's add them in and eval the model with and without them.

#Put test features into df
poly_features_test = pd.DataFrame(poly_features_test, columns = poly_transformer.
                                  get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                    'EXT_SOURCE_3', 'DAYS_BIRTH']))

#Merge polynomials into training dataframe
poly_features['SK_ID_CURR'] = df0['SK_ID_CURR']
df_poly = df0.merge(poly_features, on = 'SK_ID_CURR', how = 'left')

#Merge polnomial features into testing df
poly_features_test['SK_ID_CURR'] = df1['SK_ID_CURR']
df1_poly = df1.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')

#Align the dataframes
df_poly, df1_poly = df_poly.align(df1_poly, join = 'inner', axis = 1)

#Print out the new shapes
print('Training data with polynomial features shape: ', df_poly.shape)
print('Testing data with polynomial features shape:  ', df1_poly.shape, "\n")

#Drop target from training data
if 'TARGET' in df0:
    train = df0.drop(columns = ['TARGET'])
else:
    train = df0.copy()
features = list(train.columns)

#Copy the testing data
test = df1.copy()

#Median imputation of missing values
imputer = Imputer(strategy = 'median')

#Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

#Fit to the training data
imputer.fit(train)

#Transform both training and testing data
train = imputer.transform(train)
test = imputer.transform(df1)

#Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)

#Outliers?


#Multilabel Classification
x_train, x_test, y_train, y_test = train_test_split(df0, df0.TARGET,
                                                   test_size = 0.50,
                                                   random_state = 4)
classifier = DecisionTreeClassifier(max_depth = 2)
classifier.fit(train, test)
y_pred = classifier.predict(x_test)

#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
img = plt.matshow(cm, cmap = plt.cm.autumn)
plt.colorbar(img, fraction = 0.045)
for x in range(cm.shape[0]):
    for y in range(cm.shape[1]):
        plt.text(x, y, "%0.2f" % cm[x, y], size = 12, color = 'black',
                ha = 'center', va = 'center')
plt.show()

#Logistic Regression - lower regularlization to decrease overfitting

#Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)

#Train 
log_reg.fit(train, train_labels)

#Make some predictions - (second column only!)
lr_pred = log_reg.predict_proba(test)[:, 1]


#Confidence score
#acc_log = round(log_reg.score(train, train_labels) * 100, 2)
#print(acc_log) #outputs confidence score, NaN issue

#Random Forest

#Make random forest classifier
random_forest = RandomForestClassifier(n_estimators = 100, 
                                       random_state = 50)
#Train 
random_forest.fit(train, train_labels)

#Make predictions 
rf_pred = random_forest.predict_proba(test)[:, 1]

#Random Forest with Engineered Features - Test 

poly_features_names = list(df_poly.columns)

#Impute 
imputer = Imputer(strategy = 'median')

poly_features = imputer.fit_transform(df_poly)
poly_features_test = imputer.transform(df1_poly)

#Scale 
scaler = MinMaxScaler(feature_range = (0, 1))

poly_features = scaler.fit_transform(poly_features)
poly_features_test = scaler.transform(poly_features_test)

random_forest_poly = RandomForestClassifier(n_estimators = 100, 
                                            random_state = 50)
#Train 
random_forest_poly.fit(poly_features, train_labels)

#Make predictions 
rf_p_pred = random_forest_poly.predict_proba(poly_features_test)[:, 1]

#Light Gradient Boosting!

#Formatting
train = np.array(df.drop(columns = 'TARGET'))
test = np.array(df1)

train_labels = np.array(train_labels).reshape((-1, ))

#10 fold cross-validation
folds = KFold(n_splits=5, shuffle=True, random_state=50)

#Validation and test predictions
valid_preds = np.zeros(train.shape[0])
lgb_preds = np.zeros(test.shape[0])

#Iterate 
for n_fold, (train_indices, valid_indices) in enumerate(folds.split(train)):
    #Training data for the fold
    train_fold, train_fold_labels = train[train_indices, :], train_labels[train_indices]
    
    #Validation data for the fold
    valid_fold, valid_fold_labels = train[valid_indices, :], train_labels[valid_indices]
    
    #LightGBM classifier with hyperparameters
    clf = LGBMClassifier(
        n_estimators=10000,
        learning_rate=0.1,
        subsample=.8,
        max_depth=7,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01,
        min_child_weight=2
    )
    
    #Fit to the training data, evaluate via the validation data
    clf.fit(train_fold, train_fold_labels, 
            eval_set= [(train_fold, train_fold_labels), (valid_fold, valid_fold_labels)], 
            eval_metric='auc', early_stopping_rounds=100, verbose = False
           )
    
    #Validation preditions
    valid_preds[valid_indices] = clf.predict_proba(valid_fold, num_iteration=clf.best_iteration_)[:, 1]
    
    #Testing predictions
    lgb_preds += clf.predict_proba(test, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
    #Display performance for current fold
    print('Fold %d AUC : %0.6f' % (n_fold + 1, roc_auc_score(valid_fold_labels, valid_preds[valid_indices])))
    
    #Delete variables to free up memory
    del clf, train_fold, train_fold_labels, valid_fold, valid_fold_labels
    gc.collect()

    #Logistic Regression Submission

#Submission dataframe
submit = df1[['SK_ID_CURR']]
submit['TARGET'] = lr_pred
#submit.head()

#Save submission to csv
submit.to_csv(path + 'lr_submission.csv', index = False)

#Random Forest Submission
#Make submission df
submit = df1[['SK_ID_CURR']]
submit['TARGET'] = rf_pred

#Save submission df
submit.to_csv(path + 'rf_submission.csv', index = False)

#Random Forest w Polynomials Submission 

#Make submission df
submit = df1[['SK_ID_CURR']]
submit['TARGET'] = rf_p_pred

# Save the submission dataframe
submit.to_csv(path + 'random_forest_baseline_engineered.csv', index = False)

#Light GBM Submission
    
#Make submission df
submit = df1[['SK_ID_CURR']]
submit['TARGET'] = lgb_preds

# Save the submission file
submit.to_csv(path + "light_gbm_baseline.csv", index=False)

#Function to calculate and show feature importances
def show_feature_importances(model, features):
    plt.figure(figsize = (12, 8))
    #df of feature importances sorted most to least
    results = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    results = results.sort_values('importance', ascending = False)
    
    #Display
    print(results.head(10))
    print('\nNumber of features with importance greater than 0.01 = ', np.sum(results['importance'] > 0.01))
    
    #Plot
    results.head(20).plot(x = 'feature', y = 'importance', kind = 'barh',
                     color = 'red', edgecolor = 'k', title = 'Feature Importances');
    return results

#Show feature importances for default features
feature_importances = show_feature_importances(random_forest, features)

