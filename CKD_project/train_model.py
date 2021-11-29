import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import feather
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
from utility_func.utility import *
from pandas_summary import DataFrameSummary
from sklearn import metrics as met
import joblib

np.random.seed(42)
np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 20
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


df = pd.read_csv('input/data/raw_data.csv') # Reading the Dataset

# Data analysis

# rename the catagory column 
df.rename(columns={'classification' : 'class'}, inplace=True)
df.drop('id', axis=1, inplace=True)


# Function to find the categorical and numerical features
def extract_cat_num(df):
    cat_col=[col for col in df.columns if df[col].dtype=='O']
    num_col=[col for col in df.columns if df[col].dtype!='O']
    return cat_col,num_col 


cat_col,num_col=extract_cat_num(df)



# Convert the object string  features into float
for col in ['pcv','wc','rc']:
    df[col] = df[col].str.extract('(\d+\.\d+|\d+)').astype(float)
    
    
# Remove the tab in the categorical variables 

df['cad'].replace(to_replace={'\tno':'no'}, inplace=True)
df['dm'].replace(to_replace={'\tno':'no','\tyes':'yes',' yes':'yes'}, inplace=True)
df['class'].replace(to_replace={'ckd\t':'ckd'},inplace=True)

os.makedirs('./input/processed/', exist_ok=True)
df.to_feather('./input/processed/ckd-processed')

df = feather.read_dataframe('input/processed/ckd-processed')

for cols in ['sg', 'al', 'su']:
    df[cols] = df[cols].astype('category')
    
#Change columns of strings in dataframe to a column of categorical values
train_cats(df)

# collect the categorical varibales in dictionary 
print("\n=========Categorical cariable in dictionary format:==========")
for cols in df.columns:
    if df[cols].dtype.name == 'category':
        print(cols, dict(enumerate(df[cols].cat.categories)) )
        

# convert the appet and class categories for 1 for good/positive and 0 for poor/ negative to make constitent 
df['appet'].cat.set_categories(['poor', 'good'], ordered=True, inplace=True)
df['class'].cat.set_categories(['notckd', 'ckd'], ordered=True, inplace=True)


# Save the semi processed data 
df.to_feather('./input/processed/ckd_semi_processed')


# Creating Feature Variable and Target Variable

df = feather.read_dataframe('input/processed/ckd_semi_processed')
        
# splits off the response variable, and replaced NAN by the median value of the column.
features, response, nas = proc_df(df, 'class')


# Feature Scaling 
scaler = MinMaxScaler()
x_features = scaler.fit_transform(features)
x_features_scaled = pd.DataFrame(x_features)


# Creating training set and test set 
x_train, x_test, y_train, y_test = train_test_split(x_features_scaled, response, test_size=0.25, stratify=response, random_state=42)


# Remove the augmented features during impuatation 
x_train.drop(x_train.iloc[:,24:], axis=1, inplace=True)
x_test.drop(x_test.iloc[:,24:], axis=1, inplace=True)


# Build Model

score = []
model = []

from sklearn.model_selection import cross_val_score

# utility function to print cross k-fold cross validation scores, their mean and variance(standard deviation)
print("\n ================= Trained model statistics: ================\n")
def print_scores(model, x_train, y_train, cv, scoring):
    print('Cross validation scores:', cross_val_score(model, x_train.values, y_train, cv=cv, scoring=scoring, n_jobs=-1) )
    print( 'Mean_scores:', np.mean( cross_val_score(model, x_train.values, y_train, cv=cv, scoring=scoring, n_jobs=-1) ) )
    print( 'Variance:', np.std( cross_val_score(model, x_train.values, y_train, cv=cv, scoring=scoring, n_jobs=-1) ) )
    

# Random Forest Classifier 

random_forest = RandomForestClassifier(random_state=42, n_jobs=-1)
print_scores(random_forest, x_train, y_train, 5, 'accuracy')

random_forest.fit(x_train.values, y_train)
random_forest_preds = random_forest.predict(x_test.values)

random_forest_score = met.accuracy_score(y_test, random_forest_preds) * 100
score.append(random_forest_score)
model.append('Random Forest Classifier')

print('Random Forest Classifier Accuracy =', random_forest_score )

cf_matrix = confusion_matrix(y_test, random_forest_preds)
print('\n Condusion Matrix:',cf_matrix)

# Saving a Model for Deployment

joblib.dump(scaler, './model/min_max_ckd.pkl')

joblib.dump(random_forest, './model/random_forest_ckd.pkl')

# Feature Importance analysis

features=x_features_scaled.iloc[:, :24]
df=df.astype('string').fillna('0')
le=LabelEncoder()
for col in cat_col:
  df[col]=le.fit_transform(df[col])

ind_col=[col for col in df.columns if col!='class']
dep_col='class'

X=df[ind_col]
y=df[dep_col]

imp_features=SelectKBest(score_func=chi2,k=20)
imp_features=imp_features.fit(X,y)

datascore=pd.DataFrame(imp_features.scores_,columns=['Score'])
dfcols=pd.DataFrame(X.columns)
features_rank=pd.concat([dfcols,datascore],axis=1)
features_rank.columns=['features','score']

columns=pd.read_csv("./input/data_description.txt",sep='-')
columns=columns.reset_index(drop=True)
features_rank.features=columns.cols
ranked_features=features_rank.nlargest(24,'score')
feature_dec_order=ranked_features['features']
print('\n =============Risk factors for CKD based on the features importance (best predictors for the target variable from chi-square (χ2) test):===========\n')
print(ranked_features)
feature_dec_order.to_csv(r'./model/feature_important.csv', index=False)
