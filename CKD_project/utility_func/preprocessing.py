import joblib
import pandas as pd

# Preprocessing functions for loading data frame and features. 

def create_df():
    cols = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc',
            'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    return pd.DataFrame(columns=cols, index=[0])

  
def load_scaler():
    return joblib.load('./model/minmax.pkl')


def remove_missing(df, na_dict):
    df.replace(na_dict, inplace=True)
    
    
def load_feaure():
    return joblib.load('./model/feature_imprtance_ckd.pkl')
    

def load_model():
    return joblib.load('./model/random_forest_ckd.pkl')


