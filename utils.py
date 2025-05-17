import pandas as pd
import numpy as np
from sklearn import preprocessing

def preprocess(df):
    print(df.describe())
    print('----------------------------------------------')
    print("Before preprocessing")
    print("Number of rows with 0 values for each variable")
    for col in df.columns:
        missing_rows = df.loc[df[col] == 0].shape[0] if df[col].dtype != 'object' else 0
        print(f"{col}: {missing_rows}")
    print('----------------------------------------------')

    # Standardize column names
    df.columns = [col.strip().upper().replace(" ", "_") for col in df.columns]

    # Encode Gender and Target
    df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

    # Replace 0 values with the mean of the column for all columns except GENDER and LUNG_CANCER
    for col in df.columns:
        if col not in ['GENDER', 'LUNG_CANCER']:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].mean())
    print('----------------------------------------------')
    print("After preprocessing")
    print("Number of rows with 0 values for each variable")
    for col in df.columns:
        missing_rows = df.loc[df[col] == 0].shape[0] if df[col].dtype != 'object' else 0
        print(f"{col}: {missing_rows}")
    print('----------------------------------------------')



    # Standardization (excluding the target)
    feature_cols = [col for col in df.columns if col != 'LUNG_CANCER']
    df_scaled = preprocessing.scale(df[feature_cols])
    df_scaled = pd.DataFrame(df_scaled, columns=feature_cols)
    df_scaled['LUNG_CANCER'] = df['LUNG_CANCER'].values
    print(df_scaled.describe())

    return df_scaled
