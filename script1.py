import numpy as np
import pandas as pd

def clean_dataset(df):
    #Creating a copy of the dataset as to not alter the orignal data
    df_clean = df.copy()
    # Handling missing values
    def handle_missing_vals(df):
        num_columns = df.select_dtypes(include = [np.number]).columns
        for col in num_columns:
            df[col] = df[col].fillna(df[col].mean())


        cat_columns = df.select_dtypes(include = ['object']).columns
        for ocl in cat_columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

        return df
    
    df_clean = df_clean.drop_duplicates()

    def handle_outliers(df, columns):
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5*IQR
            upper_bound = Q3 + 1.5*IQR
            df[col] = df[col].clip(lower = lower_bound, upper = upper_bound)
        return df
    
    def clean_text_data(df):
        text_columns = df.select_dtypes(include = ['object']).columns
        for col in text_columns:
            df[col] = df[col].str.lower() if hasattr(df[col], 'str') else df[col]
            df[col] = df[col].str.strip() if hasattr(df[col], 'str') else df[col]
        return df
    
    def convert_datatypes(df):
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
        return df
    
    df_clean = handle_missing_vals(df_clean)
    numeric_columns = df_clean.select_dtypes(include =[np.number]).columns
    df_clean = handle_outliers(df_clean, numeric_columns)
    df_clean = clean_text_data(df_clean)
    df_clean = convert_datatypes(df_clean)

    cleaning_report = {
        'original rows' : len(df),
        'cleaned rows' : len(df_clean),
        'duplicates removed' : len(df) - len(df_clean),
        'missing values_removed' : df_clean.isnull().sum().sum()
    }

    