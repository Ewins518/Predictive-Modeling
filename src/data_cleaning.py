import pandas as pd

def label_nonexistent_features(df):
    df = df.copy()
    if all(col in df.columns for col in ["garagearea", "garagecars"]):
        garage_zero_mask = (df["garagearea"] == 0) | (df["garagecars"] == 0)
        garage_related = ["garagetype", "garagefinish", "garagequal", "garagecond"]
        for col in garage_related:
            if col in df.columns:
                df.loc[garage_zero_mask, col] = "Nex"

    if "fireplaces" in df.columns and "fireplacequ" in df.columns:
        fire_zero_mask = df["fireplaces"] == 0
        df.loc[fire_zero_mask, "fireplacequ"] = "Nex"

    return df


def impute_and_treat_outliers(df):
    df_clean = df.copy()

    imputation_log, outlier_log = [], []

    impute_mask = df_clean['garageyrblt'].isna() & df_clean['year_constructed'].notna()

    # Fill missing values
    df_clean.loc[impute_mask, 'garageyrblt'] = df_clean.loc[impute_mask, 'year_constructed']

    imputation_log.append({
        'Column': 'garageyrblt',
        'Method': 'Direct substitution',
        'Imputed_Value': '81 values'
    })

    miss_pct = (df_clean.isna().sum() / len(df_clean))*100
    drop_cols = miss_pct[miss_pct > 40].index.tolist()
    df_clean = df_clean.drop(columns=drop_cols).copy()

    numeric_cols = df_clean.select_dtypes(include="number").columns.tolist()
    categorical_cols = df_clean.select_dtypes(exclude="number").columns.tolist()
    
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            if abs(df_clean[col].skew()) > 1:
                val, method = df_clean[col].median(), 'Median'
            else:
                val, method = df_clean[col].mean(), 'Mean'
                
            df_clean[col] = df_clean[col].fillna(val)

            imputation_log.append({'Column': col, 'Method': method, 'Imputed_Value': val})
        Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 3*IQR, Q3 + 3*IQR
        outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
        if outliers > 0:
            df_clean[col] = df_clean[col].clip(lower, upper)
            outlier_log.append({'Column': col, 'Outliers_Capped': outliers, 'Lower': lower, 'Upper': upper})

    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode(dropna=True)
            if not mode_val.empty:
                val = mode_val[0]
                method = "Mode"

            df_clean[col] = df_clean[col].fillna(val)
            imputation_log.append({"Column": col, "Method": method, "Imputed_Value": val})

    if df_clean.isnull().sum().any():
        print("Some columns still have NaNs after imputation")
        print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])

    return df_clean, pd.DataFrame(imputation_log), pd.DataFrame(outlier_log)
