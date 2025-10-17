from src.data_loading import load_data
from src.eda_analysis import analyze_numeric_distributions, analyze_categorical_distributions, plot_boxplots, plot_correlation_matrix
from src.data_cleaning import impute_and_treat_outliers, label_nonexistent_features
from src.feature_selection import compute_feature_importance
from src.regression_model import build_regression_model
from src.clustering_model import build_clustering_model, generate_cluster_profile
from src.export_results import save_results
from src.feature_engineering import feature_engineering

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

DATA_PATH = "data/Take-home-case-Data-analyst(data).csv"


def main():
    df = load_data(DATA_PATH)

    # Datatypes summary
    datatypes_summary = df.dtypes.reset_index()
    datatypes_summary.columns = ["Column", "Dtype"]

    # missing value summary
    missing_summary = (
        df.isnull().sum().reset_index().rename(columns={"index": "Column", 0: "Missing_Count"})
    )
    missing_summary["Missing_%"] = (missing_summary["Missing_Count"] / len(df)) * 100
    missing_summary = missing_summary[missing_summary["Missing_Count"] > 0].sort_values("Missing_%", ascending=False)

    # numeric / categorical columns
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    # EDA
    numeric_summary = analyze_numeric_distributions(df, num_cols)
    categorical_summary = analyze_categorical_distributions(df, cat_cols)
    plot_boxplots(df)

    # data cleaning
    df = label_nonexistent_features(df)
    df_clean, imputation_log, outlier_log = impute_and_treat_outliers(df)
    print("null ???", sum(df_clean.isnull().sum()))
    print("shape ", df_clean.shape)

    plot_correlation_matrix(df_clean)

    #feature engineering
    df_clean = feature_engineering(df_clean)
    print("shape 0", df_clean.shape)

    df_clean.to_csv("data/clean_data.csv", index=False)

    target = "saleprice"
    feature_cols = [c for c in df_clean.columns if c != target and c != 'id' ]

    top_features, feat_imp, df_encoded = compute_feature_importance(df_clean, target, feature_cols, n_top=10)

    feature_cols = [c for c in df_encoded.columns if c != target and c != 'id' ]

    # Regression    
    models = [
    ("Lasso", Lasso()),
    ("RandomForest", RandomForestRegressor(random_state=42)),
    ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
    ("HistGradientBoosting", HistGradientBoostingRegressor(random_state=42)),
    ]

    regression_results, all_feature_importances = [], []
#
    for model_name, model_class in models:
        print(f"\n Training {model_name} model...")
        best_model, metrics = build_regression_model(
            df_encoded,
            df_clean,
            target_col=target,
            feature_cols=feature_cols,
            top_features = top_features,
            model=model_class,
            model_name = model_name,
        )
        metrics["Model"] = model_name
        regression_results.append(metrics)
        all_feature_importances.append(feat_imp)
#
    regression_summary = pd.concat(regression_results, ignore_index=True)
    feature_importance_summary = pd.concat(all_feature_importances, ignore_index=True)
#
   # Pivot regression metrics into one comparison table
    regression_pivot = regression_summary.pivot(index="Metric", columns="Model", values="Test").reset_index()
#
    print(regression_pivot)
    #Parameters finetuning

    lasso_grid = {"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0], "max_iter": [10000], "tol": [1e-3]}
    rf_grid = {"n_estimators": [100, 200], "max_depth": [8, 12, None]}
    hgb_grid = {
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [6, 10],
        "max_iter": [300, 400],
    }
    gb_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 7, 10],
        'learning_rate': [0.01, 0.1],
        'min_samples_split': [2, 5]
    } 

    models_to_run = [
        ("Lasso_FT", Lasso(random_state=42), lasso_grid),
        ("RandomForest_FT", RandomForestRegressor(random_state=42), rf_grid),
        ("HistGradientBoosting_FT", HistGradientBoostingRegressor(random_state=42), hgb_grid),
        ("GradientBoosting_FT", GradientBoostingRegressor(random_state=42), gb_grid),
    ]

    regression_results_ft, all_feature_importances_ft = [], []

    for model_name, model_class, grid in models_to_run:
        print(f"\n Training {model_name} model...")
        best_model, metrics = build_regression_model(
            df_encoded,
            df_clean,
            target_col=target,
            feature_cols=feature_cols,
            top_features = top_features,
            model=model_class,
            model_name = model_name,
            param_grid=grid,
        )
        metrics["Model"] = model_name
        regression_results_ft.append(metrics)
        all_feature_importances_ft.append(feat_imp)

    regression_summary_ft = pd.concat(regression_results_ft, ignore_index=True)
    feature_importance_summary_ft = pd.concat(all_feature_importances_ft, ignore_index=True)

   # Pivot regression metrics into one comparison table
    regression_pivot_ft = regression_summary_ft.pivot(index="Metric", columns="Model", values="Test").reset_index()

    print(regression_pivot_ft)

   # Clustering Model
    feature_cols_cluster = [c for c in num_cols if c in df_clean.columns]
    df_clustered = build_clustering_model(df_clean,feature_cols_cluster)

    cluster_profiles = generate_cluster_profile(
    df_original=df_clean,
    df_clustered=df_clustered['df_clustered'],
    cluster_col='Cluster',
    price_col='saleprice'
)
   #df_clustered_kp = build_clustering_model_kprototypes(df_clean)
#

    save_results(
       df_clean,
       {
           "1_DataTypes": datatypes_summary,
           "2_MissingValues": missing_summary,
           "3_Numeric_Summary": numeric_summary,
           "4_Categorical_Summary": categorical_summary,
           "5_Imputation_Log": imputation_log,
           "6_Outlier_Log": outlier_log,
       },
       {
           "metrics": regression_pivot,
           "feature_importance": feature_importance_summary,
       },
       {
           "metrics": regression_pivot_ft,
           "feature_importance": feature_importance_summary_ft,
       },
       {
           "metrics": df_clustered["metrics"],
           "cluster_profiles": cluster_profiles,
           "Clustered_Data": df_clustered['df_clustered']
       },
       "Answers - Damilola Romeo Ewinsou/Answers_RealEstate_Analysis2.xlsx",
   )

if __name__ == "__main__":
   main()
