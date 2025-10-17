import pandas as pd, os

def save_results(df_clean, summaries, regression_results,regression_results_ft, clustering_results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for name, df in summaries.items():
            df.to_excel(writer, sheet_name=name, index=False)
        regression_results['metrics'].to_excel(writer, sheet_name='Regression_Metrics', index=False)
        regression_results['feature_importance'].to_excel(writer, sheet_name='Feature_Importance_Reg', index=False)
        regression_results_ft['metrics'].to_excel(writer, sheet_name='Regression_Metrics_ft', index=False)
        regression_results_ft['feature_importance'].to_excel(writer, sheet_name='Feature_Importance_Reg_ft', index=False)
        clustering_results['metrics'].to_excel(writer, sheet_name='Clustering_Metrics', index=False)
        clustering_results['cluster_profiles'].to_excel(writer, sheet_name='Cluster_Profiles', index=False)
        clustering_results['Clustered_Data'].to_excel(writer, sheet_name='Clustered_Data', index=False)
    print(f"Results exported to {output_path}")
