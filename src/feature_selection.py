import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from src.feature_engineering import encode_categorical_features


def compute_feature_importance(df, target_col, feature_cols, n_top=10):
    """
    Compute feature importances using a RandomForest baseline model.
    Returns top N features and the full importance DataFrame.
    """
    X, y = df[feature_cols].copy(), df[target_col].copy()

    df_encoded, encoders = encode_categorical_features(X)

    print("shape 1", df_encoded.shape)

    X_train, _, y_train, _ = train_test_split(df_encoded, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y_train)

    feat_imp = pd.DataFrame({
        "Feature": df_encoded.columns,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)

    top_features = feat_imp.head(n_top)["Feature"].tolist()

    # Feature Importance Visualization
    top_10_importance = feat_imp.head(10)
    plt.figure(figsize=(12, 6))
    plt.barh(top_10_importance['Feature'], top_10_importance['Importance'], color='steelblue')
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Top 10 Feature Importance - Price Prediction Model', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('Answers - Damilola Romeo Ewinsou/figures/feature_importance_regression.png', dpi=300, bbox_inches='tight')
    plt.close()

    return top_features, feat_imp, df_encoded
