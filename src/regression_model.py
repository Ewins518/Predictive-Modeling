from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.feature_engineering import encode_categorical_features



def build_regression_model(df, df_clean, target_col, feature_cols,top_features, model,model_name, param_grid=None):
    X, y = df[feature_cols].copy(), df_clean[target_col].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()    

    X_train_s, X_test_s = X_train[top_features], X_test[top_features]
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train_s), scaler.transform(X_test_s)

    base = model

    if param_grid is not None:
        grid = GridSearchCV(base, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        test_r2 = r2_score(y_test,y_pred)

    else:
        base.fit(X_train_scaled, y_train)
        y_pred = base.predict(X_test_scaled)
        test_r2 = r2_score(y_test,y_pred) 
        best_model = base  

    metrics = pd.DataFrame({'Metric':['R²','RMSE','MAE','MAPE(%)'],'Test':[r2_score(y_test,y_pred), mean_squared_error(y_test,y_pred,squared=False), mean_absolute_error(y_test,y_pred), float(np.mean(np.abs((y_test-y_pred)/y_test))*100)]})

    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.5, color='steelblue')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Price', fontsize=12)
    axes[0].set_ylabel('Predicted Price', fontsize=12)
    axes[0].set_title(f'{model_name}: Actual vs Predicted (R² = {test_r2:.4f})', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Residuals
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, color='coral')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Price', fontsize=12)
    axes[1].set_ylabel('Residuals', fontsize=12)
    axes[1].set_title(f'{model_name}: Residual Plot - Error Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'Answers - Damilola Romeo Ewinsou/figures/{model_name}_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    return best_model, metrics
