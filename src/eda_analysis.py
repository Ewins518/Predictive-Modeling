import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def analyze_numeric_distributions(df, numeric_cols):
    numeric_cols = [col for col in numeric_cols if col.lower() != 'id']

    numeric_summary = df[numeric_cols].describe().T
    numeric_summary['skewness'] = df[numeric_cols].skew()
    numeric_summary['kurtosis'] = df[numeric_cols].kurtosis()

    # Separate discrete vs continuous
    discrete_vars = [col for col in numeric_cols if df[col].nunique() < 20]
    continuous_vars = [col for col in numeric_cols if df[col].nunique() >= 20]

    # discrete Distributions 
    if discrete_vars:
        n_cols, n_rows = 4, max(1, int(np.ceil(len(discrete_vars)/4)))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, col in enumerate(discrete_vars[:n_rows*n_cols]):
            sns.countplot(x=df[col], ax=axes[i], palette="crest", edgecolor='black')
            axes[i].set_title(f"{col} (Discrete)", fontsize=11, fontweight='bold')
            axes[i].set_xlabel("")
            axes[i].set_ylabel("Count")

        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig('Answers - Damilola Romeo Ewinsou/figures/02a_discrete_distributions.png', dpi=200)
        plt.close()

    # continuous Distributions
    if continuous_vars:
        n_cols, n_rows = 4, max(1, int(np.ceil(len(continuous_vars)/4)))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, col in enumerate(continuous_vars[:n_rows*n_cols]):
            sns.kdeplot(df[col].dropna(), fill=True, color='skyblue', ax=axes[i])
            axes[i].set_title(f"{col} (Skew={df[col].skew():.2f})", fontsize=11, fontweight='bold')
            axes[i].set_xlabel("")
            axes[i].set_ylabel("Density")

        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig('Answers - Damilola Romeo Ewinsou/figures/02b_continuous_distributions.png', dpi=200)
        plt.close()

    return numeric_summary

def plot_boxplots(df, by=None, save_path="Answers - Damilola Romeo Ewinsou/figures/03_boxplots.png"):
    
    continuous_vars = [col for col in df.select_dtypes('number').columns if df[col].nunique() >= 20]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Filter valid columns (avoid ID, non-numeric, or empty)
    continuous_vars = [col for col in continuous_vars if col.lower() != 'id' and np.issubdtype(df[col].dtype, np.number)]
    if not continuous_vars:
        print("No valid continuous variables to plot.")
        return
    
    # Limit to 4 per figure for readability
    n_cols, n_rows = 4, int(np.ceil(len(continuous_vars) / 2))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    
    for i, col in enumerate(continuous_vars[:len(axes)]):
        ax = axes[i]
        data = df[[col, by]] if by else df[[col]].copy()
        data = data.dropna()
        
        if by:
            sns.boxplot(data=data, x=by, y=col, ax=ax, palette="Set3")
            ax.set_title(f"{col} by {by}", fontsize=11, fontweight='bold')
            ax.set_xlabel(by)
            ax.set_ylabel(col)
            ax.tick_params(axis='x', rotation=45)
        else:
            sns.boxplot(data=data, y=col, ax=ax, color="lightblue")
            ax.set_title(f"{col} Distribution", fontsize=11, fontweight='bold')
            ax.set_xlabel("")
            ax.set_ylabel(col)
        
        ax.grid(alpha=0.3)

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()

    print(f"Boxplots saved to {save_path}")


def analyze_categorical_distributions(df, categorical_cols):
    categorical_summary = []

    for col in categorical_cols:  #limit to 16 for visualization
        freq = df[col].value_counts()
        categorical_summary.append({
            'Column': col,
            'Unique_Values': df[col].nunique(),
            'Most_Common': freq.index[0] if len(freq) > 0 else None,
            'Most_Common_Freq': freq.values[0] if len(freq) > 0 else 0,
            'Most_Common_Pct': (freq.values[0] / len(df) * 100) if len(freq) > 0 else 0
        })

    # Visualization
    n_cols = 4
    n_rows = int(np.ceil(min(len(categorical_cols), 16) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
    axes = axes.flatten() if len(categorical_cols) > 1 else [axes]

    for idx, col in enumerate(categorical_cols[:min(16, len(categorical_cols))]):
        if idx < len(axes):
            top_categories = df[col].value_counts().head(10)
            top_categories.plot(kind='bar', ax=axes[idx], color='teal', alpha=0.7)
            axes[idx].set_title(f'{col}\n({df[col].nunique()} unique values)', fontweight='bold')
            axes[idx].set_xlabel('')
            axes[idx].set_ylabel('Frequency')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(axis='y', alpha=0.3)

    # Remove empty subplots
    for idx in range(min(len(categorical_cols), 16), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig('Answers - Damilola Romeo Ewinsou/figures/03_categorical_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    return pd.DataFrame(categorical_summary)

def plot_correlation_matrix(df, method='pearson',
                            save_path="Answers - Damilola Romeo Ewinsou/figures/04_correlation_matrix.png"):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    continuous_vars = [col for col in numeric_cols if col.lower() != 'id' and df[col].nunique() >= 20]

    corr_matrix = df[continuous_vars].corr(method=method)

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        cbar_kws={'label': f'{method.capitalize()} correlation'},
        square=True
    )
    plt.title(f"Correlation Matrix ({method.capitalize()} Method)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()

