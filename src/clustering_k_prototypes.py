from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def build_clustering_model_kprototypes(df):
    """
    K-Prototypes clustering - proper handling of mixed data types
    """
    print("="*80)
    print("CLUSTERING MODEL: K-PROTOTYPES (Mixed Data Type)")
    print("="*80)
    
    X = df.copy()
    
    # Identify feature types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n✓ Numeric features: {len(numeric_cols)}")
    print(f"✓ Categorical features: {len(categorical_cols)}")
    print(f"✓ Categorical ratio: {len(categorical_cols)/(len(numeric_cols)+len(categorical_cols))*100:.1f}%")
    
    if len(categorical_cols) > len(numeric_cols):
        print("⚠️  Categorical features dominate - K-Prototypes is recommended!")
    
    # Prepare data
    # 1. Scale numeric features
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # 2. Get categorical column indices (K-Prototypes needs indices)
    categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]
    
    # 3. Convert to numpy array (K-Prototypes requirement)
    X_array = X.values
    
    print(f"\nDataset shape: {X_array.shape}")
    print(f"Categorical column indices: {categorical_indices[:5]}... ({len(categorical_indices)} total)")
    
    # OPTIMAL NUMBER OF CLUSTERS
    print("\n" + "="*80)
    print("DETERMINING OPTIMAL NUMBER OF CLUSTERS")
    print("="*80)
    
    costs = []  # K-Prototypes uses "cost" instead of "inertia"
    K_range = range(2, 11)
    
    print("\nEvaluating K=2 to K=10...")
    print("Note: K-Prototypes may take longer than K-Means\n")
    
    for k in K_range:
        kproto = KPrototypes(n_clusters=k, init='Huang', n_init=5, random_state=42, verbose=0)
        clusters = kproto.fit_predict(X_array, categorical=categorical_indices)
        
        costs.append(kproto.cost_)
        
        print(f"K={k}: Cost={kproto.cost_:.2f}")
    
    # Visualize elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, costs, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Cost (Lower = Better)', fontsize=12)
    plt.title('K-Prototypes: Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Answers - Damilola Romeo Ewinsou/figures/kprototypes_elbow.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Select optimal K (elbow point - can be visual inspection or calculation)
    # Using cost reduction rate
    cost_reductions = np.diff(costs)
    reduction_rate = np.diff(cost_reductions)
    
    # Find elbow (maximum second derivative)
    if len(reduction_rate) > 0:
        optimal_k = K_range[np.argmax(reduction_rate) + 2]  # +2 due to double diff
    else:
        optimal_k = K_range[len(K_range)//2]  # Fallback to middle
    
    # Or use a heuristic (e.g., where cost reduction < threshold)
    threshold_reduction = 0.1  # 10% of initial cost
    initial_cost = costs[0]
    for i, cost in enumerate(costs):
        if i > 0 and (costs[i-1] - cost) / initial_cost < threshold_reduction:
            optimal_k = K_range[i]
            break
    
    print(f"\n✓ OPTIMAL NUMBER OF CLUSTERS: {optimal_k}")
    print(f"  Based on elbow method")
    
    # TRAIN FINAL MODEL
    print("\n" + "="*80)
    print("TRAINING FINAL K-PROTOTYPES MODEL")
    print("="*80)
    
    final_kproto = KPrototypes(n_clusters=optimal_k, init='Huang', n_init=10, 
                               random_state=42, verbose=1)
    cluster_labels = final_kproto.fit_predict(X_array, categorical=categorical_indices)
    
    print(f"\nFinal model cost: {final_kproto.cost_:.2f}")
    print(f"Number of iterations: {final_kproto.n_iter_}")
    
    # Add clusters to dataframe
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels
    
    # CLUSTER PROFILING
    print("\n" + "="*80)
    print("CLUSTER PROFILING - BUSINESS INSIGHTS")
    print("="*80)
    
    cluster_profiles = []
    
    for cluster_id in range(optimal_k):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
        
        profile = {
            'Cluster': f'Cluster {cluster_id}',
            'Size': len(cluster_data),
            'Size_Percentage': (len(cluster_data) / len(df_clustered)) * 100
        }
        
        print(f"\n{'='*60}")
        print(f"CLUSTER {cluster_id} ({len(cluster_data)} properties - {profile['Size_Percentage']:.1f}%)")
        print(f"{'='*60}")
        
        # NUMERIC FEATURES: Calculate means
        print("\nNumeric Characteristics:")
        for feat in numeric_cols[:10]:
            if feat in cluster_data.columns:
                avg_val = cluster_data[feat].mean()
                profile[f'Avg_{feat}'] = avg_val
                if 'price' in feat.lower():
                    print(f"  • {feat}: ${avg_val:,.2f}")
                else:
                    print(f"  • {feat}: {avg_val:.2f}")
        
        # CATEGORICAL FEATURES: Show most common (mode)
        print("\nCategorical Characteristics:")
        for feat in categorical_cols[:10]:
            if feat in cluster_data.columns:
                mode_val = cluster_data[feat].mode()
                if len(mode_val) > 0:
                    mode_str = str(mode_val[0])
                    mode_count = (cluster_data[feat] == mode_val[0]).sum()
                    mode_pct = (mode_count / len(cluster_data)) * 100
                    profile[f'Mode_{feat}'] = mode_str
                    profile[f'Mode_{feat}_Pct'] = mode_pct
                    print(f"  • Most Common {feat}: '{mode_str}' ({mode_pct:.1f}%)")
        
        cluster_profiles.append(profile)
    
    cluster_profile_df = pd.DataFrame(cluster_profiles)
    
    # VISUALIZATION (use PCA for plotting only)
    print("\n" + "="*80)
    print("VISUALIZATION (PCA projection for display only)")
    print("="*80)
    
    
    # For visualization: Use only numeric features or encode categoricals
    X_viz = df[numeric_cols].fillna(df[numeric_cols].median())
    
    if X_viz.shape[1] >= 2:
        pca_viz = PCA(n_components=2, random_state=42)
        X_pca_viz = pca_viz.fit_transform(StandardScaler().fit_transform(X_viz))
        
        plt.figure(figsize=(14, 8))
        scatter = plt.scatter(X_pca_viz[:, 0], X_pca_viz[:, 1], c=cluster_labels,
                             cmap='viridis', alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        
        plt.xlabel('First Principal Component (Numeric Features)', fontsize=12)
        plt.ylabel('Second Principal Component (Numeric Features)', fontsize=12)
        plt.title(f'K-Prototypes Clustering - {optimal_k} Clusters\n(PCA visualization of numeric features only)',
                  fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('Answers - Damilola Romeo Ewinsou/figures/kprototypes_clusters.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Cluster size distribution
    plt.figure(figsize=(10, 6))
    cluster_sizes = df_clustered['Cluster'].value_counts().sort_index()
    plt.bar(cluster_sizes.index, cluster_sizes.values, color='teal', alpha=0.7, edgecolor='black')
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Number of Properties', fontsize=12)
    plt.title('K-Prototypes: Cluster Distribution', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(cluster_sizes.values):
        plt.text(i, v + 10, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Answers - Damilola Romeo Ewinsou/figures/kprototypes_distribution.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # METRICS
    print("\n" + "="*80)
    print("MODEL PERFORMANCE METRICS")
    print("="*80)
    
    metrics_df = pd.DataFrame({
        'Metric': ['Final Cost', 'Number of Clusters', 'Iterations to Converge'],
        'Value': [final_kproto.cost_, optimal_k, final_kproto.n_iter_],
        'Interpretation': [
            'Lower is better',
            'Optimal based on elbow',
            'Convergence speed'
        ]
    })
    
    print("\n" + metrics_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("K-PROTOTYPES ADVANTAGES")
    print("="*80)
    print(f"""
    ✅ PROPER HANDLING OF CATEGORICAL DATA:
    • Uses Hamming distance for categorical features
    • No artificial ordering imposed on nominal categories
    • Each categorical value treated equally
    
    ✅ INTERPRETABLE CLUSTER CENTERS:
    • Numeric features: Centroids (means)
    • Categorical features: Modes (most common values)
    • Business stakeholders can understand "typical" properties
    
    ✅ OPTIMAL FOR YOUR DATA:
    • {len(categorical_cols)} categorical features ({len(categorical_cols)/(len(categorical_cols)+len(numeric_cols))*100:.1f}%)
    • Mixed data type handling is theoretically sound
    • No information loss from encoding
    
    BUSINESS IMPACT:
    • More accurate segmentation of diverse property types
    • Clear categorical characteristics per segment
    • Better targeted marketing strategies
    """)
    
    return {
        'model': final_kproto,
        'scaler': scaler,
        'optimal_k': optimal_k,
        'cluster_labels': cluster_labels,
        'metrics': metrics_df,
        'cluster_profiles': cluster_profile_df,
        'df_clustered': df_clustered,
        'categorical_indices': categorical_indices,
        'categorical_indices': categorical_indices,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'costs': costs
    }