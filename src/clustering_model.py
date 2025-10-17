from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score, silhouette_score

def build_clustering_model(df,feature_cols_cluster):

    X = df[feature_cols_cluster].copy()
    X = X.drop(columns=['id', 'saleprice'])

    print(f"Dataset size: {X.shape[0]} properties")
    print(f"Features: {X.shape[1]} variables")

    # dimensionality reduction with PCA 

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA to reduce dimensionality while retaining 95% variance
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print(f"\n Original dimensions: {X_scaled.shape[1]}")
    print(f" Reduced dimensions: {X_pca.shape[1]}")
    print(f" Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    # Visualize variance explained
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.xlabel('Number of Components', fontsize=12)
    plt.ylabel('Cumulative Explained Variance', fontsize=12)
    plt.title('PCA - Cumulative Variance Explained', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(range(1, min(11, len(pca.explained_variance_ratio_) + 1)),
            pca.explained_variance_ratio_[:10], color='teal', alpha=0.7)
    plt.xlabel('Principal Component', fontsize=12)
    plt.ylabel('Variance Explained', fontsize=12)
    plt.title('Top 10 Components - Individual Variance', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('Answers - Damilola Romeo Ewinsou/figures/pca_dimensionality_reduction.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("DETERMINING OPTIMAL NUMBER OF CLUSTERS")

    # Elbow Method and Silhouette Analysis
    inertias = []
    silhouette_scores = []
    davies_bouldin_scores = []
    K_range = range(2, 11)

    print("\nEvaluating K=2 to K=10...")
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(X_pca)

        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))
        davies_bouldin_scores.append(davies_bouldin_score(X_pca, kmeans.labels_))

        print(f"K={k}: Silhouette={silhouette_scores[-1]:.4f}, "
              f"Davies-Bouldin={davies_bouldin_scores[-1]:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Elbow curve
    axes[0].plot(K_range, inertias, 'bo-', linewidth=2)
    axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # Silhouette score (higher is better)
    axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2)
    axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Score (Higher = Better)', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)

    # Davies-Bouldin score (lower is better)
    axes[2].plot(K_range, davies_bouldin_scores, 'ro-', linewidth=2)
    axes[2].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[2].set_ylabel('Davies-Bouldin Score', fontsize=12)
    axes[2].set_title('Davies-Bouldin Score (Lower = Better)', fontsize=14, fontweight='bold')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('Answers - Damilola Romeo Ewinsou/figures/optimal_clusters_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Select optimal K (highest silhouette score)
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"\n OPTIMAL NUMBER OF CLUSTERS: {optimal_k}")
    print(f"Based on highest Silhouette Score: {max(silhouette_scores):.4f}")

    print("TRAINING FINAL CLUSTERING MODEL")

    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20, max_iter=500)
    cluster_labels = final_kmeans.fit_predict(X_pca)

    df_clustered = X.copy()
    df_clustered['Cluster'] = cluster_labels

    print("CLUSTER PROFILING")

    cluster_profiles = []

    for cluster_id in range(optimal_k):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]

        profile = {
            'Cluster': f'Cluster {cluster_id}',
            'Size': len(cluster_data),
            'Size_Percentage': (len(cluster_data) / len(df_clustered)) * 100
        }

        # Add mean of key numeric features
        numeric_features = cluster_data.select_dtypes(include=[np.number]).columns
        numeric_features = [f for f in numeric_features if f != 'Cluster']

        for feat in numeric_features[:10]:  # Top 10 features
            if feat in cluster_data.columns:
                profile[f'Avg_{feat}'] = cluster_data[feat].mean()

        cluster_profiles.append(profile)

        print(f"\n{profile['Cluster']}:")
        print(f"  Size: {profile['Size']} properties ({profile['Size_Percentage']:.1f}%)")

    cluster_profile_df = pd.DataFrame(cluster_profiles)
    print("\n" + cluster_profile_df.to_string(index=False))

    # Visualize clusters in 2D (first 2 principal components)
    plt.figure(figsize=(14, 8))

    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels,
                         cmap='viridis', alpha=0.6, s=50, edgecolors='k', linewidth=0.5)

    # Plot cluster centers
    centers_2d = final_kmeans.cluster_centers_[:, :2]
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X',
                s=300, edgecolors='black', linewidth=2, label='Centroids')

    plt.xlabel('First Principal Component', fontsize=12)
    plt.ylabel('Second Principal Component', fontsize=12)
    plt.title(f'Property Portfolio Segmentation - {optimal_k} Clusters',
              fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Answers - Damilola Romeo Ewinsou/figures/cluster_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Cluster size distribution
    plt.figure(figsize=(10, 6))
    cluster_sizes = df_clustered['Cluster'].value_counts().sort_index()
    plt.bar(cluster_sizes.index, cluster_sizes.values, color='teal', alpha=0.7, edgecolor='black')
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Number of Properties', fontsize=12)
    plt.title('Cluster Distribution - Portfolio Composition', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    for i, v in enumerate(cluster_sizes.values):
        plt.text(i, v + 10, str(v), ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('Answers - Damilola Romeo Ewinsou/figures/cluster_size_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("CLUSTERING MODEL PERFORMANCE METRICS")

    silhouette_avg = silhouette_score(X_pca, cluster_labels)
    davies_bouldin = davies_bouldin_score(X_pca, cluster_labels)
    inertia = final_kmeans.inertia_

    metrics_df = pd.DataFrame({
        'Metric': ['Silhouette Score', 'Davies-Bouldin Index', 'Inertia', 'Number of Clusters'],
        'Value': [silhouette_avg, davies_bouldin, inertia, optimal_k],
        'Interpretation': [
            'Higher is better',
            'Lower is better',
            'Lower is better',
            'Optimal based on Silhouette'
        ]
    })

    print("\n" + metrics_df.to_string(index=False))

    return {
        'model': final_kmeans,
        'pca': pca,
        'scaler': scaler,
        'optimal_k': optimal_k,
        'cluster_labels': cluster_labels,
        'metrics': metrics_df,
        'cluster_profiles': cluster_profile_df,
        'silhouette_scores': silhouette_scores,
        'df_clustered': df_clustered
    }

def generate_cluster_profile(df_original, df_clustered, cluster_col='Cluster', price_col='saleprice', top_n_features=10):
    
    df_merged = df_original.copy()
    df_merged[cluster_col] = df_clustered[cluster_col].values

    clusters = sorted(df_merged[cluster_col].unique())
    profiles = []

    for cluster_id in clusters:
        cluster_data = df_merged[df_merged[cluster_col] == cluster_id]
        profile = {
            'Cluster': f'Cluster {cluster_id}',
            'Size': len(cluster_data),
            'Size_%': round((len(cluster_data) / len(df_merged)) * 100, 2)
        }

        if price_col in cluster_data.columns:
            profile['SalePrice_Mean'] = round(cluster_data[price_col].mean(), 0)
            profile['SalePrice_Median'] = round(cluster_data[price_col].median(), 0)

        # add averages of top numerical features 
        num_features = cluster_data.select_dtypes(include=[np.number]).columns
        num_features = [f for f in num_features if f not in [price_col, cluster_col] and 'id' not in f.lower()]

        for feat in num_features[:top_n_features]:
            profile[f'Avg_{feat}'] = round(cluster_data[feat].mean(), 2)

        profiles.append(profile)

    profile_df = pd.DataFrame(profiles)

    summary_cols = ['Cluster', 'Size', 'Size_%']
    if price_col in df_original.columns:
        summary_cols += ['SalePrice_Mean', 'SalePrice_Median']
    avg_cols = [c for c in profile_df.columns if c.startswith('Avg_')]
    profile_df = profile_df[summary_cols + avg_cols]

    print(profile_df.to_string(index=False))

    return profile_df
