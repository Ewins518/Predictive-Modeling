import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_cluster = pd.read_excel("../Answers - Damilola Romeo Ewinsou/Answers_RealEstate_Analysis2.xlsx", sheet_name="Clustered_Data")
df_original = pd.read_csv("../data/clean_data.csv")

df_original["Cluster"] = df_cluster["Cluster"].values

sns.set_theme(style="whitegrid", palette="muted")

plt.figure(figsize=(8,5))
sns.boxplot(data=df_original, x="Cluster", y="house_age", palette="crest")
plt.title("Distribution of House Age per Cluster", fontsize=14, weight='bold')
plt.xlabel("Cluster")
plt.ylabel("House Age (years)")
plt.tight_layout()
plt.savefig('../Answers - Damilola Romeo Ewinsou/figures/House_Age.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------
# Interpretation:
# Older clusters → redevelopment opportunities
# Newer clusters → recent constructions
# ---------------------------------------------

plt.figure(figsize=(8,5))
sns.countplot(data=df_original, x="Cluster", hue="streetname", palette="pastel")
plt.title("Dominant Road Type by Cluster", fontsize=14, weight='bold')
plt.xlabel("Cluster")
plt.ylabel("Number of Properties")
plt.legend(title="Street Type")
plt.tight_layout()
plt.savefig('../Answers - Damilola Romeo Ewinsou/figures/street_Type.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------
# Interpretation:
# Paved vs Gravel split shows infrastructure quality
# ---------------------------------------------


# compute top 8 neighborhoods by frequency
top_neigh = df_original['neighborhood'].value_counts().nlargest(8).index
plt.figure(figsize=(10,5))
sns.countplot(
    data=df_original[df_original['neighborhood'].isin(top_neigh)],
    y="neighborhood", hue="Cluster", palette="coolwarm"
)
plt.title("Top Neighborhood Distribution by Cluster", fontsize=14, weight='bold')
plt.xlabel("Number of Properties")
plt.ylabel("Neighborhood")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig('../Answers - Damilola Romeo Ewinsou/figures/Neighborhood.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------
# Interpretation:
# Reveals where clusters concentrate geographically
# ---------------------------------------------


# --- (a) Overall Quality
plt.figure(figsize=(8,5))
sns.boxplot(data=df_original, x="Cluster", y="ovl_quality", palette="viridis")
plt.title("Building Quality by Cluster", fontsize=14, weight='bold')
plt.xlabel("Cluster")
plt.ylabel("Overall Quality Rating")
plt.tight_layout()
plt.savefig('../Answers - Damilola Romeo Ewinsou/figures/Overall Quality Rating.png', dpi=300, bbox_inches='tight')
plt.close()

# --- (b) Type of Building
plt.figure(figsize=(9,5))
top_types = df_original['type_building'].value_counts().nlargest(5).index
sns.countplot(
    data=df_original[df_original['type_building'].isin(top_types)],
    x="Cluster", hue="type_building", palette="mako"
)
plt.title("🏘️ Dominant Building Types by Cluster", fontsize=14, weight='bold')
plt.xlabel("Cluster")
plt.ylabel("Number of Properties")
plt.legend(title="Building Type")
plt.tight_layout()
plt.savefig('../Answers - Damilola Romeo Ewinsou/figures/Building Type.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------
# Interpretation:
# Higher-quality ratings and more modern building
# types in newer clusters
# ---------------------------------------------

plt.figure(figsize=(8,5))
sns.boxplot(data=df_original, x="Cluster", y="saleprice", palette="flare")
plt.title("💰 Sale Price Distribution per Cluster", fontsize=14, weight='bold')
plt.xlabel("Cluster")
plt.ylabel("Sale Price (USD)")
plt.tight_layout()
plt.savefig('../Answers - Damilola Romeo Ewinsou/figures/Sale Price.png', dpi=300, bbox_inches='tight')
plt.close()
