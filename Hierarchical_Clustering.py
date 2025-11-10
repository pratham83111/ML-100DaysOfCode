# 🧠 Hierarchical Clustering on Inbuilt Iris Dataset
# 👨‍💻 Author: Pratham Kumar

# Step 1️⃣: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Step 2️⃣: Load Inbuilt Iris Dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
print("📊 Dataset Preview:")
print(data.head())

# Step 3️⃣: Standardize Data for Better Clustering
scaler = StandardScaler()
X = scaler.fit_transform(data)

# Step 4️⃣: Create Dendrogram
plt.figure(figsize=(10, 6))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("📈 Dendrogram to Find Optimal Number of Clusters", fontsize=14, fontweight='bold')
plt.xlabel("Samples")
plt.ylabel("Euclidean Distance")
plt.show()

# Step 5️⃣: Apply Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Step 6️⃣: Add Cluster Info to Data
data['Cluster'] = y_hc + 1  # start clusters from 1 instead of 0

# Step 7️⃣: Visualize Clusters (2D)
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
palette = sns.color_palette("Set2", 3)

sns.scatterplot(
    x=data["sepal length (cm)"],
    y=data["petal length (cm)"],
    hue=data["Cluster"],
    palette=palette,
    s=100,
    edgecolor="black"
)

# Add cluster labels
for cluster_num in range(1, 4):
    cluster_data = data[data['Cluster'] == cluster_num]
    cx = cluster_data["sepal length (cm)"].mean()
    cy = cluster_data["petal length (cm)"].mean()
    plt.text(cx, cy, f'Cluster {cluster_num}',
             fontsize=12, weight='bold', color='black',
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

plt.title("🌸 Hierarchical Clustering on Iris Dataset", fontsize=15, fontweight='bold')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Cluster")
plt.show()

# Step 8️⃣: Show Cluster Summary
print("\n✅ Clustered Data Sample:")
print(data.head())

print("\n📊 Cluster Summary (Mean Values):")
print(data.groupby('Cluster').mean())
e 