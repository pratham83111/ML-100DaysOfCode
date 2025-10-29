# K-Means Clustering on Iris Dataset with Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df)

# Add true species for comparison
df['Actual_Species'] = iris.target

# Visualize clusters
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df.iloc[:,0], 
    y=df.iloc[:,2], 
    hue=df['Cluster'], 
    palette='viridis', 
    s=80
)
plt.scatter(
    kmeans.cluster_centers_[:,0], 
    kmeans.cluster_centers_[:,2], 
    s=250, c='red', label='Centroids'
)
plt.title("K-Means Clustering on Iris Dataset ")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()

# Cluster centers
print("Cluster Centers:\n", kmeans.cluster_centers_)

# Compare with actual species
print("\nCluster vs Actual Species:")
print(pd.crosstab(df['Cluster'], df['Actual_Species']))
