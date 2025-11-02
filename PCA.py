# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import plotly.express as px

# Step 2: Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]

# Step 3: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA (4D → 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 5: Create new DataFrame for visualization
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['species'] = df['species']

# Step 6: Plot interactive PCA scatter plot
fig = px.scatter(
    pca_df,
    x='PC1', y='PC2',
    color='species',
    title='🌸 PCA on Iris Dataset (2D Projection)',
    hover_data=['species'],
    symbol='species'
)
fig.update_layout(
    template='plotly_white',
    title_font=dict(size=22, color='purple'),
    font=dict(size=14),
)
fig.show()

# Step 7: Check how much variance PCA captured
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
