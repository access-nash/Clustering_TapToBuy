# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 20:33:34 2025

@author: avina
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Sample dataset
df_gc = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/Unsupervised ML models/TapToBuy.csv')
df_gc.columns
df_gc.dtypes
df_gc.shape
df_gc.head()
df_gc.describe()

print("\nMissing Values:\n", df_gc.isnull().sum())

# Fill missing values 
df_gc['Ever_Married'].fillna(df_gc['Ever_Married'].mode()[0], inplace=True)
df_gc['Graduated'].fillna(df_gc['Graduated'].mode()[0], inplace=True)
df_gc['Profession'].fillna(df_gc['Profession'].mode()[0], inplace=True)
df_gc['Work_Experience'].fillna(df_gc['Work_Experience'].median(), inplace=True)
df_gc['Family_Size'].fillna(df_gc['Family_Size'].median(), inplace=True)


df_gc['Ever_Married'] = df_gc['Ever_Married'].map({'No': 0, 'Yes': 1})
df_gc['Graduated'] = df_gc['Graduated'].map({'No': 0, 'Yes': 1})

# Encode remaining categorical variables
label_encoders = {}
categorical_cols = ['Gender', 'Profession', 'Spending_Score']
for col in categorical_cols:
    df_gc[col] = df_gc[col].astype(str)  # Ensure all categorical values are strings
    le = LabelEncoder()
    df_gc[col] = le.fit_transform(df_gc[col])

# Verify that there are no missing values
print("\nMissing Values After Handling:\n", df_gc.isnull().sum())

df_gc['Gender'].value_counts()
df_gc['Ever_Married'].value_counts()
df_gc['Graduated'].value_counts()
df_gc['Profession'].value_counts()
df_gc['Spending_Score'].value_counts()
df_gc['Family_Size'].value_counts()


#df_gc.fillna(0, inplace=True)

# Normalize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_gc.drop(columns=['ID']))

# PCA for dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
df_gc['PCA1'], df_gc['PCA2'] = pca_data[:, 0], pca_data[:, 1]

# Define a range of cluster counts to test
cluster_range = range(2, 10)  
wcss = []

for i in cluster_range:
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
    

# Plot the Elbow method
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, wcss, marker='o')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

from sklearn.metrics import silhouette_score
silhouette_scores = []

# Silhouette score for each cluster count
for i in cluster_range:
    kmeans = KMeans(n_clusters=i, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(scaled_data, labels)
    silhouette_scores.append(silhouette_avg)

# Print Silhouette scores
for i, score in zip(cluster_range, silhouette_scores):
    print(f'Number of Clusters: {i}, Silhouette Score: {score}')

# K-Means Clustering 
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df_gc['KMeans_Cluster'] = kmeans.fit_predict(scaled_data)


# Plot KMeans clusters
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df_gc['Age'], y=df_gc['Spending_Score'], hue=df_gc['KMeans_Cluster'], palette='viridis')
plt.title("K-Means Clustering")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x=df_gc['Work_Experience'], y=df_gc['Spending_Score'], hue=df_gc['KMeans_Cluster'], palette='viridis')
plt.title("K-Means Clustering 2")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x='PCA1', y='PCA2', hue=df_gc['KMeans_Cluster'], palette='viridis', data=df_gc)
plt.title("K-Means Clustering 3")
plt.show()

# Hierarchical Clustering 
linkage_matrix = linkage(scaled_data, method='ward')
plt.figure(figsize=(8, 5))
dendrogram(linkage_matrix)
plt.title("Dendrogram - Hierarchical Clustering")
plt.show()

# Assign clusters
hier_clusters = fcluster(linkage_matrix, t=5, criterion='maxclust')
df_gc['Hierarchical_Cluster'] = hier_clusters

# DBSCAN Clustering

from sklearn.neighbors import NearestNeighbors

# Find distances to the 16th nearest neighbor (twice the no. of features)
nn = NearestNeighbors(n_neighbors=16).fit(scaled_data)
distances, indices = nn.kneighbors(scaled_data)

# Sort the distances
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
distances

plt.figure(figsize=(10,6))
plt.plot(distances)
plt.title("k-distance Graph")
plt.xlabel("Data Points sorted by distance")
plt.ylabel("Epsilon")
plt.grid(True)
plt.show()

dbscan = DBSCAN(eps=1, min_samples=16)
df_gc['DBSCAN_Cluster'] = dbscan.fit_predict(scaled_data)

# Plot DBSCAN clusters
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df_gc['Age'], y=df_gc['Spending_Score'], hue=df_gc['DBSCAN_Cluster'], palette='coolwarm')
plt.title("DBSCAN Clustering")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x=df_gc['Work_Experience'], y=df_gc['Spending_Score'], hue=df_gc['DBSCAN_Cluster'], palette='coolwarm')
plt.title("DBSCAN Clustering 2")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x='PCA1', y='PCA2', hue=df_gc['DBSCAN_Cluster'], palette='coolwarm', data=df_gc)
plt.title("DBSCAN Clustering 3")
plt.show()


