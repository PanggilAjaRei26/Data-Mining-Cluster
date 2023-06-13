import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as tf
import seaborn as sns

st.title("Clustering Besaran PB vs PIC dengan KMeans")

df = pd.read_csv('dataset.csv')

df.head()

X = df.drop(['TOKO', 'NAMA', 'BACKUP', 'URUT', 'JARAK (KM)', 'NOPIC', 'TGLPICK', 'DOL', 'KLIK', 'ZN', 'NOSJ', 'NOPB', 'CNT', 'BRJ'], axis=1)

print(X)

X.info()

clusters = []
for i in range(1, 10):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=np.arange(1, 10), y=clusters, ax=ax)
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.show()

n_clust = 4
kmean = KMeans(n_clusters=n_clust).fit(X)
X['LABELS'] = kmean.labels_

plt.figure(figsize=(10, 8))
plt.scatter(X['PB'], X['PIC'], c=X['LABELS'], marker='o', s=50, cmap='tab10')

for label in X['LABELS'].unique():
    plt.annotate(label,
                (X[X['LABELS'] == label]['PB'].mean(),
                X[X['LABELS'] == label]['PIC'].mean()),
                horizontalalignment='center',
                verticalalignment='center',
                size=20, weight='bold',
                color='black')

plt.xlabel('PB')
plt.ylabel('PIC')
plt.title('Cluster Visualization')
plt.colorbar(ticks=range(n_clust))
plt.show()

print(X)