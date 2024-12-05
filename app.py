import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Clustering Besaran PB vs PIC dengan KMeans")

@st.cache_data
def load_data():
    df = pd.read_csv('dataset.csv')
    return df

df = load_data()
st.write(df.head())

X = df[['PB', 'PIC']]  # Only select relevant columns
st.write(X)

# Elbow Method
clusters = []
for i in range(1, 10):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=np.arange(1, 10), y=clusters, ax=ax)
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
st.pyplot(fig)

# KMeans Clustering
n_clust = 4
kmean = KMeans(n_clusters=n_clust).fit(X)
X['LABELS'] = kmean.labels_

# Visualization
fig = plt.figure(figsize=(10, 8))
scatter = plt.scatter(X['PB'], X['PIC'], c=X['LABELS'], s=50, cmap='tab10')
plt.colorbar(scatter, ticks=range(n_clust))

for label in X['LABELS'].unique():
    plt.annotate(label, 
                 (X[X['LABELS'] == label]['PB'].mean(), X[X['LABELS'] == label]['PIC'].mean()),
                 horizontalalignment='center', verticalalignment='center', 
                 size=20, weight='bold', color='black')

plt.xlabel('PB')
plt.ylabel('PIC')
plt.title('Cluster Visualization')
st.pyplot(fig)

# Prediction Section
st.subheader("Prediksi Cluster")
pb_input = st.number_input("Masukkan nilai PB:")
pic_input = st.number_input("Masukkan nilai PIC:")
prediction_button = st.button("Prediksi")

if prediction_button:
    new_data = pd.DataFrame({'PB': [pb_input], 'PIC': [pic_input]})
    prediction = kmean.predict(new_data)[0]
    st.write(f"Data dengan PB={pb_input} dan PIC={pic_input} diprediksi masuk ke cluster {prediction}")
