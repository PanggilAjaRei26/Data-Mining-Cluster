import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as tf

st.title("Clustering Besaran PB vs PIC dengan KMeans")

model = pickle.load(open('cluster.sav', 'rb'))

df = pd.read_csv("dataset.csv")
df['PB'] = pd.to_datetime(df['PIC'])
df.set_index(['PB'], inplace=True)

st.title('Prediksi Cluster')
year = st.slider("Tentukan Tahun",1,5, step=1)

pred = model.forecast(year)
pred = pd.DataFrame(pred, index=pd.date_range(start=df.index[-1], periods=year, freq='Y'), columns=['PB'])

if st.button("Prediksi"):
    col1, col2 = st.columns([2,3])
    with col1:
        st.dataframe(pred)
    with col2:
        fig, ax = plt.subplots()
        df['PB'].plot(style="--", color='gray', legend=True, label='known')
        pred['PB'].plot(color='b', legend=True, label='Prediction')
        st.pyplot(fig)