import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as tf

st.title("Clustering Besaran PB vs PIC dengan KMeans")

def main():
    st.title('Aplikasi Data Mining')
    
    # Menambahkan elemen teks
    st.header('Ini adalah header')
    st.subheader('Ini adalah subheader')
    st.write('Ini adalah teks biasa')
    
    # Menambahkan elemen gambar
    st.image('gambar.png', caption='Ini adalah gambar', use_column_width=True)
    
    # Menambahkan elemen plot
    data = [1, 2, 3, 4, 5]
    st.line_chart(data)
    
    # Menambahkan elemen interaktif
    name = st.text_input('Masukkan nama Anda')
    st.write('Halo,', name, '!')
    
    # Menambahkan elemen lainnya
    option = st.selectbox('Pilih opsi', ['A', 'B', 'C'])
    st.write('Anda memilih opsi', option)
    
if __name__ == '__main__':
    main()

