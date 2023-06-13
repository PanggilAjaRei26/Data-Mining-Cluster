import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as tf



def main():
    st.title("Clustering Besaran PB vs PIC dengan KMeans")
    
    st.image('gambar.png', caption='Ini adalah gambar', use_column_width=True)
    
    data = [1, 2, 3, 4, 5]
    st.line_chart(data)
    
    name = st.text_input('Masukkan nama Anda')
    st.write('Halo,', name, '!')
    
    option = st.selectbox('Pilih opsi', ['A', 'B', 'C'])
    st.write('Anda memilih opsi', option)
    
if __name__ == '__main__':
    main()

