import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as tf

st.title("Clustering Besaran PB vs PIC dengan KMeans")