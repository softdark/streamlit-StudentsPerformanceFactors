import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Analisis Faktor Kinerja Siswa")

# Upload file CSV
data_file = st.file_uploader("Unggah file CSV", type=["csv"])

if data_file is not None:
    df = pd.read_csv(data_file)
    st.write("### Tampilan Data")
    st.dataframe(df.head())

    # Statistik deskriptif
    st.write("### Statistik Deskriptif")
    st.write(df.describe())
    
    # Pilih variabel untuk visualisasi
    st.write("### Visualisasi Data")
    col_x = st.selectbox("Pilih Variabel X", df.columns)
    col_y = st.selectbox("Pilih Variabel Y", df.columns)
    
    if col_x and col_y:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=col_x, y=col_y, ax=ax)
        st.pyplot(fig)
