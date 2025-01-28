import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

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

    # Pilih variabel target
    target = st.selectbox("Pilih Variabel Target", df.columns)
    
    if st.button("Jalankan Model Tanpa Preparation"):
        X = df.drop(columns=[target])
        y = df[target]
        
        # Mengonversi data kategorikal ke numerik jika perlu
        X = X.select_dtypes(include=["number"])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_raw = RandomForestClassifier(random_state=42)
        model_raw.fit(X_train, y_train)
        y_pred_raw = model_raw.predict(X_test)
        acc_raw = accuracy_score(y_test, y_pred_raw)
        st.write(f"Akurasi tanpa preparation: {acc_raw:.2f}")
    
    if st.button("Jalankan Model dengan Preparation"):
        X = df.drop(columns=[target])
        y = df[target]
        
        # Encoding variabel kategorikal
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        
        # Normalisasi data numerik
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_prepared = RandomForestClassifier(random_state=42)
        model_prepared.fit(X_train, y_train)
        y_pred_prepared = model_prepared.predict(X_test)
        acc_prepared = accuracy_score(y_test, y_pred_prepared)
        st.write(f"Akurasi dengan preparation: {acc_prepared:.2f}")
    
   
    if 'model_raw' in locals() and 'acc_raw' in locals():
        if 'acc_prepared' in locals() and acc_prepared > acc_raw:
            best_model = model_prepared
        else:
            best_model = model_raw

        joblib.dump(best_model, 'best_model.pkl')
        st.write("Model terbaik telah disimpan sebagai 'best_model.pkl'")
    else:
        st.write("Harap jalankan model terlebih dahulu sebelum menyimpan!")