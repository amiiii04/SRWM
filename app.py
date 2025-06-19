# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Judul Aplikasi
st.title("Sistem Rekomendasi Wisata Magelang")
st.markdown("Menggunakan Collaborative Filtering berbasis Matrix Factorization sederhana")

# Load Dataset Asli untuk info Place_Name
@st.cache_data
def load_original_data():
    return pd.read_csv("Dataset_Rating_Mgl.csv")

df = load_original_data()

# Load Matriks Rating & Similarity yang sudah disimpan
rating_matrix = joblib.load("rating_matrix.pkl")
user_similarity_df = joblib.load("user_similarity.pkl")

# Fungsi Prediksi Rating
def predict_rating(user_id, place_id):
    if place_id in rating_matrix.columns:
        sim_scores = user_similarity_df.loc[user_id]
        ratings = rating_matrix[place_id]
        mask = ratings.notna() & (ratings.index != user_id)
        relevant_sims = sim_scores[mask]
        relevant_ratings = ratings[mask]
        if relevant_sims.sum() > 0:
            return np.dot(relevant_sims, relevant_ratings) / relevant_sims.sum()
    return None

# Fungsi Rekomendasi
def recommend_places(user_id, top_n=3):
    user_ratings = rating_matrix.loc[user_id]
    unrated_places = user_ratings[user_ratings.isna()].index
    predictions = []
    for place_id in unrated_places:
        pred = predict_rating(user_id, place_id)
        if pred:
            predictions.append((place_id, pred))
    sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
    return sorted_preds[:top_n]

# Pilih User
user_list = df['User_Id'].unique()
selected_user = st.selectbox("Pilih User ID:", user_list)

# Tombol Rekomendasi
if st.button("Tampilkan Rekomendasi"):
    rekomendasi = recommend_places(selected_user, top_n=3)
    if rekomendasi:
        st.subheader("Rekomendasi Tempat Wisata:")
        for place_id, score in rekomendasi:
            place_name = df[df['Place_Id'] == place_id]['Place_Name'].values[0]
            st.write(f"**{place_name}** — Prediksi Skor: `{score:.2f}`")
    else:
        st.warning("User ini sudah menilai semua tempat wisata.")

# Footer
st.markdown("---")
st.caption("Sistem Rekomendasi Wisata Magelang - Dibuat oleh kamu")
