# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Judul Aplikasi
st.title("Sistem Rekomendasi Wisata Magelang")
st.markdown("Menggunakan Collaborative Filtering berbasis Matrix Factorization sederhana")

# Load Dataset Asli untuk info Place_Name
@st.cache_data
def load_original_data():
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "Dataset_Rating_Mgl.csv")
    return pd.read_csv(file_path)

df = load_original_data()

# Load Matriks Rating & Similarity yang sudah disimpan
current_dir = os.path.dirname(__file__)
rating_matrix_path = os.path.join(current_dir, "rating_matrix.pkl")
user_similarity_path = os.path.join(current_dir, "user_similarity.pkl")

rating_matrix = joblib.load(rating_matrix_path)
user_similarity_df = joblib.load(user_similarity_path)

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

# Fungsi Rekomendasi Tempat Wisata
def recommend_places(user_id, top_n=3):
    user_ratings = rating_matrix.loc[user_id]
    unrated_places = user_ratings[user_ratings.isna()].index
    predictions = []
    for place_id in unrated_places:
        pred = predict_rating(user_id, place_id)
        if pred is not None:
            predictions.append((place_id, pred))
    sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
    return sorted_preds[:top_n]

# Pilihan User
user_list = df['User_Id'].unique()
selected_user = st.selectbox("Pilih User ID:", user_list)

# Tombol Rekomendasi
if st.button("Tampilkan Rekomendasi"):
    rekomendasi = recommend_places(selected_user, top_n=3)
    if rekomendasi:
        st.subheader("Rekomendasi Tempat Wisata:")
        for place_id, score in rekomendasi:
            # Ambil nama tempat berdasarkan Place_Id
            place_name_row = df[df['Place_Id'] == place_id]
            if not place_name_row.empty:
                place_name = place_name_row['Place_Name'].values[0]
            else:
                place_name = f"Tempat dengan ID {place_id}"
            st.write(f"**{place_name}** — Prediksi Skor: `{score:.2f}`")
    else:
        st.warning("User ini sudah menilai semua tempat wisata.")

# Footer
st.markdown("---")
st.caption("Sistem Rekomendasi Wisata Magelang - Dibuat oleh kamu ❤️")
