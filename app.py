import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fungsi untuk memuat dan memproses data
@st.cache_data
def load_data():
    df = pd.read_csv('(10022025) final_perfume_data.csv')
    
    # Fungsi untuk konversi string JSON ke list of notes
    def safe_convert(obj):
        if pd.isna(obj):
            return []
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['nama'])
        return L

    df['notes'] = df['notes'].apply(safe_convert)
    df['tags'] = df['notes'].apply(lambda x: " ".join(x))
    return df

# Fungsi untuk menghitung similarity matrix
@st.cache_resource
def calculate_similarity(df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

# Muat data dan similarity matrix
df = load_data()
similarity = calculate_similarity(df)

# Judul aplikasi
st.title('Sistem Rekomendasi Parfum')

# User interface untuk memilih parfum
option = st.selectbox(
    'Pilih parfum yang Anda suka:',
    df['variant'].values)

# Fungsi untuk memberikan rekomendasi
def recommend(parfum):
    parfum_index = df[df['variant'] == parfum].index[0]
    distances = similarity[parfum_index]
    parfum_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_parfums = []
    for i in parfum_list:
        recommended_parfums.append(df.iloc[i[0]])
    return recommended_parfums

# Tombol untuk mendapatkan rekomendasi
if st.button('Dapatkan Rekomendasi'):
    recommendations = recommend(option)
    st.subheader('Berikut adalah 5 rekomendasi parfum untuk Anda:')
    
    # Menampilkan rekomendasi dalam kolom
    cols = st.columns(5)
    for i, rec in enumerate(recommendations):
        with cols[i]:
            st.image(rec['image url'], width=150)
            st.markdown(f"**{rec['brand']}**")
            st.markdown(rec['variant'])
            st.markdown(f"**Notes:** {', '.join(rec['notes'])}")