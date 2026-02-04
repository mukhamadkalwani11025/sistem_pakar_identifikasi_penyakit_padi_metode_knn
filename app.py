import streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd
from PIL import Image

# ======================
# KONFIGURASI
# ======================
K_VALUE = 5

# ======================
# LOAD MODEL
# ======================
knn_model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

# ======================
# BASIS PENGETAHUAN
# ======================
descriptions = {
    "Bacterial leaf blight": "Penyakit hawar daun bakteri akibat Xanthomonas oryzae yang menyebabkan daun menguning dan mengering.",
    "Brown spot": "Penyakit bercak coklat akibat jamur Bipolaris oryzae yang menurunkan kualitas dan hasil panen.",
    "Leaf smut": "Penyakit gosong daun akibat jamur Entyloma oryzae yang mengganggu proses fotosintesis."
}

solutions = {
    "Bacterial leaf blight": "Gunakan varietas tahan penyakit dan aplikasikan bakterisida sesuai dosis.",
    "Brown spot": "Lakukan pemupukan seimbang dan gunakan fungisida berbahan aktif mankozeb.",
    "Leaf smut": "Gunakan benih sehat dan lakukan sanitasi lahan secara rutin."
}

# ======================
# EKSTRAKSI FITUR
# ======================
def extract_rgb_features(image):
    image = np.array(image)
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return [
        np.mean(image[:, :, 0]),
        np.mean(image[:, :, 1]),
        np.mean(image[:, :, 2])
    ]

# ======================
# STREAMLIT SETUP
# ======================
st.set_page_config(
    page_title="Sistem Pakar Penyakit Padi",
    page_icon="🌾",
    layout="wide"
)

# ## ======================
# # SIDEBAR
# # ======================
# with st.sidebar:
#     st.markdown("## Sistem Pakar")
#     st.markdown("""
#     **Metode Klasifikasi**  
#     K-Nearest Neighbor (KNN)

#     **Parameter**
#     - Nilai K : 5  
#     - Fitur : RGB Mean  
#     - Jarak : Euclidean  

#     **Teknologi**
#     - Python  
#     - Streamlit  
#     - OpenCV  
#     """)
#     st.divider()
#     st.caption("Aplikasi Sistem Pakar Berbasis Web")

# ======================
# HEADER
# ======================
st.markdown(
    """
    <h1 style='text-align:center;'>🌾 Sistem Pakar Identifikasi Penyakit Tanaman Padi</h1>
    <p style='text-align:center; font-size:17px;'>
    Menggunakan Metode <b>K-Nearest Neighbor (KNN)</b> berbasis citra daun
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ======================
# MAIN CONTENT
# ======================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Input Citra Daun")
    uploaded_file = st.file_uploader(
        "Unggah citra daun padi",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Citra Daun Padi", use_container_width=True)

with col2:
    if uploaded_file:
        st.subheader("🔍 Proses & Hasil Identifikasi")

        features = extract_rgb_features(image)
        features_scaled = scaler.transform([features])

        prediction = knn_model.predict(features_scaled)[0]
        distances, indices = knn_model.kneighbors(features_scaled)
        neighbor_labels = knn_model._y[indices[0]]

        st.success(f"🧠 **Hasil Identifikasi:** {prediction}")

        # ======================
        # TABS
        # ======================
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Detail KNN", "📖 Penjelasan", "💡 Solusi Pakar", "🔄 Alur Metode"]
        )

        # TAB 1
        with tab1:
            # ======================
            # FITUR ASLI
            # ======================
            st.markdown("#### 🎨 Fitur RGB (Mean)")
            st.table(pd.DataFrame(
                [features],
                columns=["R Mean", "G Mean", "B Mean"]
            ))

            # ======================
            # NORMALISASI
            # ======================
            st.markdown("#### 🔄 Fitur RGB Setelah Normalisasi")
            st.table(pd.DataFrame(
                features_scaled,
                columns=["R Mean (Norm)", "G Mean (Norm)", "B Mean (Norm)"]
            ))

            # ======================
            # JARAK KNN
            # ======================
            st.markdown("#### 📏 Jarak Tetangga Terdekat")
            st.table(pd.DataFrame({
                "Tetangga ke-": range(1, K_VALUE + 1),
                "Label": neighbor_labels,
                "Jarak Euclidean": distances[0]
            }))

            # ======================
            # VOTING
            # ======================
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            st.markdown("#### 🗳️ Voting KNN")
            st.table(pd.DataFrame({
                "Kelas": unique,
                "Jumlah": counts,
                "Persentase (%)": (counts / K_VALUE) * 100
            }))


        # TAB 2
        with tab2:
            st.markdown(f"### {prediction}")
            st.write(descriptions[prediction])

        # TAB 3
        with tab3:
            st.info(solutions[prediction])

        # TAB 4
        with tab4:
            st.markdown("""
            1. Input citra daun padi  
            2. Ekstraksi fitur warna RGB  
            3. Normalisasi data  
            4. Hitung jarak Euclidean  
            5. Ambil K tetangga terdekat  
            6. Voting kelas mayoritas  
            7. Menampilkan hasil & solusi pakar  
            """)
