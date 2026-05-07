import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =============================================
# KONFIGURASI
# =============================================
st.set_page_config(
    page_title="Prediksi Harga Rumah",
    page_icon="🏠",
    layout="wide",
)

# =============================================
# CUSTOM UNPICKLER
# Bypass Orange Qt dependencies, extract sklearn model langsung
# =============================================
class ContinuousDistribution(np.ndarray):
    """Dummy untuk Orange.statistics.distribution.Continuous (subclass ndarray)"""
    def __new__(cls, *args, **kwargs):
        if args and isinstance(args[0], np.ndarray):
            return args[0].view(cls)
        return np.array([]).view(cls)
    def __setstate__(self, state):
        if isinstance(state, tuple):
            try:
                np.ndarray.__setstate__(self, state[:5] if len(state) > 5 else state)
            except Exception:
                pass
            if len(state) > 5 and isinstance(state[5], dict):
                self.__dict__.update(state[5])
        elif isinstance(state, dict):
            self.__dict__.update(state)

class OrangeBypassUnpickler(pickle.Unpickler):
    """Load pickle Orange tanpa install Orange3/Qt"""
    SPECIAL = {
        ("Orange.statistics.distribution", "Continuous"): ContinuousDistribution,
    }
    def find_class(self, module, name):
        if (module, name) in self.SPECIAL:
            return self.SPECIAL[(module, name)]
        if module.startswith("Orange"):
            return type(name, (), {
                "__init__": lambda self, *a, **k: None,
                "__setstate__": lambda self, state: (
                    self.__dict__.update(state) if isinstance(state, dict) else None
                ),
            })
        return super().find_class(module, name)

# =============================================
# LOAD MODEL
# =============================================
@st.cache_resource
def load_model():
    with open("models/house_model.pkcls", "rb") as f:
        orange_model = OrangeBypassUnpickler(f).load()
    return orange_model.skl_model

skl_model = load_model()

# =============================================
# HEADER
# =============================================
st.title("🏠 Prediksi Harga Rumah per Unit Area")
st.markdown(
    "Aplikasi ini memprediksi **harga rumah per unit area** "
    "menggunakan model **AdaBoost Regressor** dari Orange Data Mining."
)
st.markdown("---")

# =============================================
# SIDEBAR - INPUT
# =============================================
st.sidebar.header("📝 Input Data Properti")

st.sidebar.subheader("📅 Waktu Transaksi")
transaction_date = st.sidebar.slider(
    "Tahun Transaksi",
    min_value=2012.0, max_value=2014.0,
    value=2013.0, step=0.083,
    help="Format desimal, misal 2013.5 = pertengahan 2013"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🏗️ Kondisi Properti")
house_age = st.sidebar.slider(
    "Usia Bangunan (tahun)",
    min_value=0.0, max_value=50.0, value=15.0, step=0.5
)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Lokasi & Fasilitas")
distance_to_mrt = st.sidebar.number_input(
    "Jarak ke Stasiun MRT Terdekat (meter)",
    min_value=0.0, max_value=7000.0, value=500.0, step=50.0
)
num_convenience = st.sidebar.slider(
    "Jumlah Convenience Store Terdekat",
    min_value=0, max_value=10, value=5, step=1
)
latitude = st.sidebar.number_input(
    "Latitude", min_value=24.90, max_value=25.10,
    value=24.97, step=0.005, format="%.5f"
)
longitude = st.sidebar.number_input(
    "Longitude", min_value=121.40, max_value=121.60,
    value=121.52, step=0.005, format="%.5f"
)

# =============================================
# TOMBOL PREDIKSI
# =============================================
st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🚀 Jalankan Prediksi", use_container_width=True, type="primary")

# =============================================
# TAMPILAN UTAMA
# =============================================
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📋 Detail Properti")

    c1, c2, c3 = st.columns(3)
    c1.metric("Usia Bangunan", f"{house_age:.0f} tahun")
    c2.metric("Jarak ke MRT", f"{distance_to_mrt:,.0f} m")
    c3.metric("Convenience Store", f"{num_convenience} toko")

    st.markdown("")
    detail = pd.DataFrame({
        "Parameter": [
            "Tahun Transaksi", "Usia Bangunan", "Jarak ke MRT",
            "Convenience Store", "Latitude", "Longitude",
        ],
        "Nilai": [
            f"{transaction_date:.3f}", f"{house_age:.1f} tahun",
            f"{distance_to_mrt:,.1f} meter", f"{num_convenience} toko",
            f"{latitude:.5f}", f"{longitude:.5f}",
        ],
    })
    st.dataframe(detail, use_container_width=True, hide_index=True)

with col_right:
    st.subheader("🔮 Hasil Prediksi")

    if predict_btn:
        input_array = np.array([[
            transaction_date,
            house_age,
            distance_to_mrt,
            num_convenience,
            latitude,
            longitude,
        ]])

        prediction = skl_model.predict(input_array)[0]

        st.success(f"### 💰 Harga: {prediction:,.2f}")
        st.metric("Harga Prediksi per Unit Area", f"{prediction:,.2f}")
        st.caption("Satuan: sesuai dataset training di Orange")

        # Insight
        st.markdown("---")
        st.subheader("💡 Insight")
        insights = []
        if house_age > 30:
            insights.append("🏚️ Bangunan cukup tua (>30 tahun), biasanya berkorelasi negatif dengan harga.")
        if distance_to_mrt > 1500:
            insights.append("🚇 Jarak ke MRT jauh (>1.5 km). Properti dekat transportasi umum cenderung lebih mahal.")
        if distance_to_mrt < 300:
            insights.append("🚇 Sangat dekat MRT (<300 m) — lokasi premium!")
        if num_convenience >= 7:
            insights.append("🏪 Banyak fasilitas di sekitar — ini biasanya mendongkrak harga.")
        if num_convenience <= 2:
            insights.append("🏪 Fasilitas sekitar minim, bisa menekan harga.")
        if not insights:
            st.info("✅ Properti dengan karakteristik standar.")
        else:
            for i in insights:
                st.info(i)
    else:
        st.info("👈 Isi data properti di sidebar, lalu klik **Jalankan Prediksi**.")

# =============================================
# FEATURE IMPORTANCE (selalu tampil)
# =============================================
st.markdown("---")
st.subheader("📊 Feature Importance")

feat_names = [
    "Tahun Transaksi", "Usia Bangunan", "Jarak ke MRT",
    "Convenience Store", "Latitude", "Longitude",
]
importances = pd.DataFrame({
    "Faktor": feat_names,
    "Importance": skl_model.feature_importances_,
}).sort_values("Importance", ascending=True)
st.bar_chart(importances.set_index("Faktor"))

# =============================================
# FOOTER
# =============================================
st.markdown("---")
st.caption(
    "📌 Model: AdaBoost Regressor (pickle dari Orange Data Mining) | "
    "Dibuat untuk Pelatihan Data Analitik DJPb/Anggaran."
)
