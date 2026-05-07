import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "model_orange.pickle"

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE CONFIG
# Sesuaikan nama key dengan nama variabel/kolom saat training di Orange
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_CONFIG = {
    "luas_tanah": {
        "type": "numeric",
        "input": "number",
        "min": 20,
        "max": 10000,
        "default": 100,
        "label": "Luas Tanah (m²)",
    },
    "luas_bangunan": {
        "type": "numeric",
        "input": "number",
        "min": 10,
        "max": 5000,
        "default": 80,
        "label": "Luas Bangunan (m²)",
    },
    "jumlah_kamar_tidur": {
        "type": "numeric",
        "input": "slider",
        "min": 1,
        "max": 10,
        "default": 3,
        "label": "Jumlah Kamar Tidur",
    },
    "jumlah_kamar_mandi": {
        "type": "numeric",
        "input": "slider",
        "min": 1,
        "max": 8,
        "default": 2,
        "label": "Jumlah Kamar Mandi",
    },
    "jumlah_lantai": {
        "type": "numeric",
        "input": "slider",
        "min": 1,
        "max": 5,
        "default": 1,
        "label": "Jumlah Lantai",
    },
    "usia_bangunan": {
        "type": "numeric",
        "input": "slider",
        "min": 0,
        "max": 50,
        "default": 5,
        "label": "Usia Bangunan (tahun)",
    },
    "jarak_ke_pusat_kota": {
        "type": "numeric",
        "input": "number",
        "min": 0.0,
        "max": 100.0,
        "default": 10.0,
        "label": "Jarak ke Pusat Kota (km)",
    },
    "kondisi_bangunan": {
        "type": "categorical",
        "options": ["Baru", "Baik", "Sedang", "Perlu Renovasi"],
        "label": "Kondisi Bangunan",
    },
    "lokasi": {
        "type": "categorical",
        "options": ["Strategis", "Cukup Strategis", "Pinggiran"],
        "label": "Lokasi",
    },
    "fasilitas": {
        "type": "categorical",
        "options": ["Lengkap", "Cukup", "Minim"],
        "label": "Fasilitas",
    },
}


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the Orange pickle model from the repository."""
    if not MODEL_PATH.exists():
        st.error(
            f"❌ File model tidak ditemukan di `{MODEL_PATH.name}`. "
            "Pastikan file `model_orange.pickle` sudah di-upload ke GitHub repository."
        )
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None


# ─────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────
def create_input_form():
    """Render input widgets and return dict of user values."""
    input_data = {}
    with st.form("prediction_form"):
        st.subheader("📋 Masukkan Data Properti")

        col1, col2 = st.columns(2)
        feature_items = list(FEATURE_CONFIG.items())
        mid = len(feature_items) // 2

        for idx, (feature_name, config) in enumerate(feature_items):
            col = col1 if idx < mid else col2
            label = config.get("label", feature_name)

            with col:
                if config["type"] == "numeric":
                    if config["input"] == "slider":
                        val = st.slider(
                            label,
                            min_value=int(config["min"]),
                            max_value=int(config["max"]),
                            value=int(config["default"]),
                        )
                    else:
                        val = st.number_input(
                            label,
                            min_value=float(config["min"]),
                            max_value=float(config["max"]),
                            value=float(config["default"]),
                            step=1.0,
                        )
                    input_data[feature_name] = val

                elif config["type"] == "categorical":
                    val = st.selectbox(label, options=config["options"])
                    input_data[feature_name] = val

        submitted = st.form_submit_button("🔍 Prediksi Harga", use_container_width=True)

    return input_data, submitted


# ─────────────────────────────────────────────
# PREDICTION — scikit-learn style
# ─────────────────────────────────────────────
def predict_with_model(model, input_df: pd.DataFrame):
    """Try sklearn-style .predict()."""
    prediction = model.predict(input_df)
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(input_df)
        except Exception:
            pass
    return prediction, proba


# ─────────────────────────────────────────────
# PREDICTION — Orange fallback
# ─────────────────────────────────────────────
def predict_with_orange_fallback(model, input_df: pd.DataFrame):
    """Fallback: convert DataFrame to Orange Table and run prediction."""
    try:
        import Orange.data as od

        domain_vars = []
        for col in input_df.columns:
            cfg = FEATURE_CONFIG.get(col, {})
            if cfg.get("type") == "categorical":
                values = [str(v) for v in cfg.get("options", input_df[col].unique())]
                domain_vars.append(od.DiscreteVariable(col, values=values))
            else:
                domain_vars.append(od.ContinuousVariable(col))

        domain = od.Domain(domain_vars)

        # Build numpy array (categorical → index)
        row = []
        for col in input_df.columns:
            val = input_df[col].iloc[0]
            cfg = FEATURE_CONFIG.get(col, {})
            if cfg.get("type") == "categorical":
                options = cfg.get("options", [])
                row.append(float(options.index(val)) if val in options else 0.0)
            else:
                row.append(float(val))

        X = np.array([row])
        orange_table = od.Table.from_numpy(domain, X)
        result = model(orange_table)

        # Orange returns array-like
        prediction = np.array(result).flatten()
        return prediction, None

    except ImportError:
        raise RuntimeError(
            "Library `orange3` tidak tersedia. "
            "Tambahkan `orange3` ke `requirements.txt` dan deploy ulang."
        )
    except Exception as e:
        raise RuntimeError(f"Prediksi Orange gagal: {e}")


# ─────────────────────────────────────────────
# FORMAT CURRENCY
# ─────────────────────────────────────────────
def format_rupiah(value):
    try:
        num = float(value)
        if num >= 1_000_000_000:
            return f"Rp {num/1_000_000_000:,.2f} Miliar"
        elif num >= 1_000_000:
            return f"Rp {num/1_000_000:,.0f} Juta"
        else:
            return f"Rp {num:,.0f}"
    except Exception:
        return str(value)


# ─────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Rumah",
    page_icon="🏠",
    layout="wide",
)

# Sidebar
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/home.png",
        width=80,
    )
    st.title("🏠 Prediksi Harga Rumah")
    st.markdown(
        """
        **Cara Penggunaan:**
        1. Isi data properti pada form utama.
        2. Klik tombol **Prediksi Harga**.
        3. Hasil prediksi akan muncul di bawah form.

        ---
        **Tentang Model:**
        - Model dilatih menggunakan **Orange Data Mining**.
        - File model: `model_orange.pickle` (disimpan di GitHub repository).
        - Prediksi bersifat estimasi berdasarkan data historis.
        ---
        """
    )
    st.caption("© 2025 · Aplikasi Prediksi Harga Rumah")

# Header
st.title("🏠 Aplikasi Prediksi Harga Rumah")
st.markdown(
    "Aplikasi ini menggunakan model machine learning hasil training dari **Orange Data Mining** "
    "dan dijalankan melalui **Streamlit Cloud**. Masukkan data properti untuk mendapatkan estimasi harga."
)
st.divider()

# Load model
model = load_model()

if model is None:
    st.warning(
        "⚠️ Model belum dimuat. Pastikan `model_orange.pickle` ada di root repository GitHub."
    )
    st.stop()

st.success("✅ Model berhasil dimuat.")

# Input form
input_data, submitted = create_input_form()

# Predict
if submitted:
    input_df = pd.DataFrame([input_data], columns=list(FEATURE_CONFIG.keys()))

    st.divider()
    st.subheader("📊 Data yang Dimasukkan")
    display_df = input_df.copy()
    display_df.columns = [
        FEATURE_CONFIG[c].get("label", c) for c in display_df.columns
    ]
    st.dataframe(display_df, use_container_width=True)

    with st.spinner("Menghitung prediksi..."):
        prediction = None
        proba = None
        error_msg = None

        # Try sklearn-style first
        try:
            prediction, proba = predict_with_model(model, input_df)
        except Exception as sklearn_err:
            # Fallback to Orange
            try:
                prediction, proba = predict_with_orange_fallback(model, input_df)
            except Exception as orange_err:
                error_msg = (
                    f"**Prediksi sklearn:** {sklearn_err}\n\n"
                    f"**Prediksi Orange:** {orange_err}"
                )

    st.divider()
    st.subheader("🎯 Hasil Prediksi")

    if error_msg:
        st.error(
            "❌ Prediksi gagal. Kemungkinan penyebab:\n"
            "- Nama kolom input tidak cocok dengan fitur saat training.\n"
            "- Format model berbeda.\n\n"
            f"Detail error:\n{error_msg}"
        )
    elif prediction is not None:
        pred_val = prediction[0] if hasattr(prediction, "__len__") else prediction

        col1, col2 = st.columns([1, 1])
        with col1:
            st.success(f"### 💰 Estimasi Harga: **{format_rupiah(pred_val)}**")
            st.caption(f"Nilai mentah: {pred_val}")

        with col2:
            if proba is not None:
                st.info("**Probabilitas / Confidence:**")
                st.write(proba)
            else:
                st.info("ℹ️ Model tidak menyediakan nilai probabilitas (model regresi).")
    else:
        st.error("❌ Prediksi menghasilkan nilai None yang tidak terduga.")
