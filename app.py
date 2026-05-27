"""
🌿 Leaf Condition Classifier — Dual Model App
MobileNetV2 + DCNN Comparison
Deploy Ready for Streamlit Cloud
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="🌿 Leaf Condition Classifier",
    page_icon="🌿",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #07130a 0%,
        #102418 50%,
        #07130a 100%
    );
}

.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    color: #9be7a1;
    margin-bottom: 0.5rem;
}

.sub-title {
    text-align: center;
    color: #b7c9bb;
    margin-bottom: 2rem;
}

.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
}

.result-good {
    color: #69f0ae;
    font-size: 1.8rem;
    font-weight: 700;
}

.result-bad {
    color: #ff8a65;
    font-size: 1.8rem;
    font-weight: 700;
}

.metric-text {
    color: #d0d7de;
    font-size: 1rem;
}

.model-title {
    color: #ffffff;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

.stProgress > div > div {
    background: linear-gradient(
        90deg,
        #69f0ae,
        #43a047
    );
}

.info-box {
    background: rgba(105,240,174,0.08);
    border-left: 4px solid #69f0ae;
    padding: 1rem;
    border-radius: 10px;
    color: #e0e0e0;
}

.footer {
    text-align:center;
    color:#90a4ae;
    margin-top:2rem;
    font-size:0.9rem;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================

st.markdown("""
<h1 class="main-title">
🌿 Leaf Condition Classifier
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p class="sub-title">
Perbandingan prediksi MobileNetV2 dan DCNN untuk klasifikasi kondisi daun
</p>
""", unsafe_allow_html=True)

# ==========================================================
# INFO
# ==========================================================

st.markdown("""
<div class="info-box">

✅ Model yang digunakan:

• MobileNetV2 (Transfer Learning)  
• DCNN (Deep CNN From Scratch)

📌 Kelas:
- Daun Sehat
- Daun Kering

📷 Gunakan upload gambar atau kamera HP untuk prediksi realtime.

</div>
""", unsafe_allow_html=True)

# ==========================================================
# FIX TrueDivide ERROR
# ==========================================================

class TrueDivide(tf.keras.layers.Layer):

    def call(self, inputs):
        return inputs / 127.5


# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():

    # LOAD MOBILENET
    mobilenet_model = tf.keras.models.load_model(
        "best_mobilenet.h5",
        compile=False
    )

    # LOAD DCNN
    dcnn_model = tf.keras.models.load_model(
        "dcnn_model.h5",
        compile=False
    )

    return mobilenet_model, dcnn_model


try:

    mobilenet_model, dcnn_model = load_models()

    st.success("✅ Semua model berhasil dimuat")

except Exception as e:

    st.error(f"Gagal load model:\n\n{e}")

    st.info("""
    Pastikan file berikut ada dalam folder yang sama:

    • app.py
    • best_mobilenet.h5
    • dcnn_model.h5
    • requirements.txt
    """)

    st.stop()

# ==========================================================
# CLASS NAMES
# ==========================================================

CLASS_NAMES = [
    "Daun Kering",
    "Daun Sehat"
]

# ==========================================================
# PREPROCESSING
# ==========================================================

def preprocess_mobilenet(image):

    img = image.convert("RGB")
    img = img.resize((224, 224))

    arr = np.array(img).astype(np.float32)

    # preprocess manual
    arr = (arr / 127.5) - 1.0

    arr = np.expand_dims(arr, axis=0)

    return arr


def preprocess_dcnn(image):

    img = image.convert("RGB")
    img = img.resize((224, 224))

    arr = np.array(img).astype(np.float32)

    arr = arr / 255.0

    arr = np.expand_dims(arr, axis=0)

    return arr

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_model(model, img_array):

    prediction = model.predict(
        img_array,
        verbose=0
    )[0]

    class_idx = int(np.argmax(prediction))

    label = CLASS_NAMES[class_idx]

    confidence = float(prediction[class_idx]) * 100

    scores = {
        CLASS_NAMES[i]: float(prediction[i]) * 100
        for i in range(len(CLASS_NAMES))
    }

    return label, confidence, scores

# ==========================================================
# INPUT SECTION
# ==========================================================

st.markdown("---")

input_mode = st.radio(
    "Pilih input gambar:",
    [
        "📤 Upload Gambar",
        "📷 Kamera"
    ],
    horizontal=True
)

image_input = None

if input_mode == "📤 Upload Gambar":

    uploaded_file = st.file_uploader(
        "Upload gambar daun",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image_input = Image.open(uploaded_file)

else:

    camera_image = st.camera_input(
        "Ambil foto daun"
    )

    if camera_image is not None:
        image_input = Image.open(camera_image)

# ==========================================================
# PREDICTION
# ==========================================================

if image_input is not None:

    st.markdown("---")

    preview_col, result_col = st.columns([1, 2])

    # ======================================================
    # IMAGE PREVIEW
    # ======================================================

    with preview_col:

        st.image(
            image_input,
            caption="Input Image",
            use_container_width=True
        )

    # ======================================================
    # RESULTS
    # ======================================================

    with result_col:

        with st.spinner("🔍 Menganalisis gambar..."):

            # MobileNet
            mn_input = preprocess_mobilenet(image_input)

            mn_label, mn_conf, mn_scores = predict_model(
                mobilenet_model,
                mn_input
            )

            # DCNN
            dcnn_input = preprocess_dcnn(image_input)

            dcnn_label, dcnn_conf, dcnn_scores = predict_model(
                dcnn_model,
                dcnn_input
            )

        col_mn, col_dcnn = st.columns(2)

        # ==================================================
        # MOBILENET RESULT
        # ==================================================

        with col_mn:

            st.markdown("""
            <div class="card">
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="model-title">
            🚀 MobileNetV2
            </div>
            """, unsafe_allow_html=True)

            mn_class = (
                "result-good"
                if mn_label == "Daun Sehat"
                else "result-bad"
            )

            st.markdown(f"""
            <div class="{mn_class}">
            {mn_label}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-text">
            Confidence: <b>{mn_conf:.2f}%</b>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Probabilitas")

            for cls, score in sorted(
                mn_scores.items(),
                key=lambda x: -x[1]
            ):

                st.write(f"{cls} — {score:.2f}%")
                st.progress(int(score))

            st.markdown("""
            </div>
            """, unsafe_allow_html=True)

        # ==================================================
        # DCNN RESULT
        # ==================================================

        with col_dcnn:

            st.markdown("""
            <div class="card">
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="model-title">
            🧠 DCNN
            </div>
            """, unsafe_allow_html=True)

            dcnn_class = (
                "result-good"
                if dcnn_label == "Daun Sehat"
                else "result-bad"
            )

            st.markdown(f"""
            <div class="{dcnn_class}">
            {dcnn_label}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-text">
            Confidence: <b>{dcnn_conf:.2f}%</b>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Probabilitas")

            for cls, score in sorted(
                dcnn_scores.items(),
                key=lambda x: -x[1]
            ):

                st.write(f"{cls} — {score:.2f}%")
                st.progress(int(score))

            st.markdown("""
            </div>
            """, unsafe_allow_html=True)

    # ======================================================
    # COMPARISON
    # ======================================================

    st.markdown("---")

    st.subheader("📊 Perbandingan Model")

    compare_col1, compare_col2 = st.columns(2)

    with compare_col1:

        st.metric(
            "MobileNetV2 Confidence",
            f"{mn_conf:.2f}%"
        )

    with compare_col2:

        st.metric(
            "DCNN Confidence",
            f"{dcnn_conf:.2f}%"
        )

    # ======================================================
    # FINAL INTERPRETATION
    # ======================================================

    st.markdown("---")

    if mn_conf >= dcnn_conf:

        st.success(
            f"""
            ✅ Prediksi paling meyakinkan berasal dari MobileNetV2:
            {mn_label} ({mn_conf:.2f}%)
            """
        )

    else:

        st.success(
            f"""
            ✅ Prediksi paling meyakinkan berasal dari DCNN:
            {dcnn_label} ({dcnn_conf:.2f}%)
            """
        )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("""
<div class="footer">

🌿 Leaf Condition Classifier  
Deep Learning Project — MobileNetV2 + DCNN

</div>
""", unsafe_allow_html=True)