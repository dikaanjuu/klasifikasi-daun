# 🌿 Leaf Condition Classifier — Deep Learning Project

Klasifikasi kondisi daun tanaman (**Daun Sehat** / **Daun Kering**) menggunakan Deep Learning berbasis citra gambar.

Project ini dibangun menggunakan:

* Transfer Learning — **MobileNetV2**
* Custom CNN — **DCNN From Scratch**
* Streamlit Deployment
* Google Colab Training Pipeline

---

# 🚀 FITUR PROJECT

Pipeline project ini sudah dioptimalkan untuk:

✅ Dataset split otomatis
✅ Randomized train / validation / test
✅ Anti data leakage
✅ MobileNetV2 Fine Tuning
✅ DCNN Custom Architecture
✅ Focal Loss
✅ Class Weight otomatis
✅ Anti bias training
✅ Realistic augmentation
✅ Early stopping optimal
✅ Reduce learning rate otomatis
✅ Confusion Matrix
✅ Classification Report
✅ Export `.h5` stabil
✅ Sinkron preprocessing dengan Streamlit
✅ Akurasi lebih stabil untuk kamera HP
✅ Dual-model prediction comparison di Streamlit

---

# 📁 STRUKTUR PROJECT

```text
leaf-condition-classifier/
│
├── notebooks/
│   └── Leaf_Classification.ipynb
│
├── streamlit_app/
│   ├── app.py
│   ├── requirements.txt
│   ├── mobilenet_model.h5
│   └── dcnn_model.h5
│
├── README.md
└── .gitignore
```

---

# 📂 STRUKTUR DATASET GOOGLE DRIVE

Gunakan struktur berikut:

```text
Dataset Project Deep Learning/
│
├── Dataset Mentah/
│   ├── Daun Kering/
│   └── Daun Sehat/
│
├── Split Dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
└── Models/
```

---

# ⚡ TRAINING DI GOOGLE COLAB

## 1. Buka Google Colab

[Google Colab](https://colab.research.google.com?utm_source=chatgpt.com)

Aktifkan GPU:

```text
Runtime → Change Runtime Type → T4 GPU
```

---

## 2. Upload Dataset

Masukkan seluruh gambar mentah ke:

```text
Dataset Mentah/
├── Daun Kering/
└── Daun Sehat/
```

JANGAN isi manual folder:

```text
train/
val/
test/
```

Karena pipeline akan melakukan split otomatis dan random.

---

## 3. Jalankan Semua Cell

Pipeline akan otomatis:

✅ split dataset
✅ randomize gambar
✅ training MobileNetV2
✅ training DCNN
✅ evaluasi model
✅ confusion matrix
✅ classification report
✅ save `.h5` model

---

# 🧠 MODEL YANG DIGUNAKAN

---

# 📌 Model 1 — MobileNetV2

Model utama untuk deployment.

## Kelebihan

✅ Akurasi tinggi
✅ Stabil di kamera HP
✅ Generalisasi lebih baik
✅ Transfer learning ImageNet
✅ Lebih ringan untuk Streamlit

## Arsitektur

```text
Input (224x224x3)
↓
Data Augmentation
↓
MobileNetV2 (Fine Tuning)
↓
GlobalAveragePooling2D
↓
Dense(128)
↓
Dropout
↓
Dense(2, Softmax)
```

---

# 📌 Model 2 — DCNN From Scratch

Model pembanding.

## Fungsi

✅ Perbandingan prediksi
✅ Analisis hasil CNN manual
✅ Demonstrasi arsitektur custom

## Arsitektur

```text
Input
↓
Conv2D + BatchNorm + MaxPool
↓
Conv2D + BatchNorm + MaxPool
↓
Conv2D + BatchNorm + MaxPool
↓
Conv2D + BatchNorm + MaxPool
↓
GlobalAveragePooling
↓
Dense
↓
Dropout
↓
Softmax
```

---

# 📊 EVALUASI MODEL

Kedua model dievaluasi menggunakan:

✅ Training Accuracy
✅ Validation Accuracy
✅ Test Accuracy
✅ Loss Curve
✅ Confusion Matrix
✅ Precision
✅ Recall
✅ F1-Score

---

# 🌐 STREAMLIT APP

Aplikasi Streamlit mendukung:

✅ Upload gambar
✅ Kamera HP langsung
✅ Dual prediction model
✅ Confidence score
✅ Perbandingan MobileNetV2 vs DCNN
✅ Probability distribution

---

# 📦 FILE YANG DIUPLOAD KE GITHUB

Upload file berikut:

```text
app.py
requirements.txt
mobilenet_model.h5
dcnn_model.h5
README.md
```

---

# 🚀 DEPLOY KE STREAMLIT CLOUD

## 1. Upload Project ke GitHub

Buat repository GitHub lalu upload semua file.

## 2. Deploy

Buka:

[Streamlit Cloud](https://share.streamlit.io?utm_source=chatgpt.com)

Lalu:

```text
New App
↓
Pilih Repository GitHub
↓
Main File:
streamlit_app/app.py
↓
Deploy
```

---

# ⚠️ PENTING — FILE MODEL

Gunakan file:

```text
mobilenet_model.h5
```

BUKAN:

```text
best_mobilenet.h5
```

Karena:

* `best_mobilenet.h5` → checkpoint training
* `mobilenet_model.h5` → final deploy model

---

# ⚙️ REQUIREMENTS

```text
tensorflow==2.17.0
streamlit==1.35.0
numpy==1.26.4
Pillow==10.3.0
scikit-learn==1.4.2
matplotlib==3.9.0
seaborn==0.13.2
```

---

# 📱 PREPROCESSING STREAMLIT

## MobileNetV2

WAJIB menggunakan preprocessing ini:

```python
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
    img_array
)
```

JANGAN menggunakan:

```python
img_array / 255.0
```

untuk MobileNetV2.

---

## DCNN

Gunakan:

```python
img_array = img_array / 255.0
```

karena DCNN memakai Rescaling biasa.

---

# 📈 HASIL PROJECT

Dengan pipeline ini:

✅ Training lebih stabil
✅ Overfitting lebih kecil
✅ Confidence prediction lebih realistis
✅ Deploy lebih konsisten
✅ Lebih akurat untuk kamera HP
✅ Cocok untuk presentasi dan tugas akhir

---

# 🔥 REKOMENDASI PENINGKATAN DATASET

Agar model lebih kuat di dunia nyata:

✅ tambah variasi background
✅ tambah kondisi cahaya berbeda
✅ tambah sudut pengambilan gambar
✅ tambah blur ringan
✅ tambah foto outdoor
✅ tambah data daun sehat

---

# 👨‍💻 TECHNOLOGY STACK

* TensorFlow / Keras
* MobileNetV2 Transfer Learning
* DCNN Custom CNN
* Google Colab GPU
* Streamlit Cloud
* Python
* Scikit-learn
* Matplotlib
* Seaborn

---

# 🎓 PROJECT

Project Deep Learning — Klasifikasi Kondisi Daun Tanaman menggunakan CNN dan Transfer Learning.
