# CaVoLab-cat-sound-classification-system

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

Aplikasi berbasis web untuk menerjemahkan arti suara kucing menggunakan **Machine Learning (SVM)** dan **Digital Signal Processing**. Aplikasi ini dapat membedakan apakah kucing sedang ingin dimanja (brushing), lapar, atau merasa terisolasi.

🔗 **Demo Aplikasi:** [Klik di sini untuk mencoba](MASUKKAN_LINK_STREAMLIT_CLOUD_DISINI)

## 📊 Dataset & Kredit
Dataset yang digunakan untuk melatih model ini bersumber dari Kaggle:

* **Nama Dataset:** Cat Meow Classification
* **Sumber:** Kaggle - Cat Meow Classification(https://www.kaggle.com/datasets/andrewmvd/cat-meow-classification)
* **Penulis Asli:** Larxel
* **Lisensi:** CC BY NC 4.0

Terima kasih kepada penulis asli yang telah menyediakan dataset ini secara publik.

## 📋 Fitur Utama
* **Sistem Presensi:** Mencatat data pemilik dan nama kucing sebelum melakukan prediksi.
* **Upload Audio:** Mendukung format `.mp3` dan `.wav` dengan batas ukuran 5MB.
* **Prediksi Cerdas:** Mengklasifikasikan suara ke dalam 3 kategori:
    1.  😺 **Brushing** (Kucing sedang senang/manja)
    2.  🍽 **Menunggu Makanan** (Lapar)
    3.  😾 **Terisolasi** (Kesepian/Marah)
* **Confidence Score:** Menampilkan tingkat keyakinan (persentase) dari prediksi AI.
* **Visualisasi Data:** Grafik batang probabilitas untuk setiap kemungkinan label.

## 📂 Struktur Proyek
Proyek ini menggunakan arsitektur modular agar kode lebih rapi:

```text
├── .streamlit/
│   └── config.toml      # Konfigurasi server (Max Upload 5MB)
├── model/
│   └── best_svm.pkl     # Model SVM yang sudah dilatih (Pickle)
├── daftar_hadir.py      # Halaman input data pemilik & kucing
├── hasil.py             # Halaman utama proses prediksi & visualisasi
├── main.py              # File utama (Navigation & Routing)
├── packages.txt         # Dependensi sistem Linux (libsndfile1)
└── requirements.txt     # Daftar library Python
