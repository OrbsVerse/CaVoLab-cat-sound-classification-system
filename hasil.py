import streamlit as st
import numpy as np
import joblib
import librosa
import os
import pandas as pd

# Konstanta
MODEL_PATH = "model/best_svm.pkl"
N_MFCC = 20


# ---------------------------------------------------------
# 1. CLASS INPUT (Sesuai Diagram)
# ---------------------------------------------------------
class Input:
    def __init__(self, uploaded_file):
        """
        Menangani input file dari user.
        """
        self.file = uploaded_file
        self.path = None

    def save_temp(self):
        """Menyimpan file sementara agar bisa dibaca Librosa"""
        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        self.path = f"uploads/{self.file.name}"
        with open(self.path, "wb") as f:
            f.write(self.file.getbuffer())
        return self.path


# ---------------------------------------------------------
# 2. CLASS PREPROCESSING (Sesuai Diagram)
# ---------------------------------------------------------
class Preprocessing:
    def __init__(self):
        self.sampling_rate = None
        self.features = None
        self.duration = 0.0
        self.noise_level = 0.0  # Typo di diagram 'noice', di kode sudah benar 'noise'

    def resample(self, path, target_sr=None):
        """
        Menambahkan method ini agar sesuai Class Diagram.
        """
        y, sr = librosa.load(path, sr=target_sr)
        self.sampling_rate = sr
        return y, sr

    def noiseReduction(self, y):
        """
        Sesuai Diagram: noiseReduction(input)
        Implementasi sederhana: Menghapus silence (hening) di awal/akhir
        sebagai bentuk pengurangan noise dasar.
        """
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        return y_trimmed

    def normalize(self, y):
        """
        Sesuai Diagram: normalize(input)
        Mengubah amplitudo audio ke range -1 hingga 1 (Norm)
        """
        return librosa.util.normalize(y)

    def featureExtraction(self, y, sr):
        """
        Sesuai Diagram: featureExtraction(input)
        Mengekstrak MFCC, Chroma, ZCR, RMS
        """
        self.sampling_rate = sr
        self.duration = librosa.get_duration(y=y, sr=sr)

        # Ekstraksi Fitur
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        rms = librosa.feature.rms(y=y)

        # Aggregasi (Mean & Std)
        self.features = np.concatenate(
            [
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.mean(chroma, axis=1),
                [np.mean(zcr)],
                [np.mean(rms)],
            ]
        )
        return self.features.reshape(1, -1)

    def getFeatures(self):
        return self.features


# ---------------------------------------------------------
# 3. CLASS CLASSIFICATION (Sesuai Diagram)
# ---------------------------------------------------------
class Classification:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, features):
        """
        Sesuai Diagram: predict(input)
        Mengembalikan probabilitas prediksi
        """
        return self.model.predict_proba(features)[0]


# ---------------------------------------------------------
# 4. MAIN CONTROLLER & UI (Menggabungkan Semuanya)
# ---------------------------------------------------------
def run():
    st.title("📋 Hasil Kehadiran & Prediksi Suara Kucing")

    if "nama_pemilik" in st.session_state and "nama_kucing" in st.session_state:
        st.divider()
        st.subheader("🎧 Unggah Suara Kucing")

        uploaded_file = st.file_uploader(
            "Upload rekaman suara kucing",
            type=["wav", "mp3"],
            help="Maksimal ukuran file 5 MB",
        )

        if uploaded_file is not None:
            # 1. Instansiasi Objek Input
            input_obj = Input(uploaded_file)
            audio_path = input_obj.save_temp()

            st.audio(audio_path)

            try:
                # Load Audio menggunakan Librosa (Awal proses)
                prep = Preprocessing()
                y, sr = prep.resample(audio_path, target_sr=None)

                # --- Alur sesuai Method di Diagram ---
                y_denoised = prep.noiseReduction(y)  # Panggil noiseReduction
                y_normalized = prep.normalize(y_denoised)  # Panggil normalize
                features = prep.featureExtraction(
                    y_normalized, sr
                )  # Panggil featureExtraction

                # 3. Instansiasi Objek Classification
                classifier = Classification(MODEL_PATH)
                prediction_prob = classifier.predict(features)

                # --- Menampilkan Hasil (UI) ---
                pred_index = np.argmax(prediction_prob)
                confidence = prediction_prob[pred_index] * 100

                label_map = {
                    0: "Kucing sedang brushing 😺",
                    1: "Kucing menunggu makanan 🍽",
                    2: "Kucing terisolasi 😾",
                }
                hasil_prediksi = label_map.get(pred_index, "Tidak diketahui")
                st.session_state.hasil_prediksi = hasil_prediksi

                st.divider()
                st.subheader("📊 Hasil Prediksi Otomatis")

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Nama Pemilik:** {st.session_state.nama_pemilik}")
                    st.write(f"**Nama Kucing:** {st.session_state.nama_kucing}")

                st.info(
                    f"**Hasil:** {hasil_prediksi}\n\n**Tingkat Keyakinan:** {confidence:.1f}%"
                )

                # Visualisasi Chart
                st.write("---")
                st.caption("Rincian Probabilitas Sistem:")
                data_prob = {
                    "Kondisi": [label_map[0], label_map[1], label_map[2]],
                    "Persentase": [f"{p*100:.1f}%" for p in prediction_prob],
                    "Nilai": prediction_prob,
                }
                df_prob = pd.DataFrame(data_prob)
                st.bar_chart(df_prob.set_index("Kondisi")["Nilai"])

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

        if st.button("⬅ Kembali ke Daftar Hadir"):
            st.session_state.page = "daftar_hadir"
            st.rerun()
    else:
        st.warning("Silakan isi daftar hadir terlebih dahulu.")
