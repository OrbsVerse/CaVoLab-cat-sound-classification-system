import streamlit as st
import joblib
import numpy as np
import librosa
import os
import pandas as pd  

# Konstanta
MODEL_PATH = "model/best_svm.pkl"
N_MFCC = 20


def extract_features(path):
    y, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    rms = librosa.feature.rms(y=y)

    feat = np.concatenate(
        [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(chroma, axis=1),
            [np.mean(zcr)],
            [np.mean(rms)],
        ]
    )
    return feat.reshape(1, -1)


def run():
    st.title("📋 Hasil Kehadiran & Prediksi Suara Kucing")

    if "nama_pemilik" in st.session_state and "nama_kucing" in st.session_state:
        st.divider()
        st.subheader("🎧 Unggah Suara Kucing")

        audio_file = st.file_uploader(
            "Upload rekaman suara kucing",
            type=["wav", "mp3"],
            help="Maksimal ukuran file 5 MB",
        )

        if audio_file is not None:
            # Validasi Manual (Backup jika config.toml belum diset)
            file_type = audio_file.name.split(".")[-1].lower()
            file_size_mb = audio_file.size / (1024 * 1024)

            if file_type not in ["wav", "mp3"]:
                st.error("Format file salah. Harap upload file .mp3 atau .wav.")
            elif file_size_mb > 5:
                st.error("Ukuran file melebihi 5 MB.")
            else:
                # Simpan file
                if not os.path.exists("uploads"):
                    os.makedirs("uploads")

                uploaded_path = f"uploads/{audio_file.name}"
                with open(uploaded_path, "wb") as f:
                    f.write(audio_file.getbuffer())

                st.audio(uploaded_path)

                try:
                    # 1. Load Model
                    model = joblib.load(MODEL_PATH)
                    feat = extract_features(uploaded_path)

                    # 2. Prediksi Label & Probabilitas
                    # predict_proba mengembalikan array probabilitas untuk setiap kelas [p1, p2, p3]
                    prediction_prob = model.predict_proba(feat)[0]
                    pred_index = np.argmax(
                        prediction_prob
                    )  # Ambil index dengan nilai tertinggi
                    confidence = prediction_prob[pred_index] * 100  # Ubah ke persen

                    label_map = {
                        0: "Kucing sedang brushing 😺",
                        1: "Kucing menunggu makanan 🍽",
                        2: "Kucing terisolasi 😾",
                    }

                    hasil_prediksi = label_map.get(pred_index, "Tidak diketahui")
                    st.session_state.hasil_prediksi = hasil_prediksi

                    # 3. Tampilkan Hasil Utama
                    st.divider()
                    st.subheader("📊 Hasil Prediksi Otomatis")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Nama Pemilik:** {st.session_state.nama_pemilik}")
                        st.write(f"**Nama Kucing:** {st.session_state.nama_kucing}")

                    # Menampilkan hasil prediksi dengan persentase besar
                    st.info(
                        f"**Hasil:** {hasil_prediksi}\n\n**Tingkat Keyakinan:** {confidence:.1f}%"
                    )

                    # 4. Tampilkan Rincian Persentase (Opsional tapi bagus)
                    st.write("---")
                    st.caption("Rincian Probabilitas Sistem:")

                    # Membuat DataFrame untuk chart
                    data_prob = {
                        "Kondisi": [label_map[0], label_map[1], label_map[2]],
                        "Persentase": [f"{p*100:.1f}%" for p in prediction_prob],
                        "Nilai": prediction_prob,  # Untuk bar chart
                    }

                    df_prob = pd.DataFrame(data_prob)

                    # Menampilkan tabel sederhana
                    st.table(df_prob[["Kondisi", "Persentase"]])

                    # Menampilkan Bar Chart
                    st.bar_chart(df_prob.set_index("Kondisi")["Nilai"])

                except AttributeError:
                    st.error(
                        "Model SVM Anda tidak mendukung probabilitas. Pastikan saat training menggunakan parameter `probability=True`."
                    )
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

        if st.button("⬅ Kembali ke Daftar Hadir"):
            st.session_state.page = "daftar_hadir"
            st.rerun()
    else:
        st.warning("Silakan isi daftar hadir terlebih dahulu.")
