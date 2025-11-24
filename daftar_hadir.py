import streamlit as st


def run():
    st.title("🐾 Daftar Hadir Kucing")

    # Input data
    nama_pemilik = st.text_input("Nama Pemilik")
    nama_kucing = st.text_input("Nama Kucing")

    # Tombol untuk pindah ke halaman hasil
    if st.button("Masuk"):
        if nama_pemilik and nama_kucing:
            # Simpan input ke session_state
            st.session_state.nama_pemilik = nama_pemilik
            st.session_state.nama_kucing = nama_kucing
            st.session_state.page = "hasil"
            st.rerun()
        else:
            st.warning("Mohon isi semua kolom sebelum melanjutkan.")
