import streamlit as st
import daftar_hadir
import hasil

st.set_page_config(page_title="Daftar Hadir Kucing", page_icon="🐱")

# Inisialisasi state
if "page" not in st.session_state:
    st.session_state.page = "daftar_hadir"

# Navigasi antar halaman
if st.session_state.page == "daftar_hadir":
    daftar_hadir.run()
elif st.session_state.page == "hasil":
    hasil.run()
