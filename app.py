"""Sistem Informasi Klasifikasi Penggunaan dan Tutupan Lahan Jawa Timur (UTS Sistem Informasi Geografis).

Aplikasi hanya membaca berkas pada folder `hasil/` yang dihasilkan oleh notebook
`klasifikasi_lulc_jawa_timur_uts_sig.ipynb`, sehingga tidak memerlukan kredensial Copernicus.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

HASIL = Path(__file__).parent / "hasil"

st.set_page_config(page_title="SIG LULC Jawa Timur", layout="wide")


def baca_json(nama):
    p = HASIL / nama
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def baca_csv(nama):
    p = HASIL / nama
    return pd.read_csv(p) if p.exists() else None


def baca_teks(nama):
    p = HASIL / nama
    return p.read_text(encoding="utf-8") if p.exists() else None


def tampil_gambar(nama, keterangan):
    p = HASIL / nama
    if p.exists():
        st.image(str(p), caption=keterangan, width="stretch")
    else:
        st.info(f"Berkas {nama} belum tersedia pada folder hasil/.")


ringkasan = baca_json("ringkasan.json")

st.sidebar.title("SIG LULC Jawa Timur")
menu = ["Ringkasan", "Peta Klasifikasi", "Evaluasi Model"]
if (HASIL / "proses_notebook.html").exists():          # menu ini hanya muncul bila berkasnya disertakan
    menu.append("Proses Analisis (Web Statis)")
halaman = st.sidebar.radio("Navigasi", menu)
st.sidebar.caption(
    "Ujian Tengah Semester Sistem Informasi Geografis: analisis spasial penggunaan lahan dan kebijakan."
)

if ringkasan is None:
    st.error("Berkas hasil/ringkasan.json tidak ditemukan. Jalankan notebook terlebih dahulu, lalu salin folder hasil/ ke repositori.")
    st.stop()

# ---------------------------------------------------------------- Ringkasan
if halaman == "Ringkasan":
    st.title(ringkasan["judul"])
    st.write(
        "Sistem informasi ini menyajikan hasil klasifikasi penggunaan dan tutupan lahan Provinsi Jawa Timur "
        "ke dalam enam kelas menggunakan citra Sentinel-2 dan model pembelajaran mesin berbasis poligon."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Jumlah kelas", ringkasan["jumlah_kelas"])
    k2.metric("Data training (poligon)", ringkasan["jumlah_training"])
    k3.metric("Data testing (poligon)", ringkasan["jumlah_testing"])
    k4.metric("Akurasi testing", f"{ringkasan['akurasi_testing']:.2%}")

    k5, k6, k7 = st.columns(3)
    k5.metric("Model akhir", ringkasan["model_akhir"])
    k6.metric("Kappa", f"{ringkasan['kappa_testing']:.3f}")
    k7.metric("F1 macro", f"{ringkasan['f1_macro_testing']:.3f}")

    ekstra = []
    if ringkasan.get("akurasi_validasi_spasial") is not None:
        ekstra.append(f"Akurasi validasi spasial: {ringkasan['akurasi_validasi_spasial']:.1%}.")
    if ringkasan.get("kesesuaian_worldcover") is not None:
        ekstra.append(f"Kesesuaian dengan ESA WorldCover (pembanding, dibobot luas kelas): {ringkasan['kesesuaian_worldcover']:.1%}.")
    if ekstra:
        st.caption(" ".join(ekstra))

    st.caption(
        f"Selang kepercayaan 95% akurasi: ±{ringkasan['selang_kepercayaan_95']:.1%}. "
        f"Periode citra: {ringkasan['periode_citra'][0]} s.d. {ringkasan['periode_citra'][1]}. "
        f"Ukuran sel peta: sekitar {ringkasan['ukuran_sel_m']} m. "
        f"Fitur akhir: {len(ringkasan['fitur_akhir'])} ({ringkasan['skenario_fitur']})."
    )

    st.subheader("Jumlah data per kelas")
    tabel = pd.DataFrame(ringkasan["per_kelas"]).T.rename(
        columns={"total": "Total", "training": "Training", "testing": "Testing"}
    )
    tabel.index.name = "Kelas"
    st.dataframe(tabel, width="stretch")

    luas = baca_csv("luas_kelas.csv")
    if luas is not None:
        st.subheader("Luas tiap kelas pada area studi")
        st.dataframe(luas, width="stretch", hide_index=True)
        st.bar_chart(luas.set_index("Kelas")["Luas (km²)"])

# ---------------------------------------------------------------- Peta
elif halaman == "Peta Klasifikasi":
    st.title("Peta Jawa Timur")
    st.write(
        "Peta dasar menggunakan Google Satellite. Hasil klasifikasi dan sampel referensi ditampilkan pada "
        "dua peta terpisah. Gunakan kontrol lapisan di kanan atas peta untuk menampilkan atau menyembunyikan kelas."
    )
    tab_hasil, tab_sampel = st.tabs(["Hasil klasifikasi", "Sampel referensi (data latih)"])
    with tab_hasil:
        st.caption("Centang kelas pada kontrol lapisan untuk menampilkan hasil klasifikasi. Peta dipotong mengikuti batas Provinsi Jawa Timur.")
        html = baca_teks("peta_klasifikasi_jatim.html")
        if html is None:
            st.warning("Berkas hasil/peta_klasifikasi_jatim.html belum tersedia.")
        else:
            components.html(html, height=720, scrolling=False)
    with tab_sampel:
        st.caption("Poligon sampel yang dipakai melatih model, digambar dengan garis tepi putus-putus.")
        html = baca_teks("peta_sampel_referensi.html")
        if html is None:
            st.warning("Berkas hasil/peta_sampel_referensi.html belum tersedia.")
        else:
            components.html(html, height=720, scrolling=False)

# ---------------------------------------------------------------- Evaluasi
elif halaman == "Evaluasi Model":
    st.title("Evaluasi Model")
    st.write(
        f"Model akhir: **{ringkasan['model_akhir']}** (dipilih dari tiga algoritma: Extra Trees, Regresi Logistik, "
        "dan SVM, berdasarkan validasi silang berulang pada data training)."
    )

    st.subheader("Perbandingan model")
    banding = baca_csv("perbandingan_model.csv")
    if banding is not None:
        st.dataframe(banding, width="stretch", hide_index=True)
        st.caption(
            "Kolom Akurasi Testing hanya sebagai informasi; pemilihan model akhir didasarkan pada Akurasi CV (data training)."
        )
    tampil_gambar("perbandingan_model.png", "Akurasi validasi silang berulang")

    diag = baca_csv("diagnosis_overfitting.csv")
    if diag is not None:
        st.subheader("Diagnosis overfitting dan underfitting")
        st.write(
            "Perbandingan akurasi latih, validasi silang, testing, dan validasi spasial (poligon berdekatan tidak dipisah antar lipatan). "
            "Selisih latih-validasi yang besar menandakan kecenderungan overfitting; akurasi validasi yang rendah menandakan underfitting."
        )
        if ringkasan.get("diagnosis_model_akhir"):
            st.caption(f"Model akhir: {ringkasan['diagnosis_model_akhir']}.")
        st.dataframe(diag, width="stretch", hide_index=True)
        k1, k2 = st.columns(2)
        with k1:
            tampil_gambar("kurva_belajar.png", "Kurva belajar model akhir")
        with k2:
            tampil_gambar("kurva_validasi.png", "Kurva validasi model akhir")

    wc = baca_csv("validasi_worldcover.csv")
    if wc is not None:
        st.subheader("Pembanding eksternal: ESA WorldCover 2021")
        st.write(
            "Sejumlah sel diambil acak dari tiap kelas peta dan dibandingkan dengan kelas mayoritas ESA WorldCover. "
            "Ini hanya pembanding kewajaran, bukan akurasi sebenarnya (WorldCover juga memiliki galat dan tahunnya berbeda)."
        )
        st.dataframe(wc, width="stretch", hide_index=True)
        silang = baca_csv("validasi_worldcover_silang.csv")
        if silang is not None:
            st.caption("Kelas WorldCover yang jatuh pada tiap kelas peta (jumlah sel).")
            st.dataframe(silang, width="stretch", hide_index=True)

    st.subheader("Laporan klasifikasi (data testing)")
    laporan = baca_csv("laporan_klasifikasi.csv")
    if laporan is not None:
        st.dataframe(laporan, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        tampil_gambar("confusion_matrix.png", "Confusion matrix (data testing)")
    with c2:
        tampil_gambar("kepentingan_fitur.png", "Kepentingan fitur (permutation importance)")

# ---------------------------------------------------------------- Proses (web statis)
else:
    st.title("Proses Analisis (Web Statis)")
    st.write("Halaman berikut adalah ekspor HTML dari notebook beserta seluruh keluarannya.")
    html = baca_teks("proses_notebook.html")
    if html is None:
        st.warning(
            "Berkas hasil/proses_notebook.html belum tersedia. Buat dengan perintah: "
            "jupyter nbconvert --to html klasifikasi_lulc_jawa_timur_uts_sig.ipynb --output proses_notebook --output-dir hasil"
        )
    else:
        components.html(html, height=900, scrolling=True)
