import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Konfigurasi koneksi ke Google Spreadsheet
SHEET_URL = "https://docs.google.com/spreadsheets/d/1AqC7MXO-n4CFkKrf5WDdf1cU1HjZZ-CyQtBirmCBRNk/edit?gid=878595289"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Load credentials dari Streamlit Secrets
creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
client = gspread.authorize(creds)
spreadsheet = client.open_by_url(SHEET_URL)
worksheet = spreadsheet.worksheet("dataset")
data = worksheet.get_all_records()
df = pd.DataFrame(data)

# Bersihkan dan format data
df.columns = [col.replace("data_", "") for col in df.columns]
df["TANGGAL"] = pd.to_datetime(df["TANGGAL"], format="%d-%b-%Y")

# Sidebar: Filter data
st.sidebar.header("Filter Data")
start_date = st.sidebar.date_input("Mulai Tanggal", df["TANGGAL"].min())
end_date = st.sidebar.date_input("Akhir Tanggal", df["TANGGAL"].max())
search_query = st.sidebar.text_input("Cari Judul atau Tema", "")

# Terapkan filter
filtered_df = df[(df["TANGGAL"] >= pd.to_datetime(start_date)) & (df["TANGGAL"] <= pd.to_datetime(end_date))]
if search_query:
    filtered_df = filtered_df[
        filtered_df["JUDUL"].str.contains(search_query, case=False, na=False) |
        filtered_df["TEMA"].str.contains(search_query, case=False, na=False)
    ]

filtered_df = filtered_df.sort_values(by="TANGGAL", ascending=False)
filtered_df["TANGGAL_TAMPIL"] = filtered_df["TANGGAL"].dt.strftime("%d %b %Y")

# Header utama
st.title("Dashboard Konten Komdigi Newsroom")
if filtered_df.empty:
    st.warning("Tidak ada data yang ditemukan untuk filter ini.")
    st.image("https://via.placeholder.com/500x300.png?text=Data+Not+Found", use_column_width=True)
    st.stop()

if start_date != df["TANGGAL"].min() or end_date != df["TANGGAL"].max() or search_query:
    sub_header = f"Menampilkan data rentang waktu {start_date.strftime('%d %b %Y')} hingga {end_date.strftime('%d %b %Y')}"
    if search_query:
        sub_header += f", dengan kata kunci \"{search_query}\""
    st.subheader(sub_header)
    st.divider()

# Time Series Produksi Harian per Format
st.subheader("📌 Tren Produksi Harian per Format Konten")
if "FORMAT" in filtered_df.columns:
    produksi_harian = filtered_df.groupby(["TANGGAL", "FORMAT"]).size().reset_index(name="jumlah")
    fig_time_series = px.line(
        produksi_harian, x="TANGGAL", y="jumlah", color="FORMAT",
        labels={"TANGGAL": "Tanggal", "jumlah": "Jumlah Produksi", "FORMAT": "Format Konten"},
        markers=True,
    )
    st.plotly_chart(fig_time_series)
else:
    st.warning("Kolom 'FORMAT' tidak ditemukan di data.")

st.divider()
