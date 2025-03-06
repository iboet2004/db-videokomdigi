import json
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk


# Konfigurasi koneksi ke Google Spreadsheet
SHEET_URL = "https://docs.google.com/spreadsheets/d/1AqC7MXO-n4CFkKrf5WDdf1cU1HjZZ-CyQtBirmCBRNk/edit?gid=878595289"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_file("credentials.json", scopes=scope)
client = gspread.authorize(creds)
spreadsheet = client.open_by_url(SHEET_URL)
worksheet = spreadsheet.worksheet("dataset")
data = worksheet.get_all_records()
df = pd.DataFrame(data)

# Load credentials dari Streamlit Secrets
creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
creds = Credentials.from_service_account_info(creds_dict)

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
st.title("Dashboard Video Komdigi Newsroom")
if start_date != df["TANGGAL"].min() or end_date != df["TANGGAL"].max() or search_query:
    sub_header = f"Menampilkan data rentang waktu {start_date.strftime('%d %b %Y')} hingga {end_date.strftime('%d %b %Y')}"
    if search_query:
        sub_header += f", dengan kata kunci \"{search_query}\""
    st.subheader(sub_header)
st.divider()

# Word Cloud
text_data = " ".join(filtered_df["JUDUL"].astype(str) + " " + filtered_df["ATRIBUSI"].astype(str))
custom_stopwords = STOPWORDS.union({"pastikan", "bisa", "tak", "Jadi", "unknown", "di", "ke", "Ini", "bagi", "resmi", "siap", "dapat", "akan", "dan", "atau", "yang", "untuk", "dalam", "dengan", "pada"})
wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='coolwarm', contour_color='white', contour_width=2, stopwords=custom_stopwords).generate(text_data)

st.subheader("☁️ Word Cloud Topik Utama")
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)
st.divider()

# Heatmap Tren Tema
filtered_df["MINGGU"] = filtered_df["TANGGAL"].dt.to_period("W").astype(str)
topic_counts = filtered_df.groupby(["MINGGU", "TEMA"]).size().reset_index(name='jumlah')
top_topics = topic_counts.groupby("TEMA")["jumlah"].sum().nlargest(10).index
topic_counts = topic_counts[topic_counts["TEMA"].isin(top_topics)]

fig_heatmap = px.density_heatmap(
    topic_counts, x="MINGGU", y="TEMA", z="jumlah", 
    labels={"jumlah": "Jumlah Video", "MINGGU": "Minggu", "TEMA": "Topik"},
    color_continuous_scale="plasma"
)
st.subheader("🔥 Heatmap - Tren Dominasi Tema dari Waktu ke Waktu")
st.plotly_chart(fig_heatmap)
st.divider()

# Scatter Plot Penyebutan Narasumber
atribusi_counts = filtered_df.groupby(["TANGGAL", "ATRIBUSI"]).size().reset_index(name="jumlah")
top_atribusi = atribusi_counts.groupby("ATRIBUSI")["jumlah"].sum().nlargest(10).index  
filtered_atribusi = atribusi_counts[atribusi_counts["ATRIBUSI"].isin(top_atribusi)]

fig_scatter = px.scatter(
    filtered_atribusi, x="TANGGAL", y="ATRIBUSI", size="jumlah", color="ATRIBUSI",
    labels={"jumlah": "Jumlah Penyebutan", "TANGGAL": "Tanggal", "ATRIBUSI": "Narasumber"},
)
fig_scatter.update_layout(showlegend=False)

st.subheader("📌 Scatter Plot Tren Penyebutan Narasumber")
st.plotly_chart(fig_scatter)
st.divider()

# Download VADER Lexicon (hanya perlu sekali)
nltk.download('vader_lexicon')

# Inisialisasi Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Tambahkan kolom sentimen berdasarkan judul
def get_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positif"
    elif score <= -0.05:
        return "Negatif"
    else:
        return "Netral"

# Terapkan sentimen ke judul video
filtered_df["Sentimen"] = filtered_df["JUDUL"].astype(str).apply(get_sentiment)

# Tampilkan hasil di Streamlit
st.subheader("📊 Analisis Sentimen pada Judul Video")
sentiment_counts = filtered_df["Sentimen"].value_counts().reset_index()
sentiment_counts.columns = ["Sentimen", "Jumlah"]

fig_sentiment = px.bar(
    sentiment_counts, x="Sentimen", y="Jumlah", color="Sentimen",
    title="Distribusi Sentimen dalam Judul Video",
    labels={"Jumlah": "Jumlah Video", "Sentimen": "Kategori Sentimen"},
    color_discrete_map={"Positif": "green", "Netral": "gray", "Negatif": "red"}
)
st.plotly_chart(fig_sentiment)

# Tampilkan tabel dengan kolom sentimen
st.subheader("📄 Data dengan Sentimen")
st.dataframe(filtered_df[["TANGGAL_TAMPIL", "JUDUL", "TEMA", "Sentimen"]])
