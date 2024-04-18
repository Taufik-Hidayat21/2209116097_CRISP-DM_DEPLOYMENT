import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import joblib
import sklearn

print(sklearn.__version__)


@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    return df

# Fungsi untuk menampilkan halaman utama
def show_about():
    st.title("Spotify Track Popularity Analysis")
    st.write("""

# Data Understanding

## Business Understanding

### Business objective
Tujuan utama analisis pada dataset ini adalah meningkatkan popularitas dan potensi lagu-lagu baru yang akan dirilis oleh musisi. Analisis dilakukan dengan merujuk pada popularitas lagu-lagu yang telah ada di platform Spotify, dengan mempertimbangkan berbagai faktor penilaian yang telah ditentukan untuk setiap lagu. Dengan memahami karakteristik dan tren dari lagu-lagu yang sukses dan populer, kita dapat mengembangkan strategi yang lebih terarah untuk mengoptimalkan potensi keberhasilan lagu-lagu baru di pasar musik yang kompetitif.

### Assess situation
Analisis ini didasari oleh dorongan untuk memperluas popularitas seorang artis musik melalui penilaian mendalam terhadap lagu-lagu yang akan dirilis. Pendekatan ini didasarkan pada kebutuhan mendalam untuk memahami selera dan harapan pendengar, dengan harapan dapat menciptakan karya yang tidak hanya memenuhi ekspektasi pendengar, tetapi juga meraih daya tarik luas. Dengan menganalisis kebutuhan pendengar secara cermat, kita dapat merancang strategi yang lebih terfokus untuk meningkatkan daya tarik setiap rilisan musik yang akan datang.

### Tujuan data mining
Tujuan dari analisis data ini adalah untuk menentukan pola-pola dari setiap lagu berdasarkan faktor-faktor yang mempengaruhi sebuah lagu yang populer dan trending di platform Spotify seperti durasi, tempo, dan lain-lain. Selanjutnya, rancangan kesuksesan lagu akan dibangun, memberikan pandangan yang dapat membantu dalam mengidentifikasi kesuksesan yang mungkin dapat diadaptasi sebagai strategi pembuatan dan pemasaran lagu-lagu baru di platform Spotify. Dengan demikian, analisis data ini diharapkan dapat memberikan wawasan yang mendalam dan bermanfaat untuk mendukung pengambilan keputusan terkait strategi musik khususnya di platform Spotify.

### Rencana proyek
Proyek ini diawali dengan tahap pengumpulan data dari sumber yang tersedia, yang melibatkan penggalian dataset yang relevan. Setelah itu, karakteristik dataset akan dijelaskan dengan mengidentifikasi variabel kunci terkait karakteristik lagu dan popularitas. Selanjutnya, kita akan melihat data lebih detail untuk mengidentifikasi pola-pola dan tren yang ada. Setelah itu, fokus akan beralih ke tahap preprocessing dan penilaian kualitas data dan analisis lebih lanjut akan dilakukan untuk mengidentifikasi korelasi dan hubungan antara fitur-fitur lagu dan popularitasnya. Pada tahap inti, model prediktif akan dikembangkan untuk memprediksi popularitas lagu, kemudian hasil dari model yang telah diimplementasikan akan digunakan untuk merumuskan rekomendasi dan mengembangkan strategi yang dapat meningkatkan popularitas lagu.
    """)
    # Path ke file CSV
    file_path = 'dataset.csv'

    # Periksa keberadaan file
    if not os.path.exists(file_path):
        st.error(f"File '{file_path}' tidak ditemukan. Pastikan file berada di lokasi yang benar.")
        st.stop()

    # Muat data CSV
    try:
        df = pd.read_csv(file_path)
        st.write(df)  # Tampilkan data jika berhasil dimuat
    except Exception as e:
        st.error(f"Gagal memuat file CSV: {e}")

# Fungsi untuk menampilkan halaman tentang
def show_Distribusi(df):
# Judul dan deskripsi
    st.title("Track Genre With High Popularity")
    

    
    top_10_genres = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False).head(10).index
    df_top_10 = df[df['track_genre'].isin(top_10_genres)]

    # Create the bar plot without error bars and in descending order
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(x='popularity', y='track_genre', data=df_top_10, ci=None,
                order=top_10_genres,
                ax=ax)
    
    
    # Set labels and title
    ax.set_xlabel('Popularity')
    ax.set_ylabel('Track Genre')
    ax.set_title('Top 10 Track Genres by Popularity')

    # Show the plot in Streamlit
    st.pyplot(fig)

    st.subheader("Interpretation")
    st.write("Dari sekian banyak genre yang ada pada Spotify, yang memiliki tingkat popularitas yang tinggi dengan nilai popularity > 40 adalah 10 genre seperti yang ada di barplot diatas.")

    st.subheader("Insight")
    st.write("Rata-rata pengguna spotify memilih untuk lebih banyak mendengarkan 10 top genre yang ada pada barplot diatas.")

    st.subheader("Actionable Insight")
    st.write("Artis yang ingin memulai karir musik dapat mempertimbangkan 10 top genre yang ada tersebut jika ingin meningkatkan popularitas musik yang akan dibuatnya nanti.")

    st.title("Average Danceability Within Top 10 Track Genre")
    # Calculate the top 10 genres by popularity
    top_10_genres = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False).head(10).index

    # Filter the DataFrame to include only the top 10 genres
    df_top_10 = df[df['track_genre'].isin(top_10_genres)]

    # Create a bar plot for average danceability for each genre
    plt.figure(figsize=(12, 8))
    sns.barplot(x='track_genre', y='danceability', data=df_top_10, palette='Set3', ci=None)
    plt.xlabel('Track Genre')
    plt.ylabel('Average Danceability')
    plt.title('Average Danceability across Top 10 Genres with High Popularity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Show the plot in Streamlit
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='track_genre', y='danceability', data=df_top_10, palette='Set3', ci=None, ax=ax)
    ax.set_xlabel('Track Genre')
    ax.set_ylabel('Average Danceability')
    ax.set_title('Average Danceability across Top 10 Genres with High Popularity')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    st.subheader("Interpretation")
    st.write("Pada gambar diatas merupakan nilai Danceability dari Top 10 Track Genre dengan nilai Popularity tertinggi.")

    st.subheader("Insight")
    st.write("Dari gambar diatas dapat diketahui bahwa dari top 10 nilai Danceability berkisar mulai dari > 0.4 hingga < 0.7. Dan juga genre dengan Danceability tertinggi dari Top 10 tersebut  adalah sad, chill, dan k-pop.")

    st.subheader("Actionable Insight")
    st.write("Berdasarkan insight yang didapat  diatas dapat disimpulkan untuk meningkatkan nilai popularitas dari lagu yang akan dibuat nanti dapat mempertimbangkan untuk nilai dari Danceability mulai dari 0.4 hinga 0.7 atau dengan mempertimbangkan menggunakan unsur genre sad, chill, atau k-pop. Dengan berikut kemungkinan dari tingginya popularitas lagu yang akan dirilis dapat meningkat.")

def show_hubungan(df):
    st.title("Distribution of Valence")
    top_10_genres = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False).head(10).index

    # Filter the DataFrame to include only the top 10 genres
    df_top_10 = df[df['track_genre'].isin(top_10_genres)]
    fig, ax = plt.subplots()
    ax.hist(df_top_10["valence"])
    ax.set_xlabel("Valence")
    ax.set_ylabel("Number of Tracks")
    ax.set_title("Distribution of Valence in Tracks")
    st.pyplot(fig)

    st.subheader("Interpretation")
    st.write("Pada gambar diatas menunjukan distribusi nilai Valence dari lagu-lagu yang berada di Top 10 Genre dengan nilai Valence yang berkisar mulai dari 0.0 hingga 1.0")

    st.subheader("Insight")
    st.write("Dari gambar diatas dapat diketahui bahwa distribusi nilai Valence menunjukkan distribusi yang normal, yang berarti bahwa tingkat positifitas pada lagu-lagu yang berada di Top 10 Genre tidak memiliki kecenderungan yang tinggi maupun rendah.")

    st.subheader("Actionable Insight")
    st.write("Berdasarkan insight yang didapat  diatas dapat disimpulkan untuk meningkatkan nilai popularitas dari lagu yang akan dibuat nanti dapat mempertimbangkan untuk membuat lagu yang tidak terlalu condong ke arah terlalu sedih maupun terlalu bahagia. Namun, bisa menggunakan lagu dengan nilai valence mulai dari > 0.2 hingga < 0.8, atau lagu yang dibuat nanti bisa mengkombinasikan emosi sedih dan bahagia pada lagu yang akan dibuat nantinya.")

def show_Perbandingan(df):
    st.title("Relationship Between Danceablity and Valence")
    sampeldf = df.sample(n=5000)
    top_10_genres = sampeldf.groupby('track_genre')['popularity'].mean().sort_values(ascending=False).head(10).index

    # Filter the DataFrame to include only the top 10 genres
    df_top_10 = sampeldf[sampeldf['track_genre'].isin(top_10_genres)]
    st.header("Tempo vs. Valence")
    # Assuming your data is in a DataFrame named 'df'
    # Make sure the DataFrame has columns for 'danceability' and 'valence'

    # Calculate a linear regression line (optional)
    # Replace 'm' and 'b' with the actual slope and intercept values if you have them
    m = 0.5  # Example slope (replace with your calculated value)
    b = 0.2  # Example intercept (replace with your calculated value)

    # Create the scatter plot
    st.header("Danceability vs. Valence")
    fig, ax = plt.subplots()
    ax.scatter(df_top_10["danceability"], df_top_10["valence"])

    from sklearn.linear_model import LinearRegression  # Uncomment if using linear regression
    model = LinearRegression()  # Uncomment if using linear regression
    model.fit(df_top_10[['danceability']], df_top_10['valence'])  # Uncomment if using linear regression
    m = model.coef_[0]  # Uncomment if using linear regression
    b = model.intercept_  # Uncomment if using linear regression
    ax.plot(df_top_10["danceability"], m * df_top_10["danceability"] + b, color='red')  # Line from regression (uncomment if using)

    ax.set_xlabel("Danceability")
    ax.set_ylabel("Valence")
    ax.set_title("Relationship Between Danceability and Valence")
    st.pyplot(fig)

    st.subheader("Interpretation")
    st.write("Pada gambar diatas menunjukkan scatter plot tentang hubungan antara danceability dan valence. Tiap poin merepresentasikan lagu dimana memberikan nilai danceability dan valence pada posisinya.")

    st.subheader("Insight")
    st.write("Dari gambar diatas dapat diketahui bahwa hubungan antara nilai danceability dan valence memiliki korelasi yang positif, dimana hal itu mengindikasikan bahwa lagu dengan nilai danceability yang tinggi cenderung memiliki nilai valence yang tinggi pula. Hal ini dapat menjadi poin bahwa dengan tinggi nya danceability dari sebuah lagu maka positifitas dari lagu tersebut cenderung lebih tinggi.")

    st.subheader("Actionable Insight")
    st.write("Berdasarkan insight yang didapat  diatas dapat disimpulkan jika lagu yang akan dibuat diperuntukkan untuk audiens yang menikmati musik dengan energi yang tinggi maka lagu yang akan dibuat nantinya sebaiknya memfokuskan pada nilai danceability yang tinggi serta nilai valence yang tinggi pula.")


def predict_cancellation(df):
    # # Select features
    # feature_columns = [
    #     'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
    #     'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 
    #     'density', 'pH', 'sulphates', 'alcohol'
    # ]

    # selected_features = {}
    # for feature in feature_columns:
    #     selected_features[feature] = st.selectbox(f"{feature.replace('_', ' ').title()}", sorted(df[feature].unique()))

    # data = pd.DataFrame(selected_features, index=[0])

    # # Ensure all columns in data are numeric
    # data = data.astype(float)

    # # Button for prediction
    # button = st.button('Prediksi')
    # if button:
    #     try:
    #         loaded_model = joblib.load('model.pkl')
    #         predicted = loaded_model.predict(data)
    #         print(predicted)
    #         if predicted[0] == 1:
    #             st.write('buruk')
    #         elif predicted[0] == 2:
    #             st.write("baik")
    #         elif predicted[0] == 3:
    #             st.write("sangat baik")
    #         else:
    #             st.write('Dibatalkan')
    #     except FileNotFoundError:
    #         st.write("Model tidak ditemukan. Silakan pastikan bahwa model sudah tersedia.")
    with open('spotify_model.pkl', 'rb') as f:
        spotify_model = pickle.load(f)
    st.title('Data Mining Popularity Predict')

    popularity = st.text_input('Input nilai popularity')
    duration_ms = st.text_input('Input nilai duration_ms')
    danceability = st.text_input('Input nilai danceability')
    energy = st.text_input('Input nilai energy')
    key = st.text_input('Input nilai key')
    loudness = st.text_input('Input nilai loudness')
    mode = st.text_input('Input nilai mode')
    speechiness = st.text_input('Input nilai speechiness')
    acousticness = st.text_input('Input nilai acousticness')
    instrumentalness = st.text_input('Input nilai instrumentalness')
    liveness = st.text_input('Input nilai liveness')
    valence = st.text_input('Input nilai valence')
    tempo = st.text_input('Input nilai tempo')
    time_signature = st.text_input('Input nilai time_signature')
    tempo_type = st.text_input('Input nilai tempo_type')
    

    if st.button('Prediksi Tipe Popularitas'):
        # Check if all input fields are filled
        if popularity and duration_ms and danceability and energy and key and loudness and mode and speechiness and acousticness and instrumentalness and liveness and valence and tempo and time_signature and tempo_type:
            # Prepare input for prediction
            input_data = [[popularity, duration_ms, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature, tempo_type]]
            
            # Perform prediction
            type_popular = spotify_model.predict(input_data)
            
            # Map prediction to "good" or "bad"
            prediction_result = "low" if type_popular[0] == 1 else "high"
            
            # Display prediction
            st.write(f'Predicted popularity type: {prediction_result}')
        else:
            st.error("Mohon isi semua nilai input sebelum melakukan prediksi.")

# Assuming df is your DataFrame
# df = pd.read_csv('hotel_data.csv')  # Load your data here
# predict_cancellation(df)

# Memuat data
df = load_data()

# Mengatur sidebar
df2 = pd.read_csv('DataCleanedNew.csv')
nav_options = {
    "About": show_about,
    "Comparison": lambda: show_Distribusi(df),
    "Distribution": lambda: show_hubungan(df),
    "Relationship": lambda: show_Perbandingan(df),
    "Predict": lambda: predict_cancellation(df2)
}

# Menampilkan sidebar
st.sidebar.title("Spotify Track Popularity Analysis")
selected_page = st.sidebar.radio("Menu", list(nav_options.keys()))

# Menampilkan halaman yang dipilih
nav_options[selected_page]()