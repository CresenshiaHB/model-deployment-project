🎬 Netflix Content-Based Recommendation SystemRepositori ini berisi implementasi end-to-end dari content-based recommendation system untuk film dan acara TV Netflix. Tujuannya adalah untuk merekomendasikan 5 judul yang paling mirip berdasarkan judul yang diinput oleh pengguna.Aplikasi interaktif ini dibangun menggunakan Scikit-learn untuk modelling dan Streamlit untuk deployment antarmuka web.📊 Dataset OverviewSumber: Netflix Movies and TV Shows Dataset (Kaggle)Ukuran: 8,800+ sampel film dan acara TV.Fitur Kunci yang Digunakan:FiturDeskripsititleJudul dari film atau acara TV.directorSutradara yang mengarahkan.castDaftar 3 aktor/aktris utama.listed_in (genres)Kategori/genre dari konten.descriptionSinopsis singkat dari konten.typeTipe konten (misalnya, Movie atau TV Show).⚙️ Methodology & Feature EngineeringModel ini murni berbasis konten. Rekomendasi dihasilkan dengan mengukur kesamaan (similaritas) antar judul berdasarkan atribut teks mereka. Ini dicapai melalui beberapa langkah preprocessing dan feature engineering.1. Data CleaningFitur-fitur tekstual utama (director, cast, description) memiliki missing values. Langkah pertama adalah mengisi nilai NaN dengan string kosong ('') untuk memastikan tidak ada data yang hilang selama proses penggabungan.2. Feature ExtractionFitur cast dan genres (dari listed_in) adalah string yang berisi banyak nama/kategori. Fitur-fitur ini diekstraksi dan dibersihkan untuk mengambil komponen yang paling penting.Cast: Hanya 3 aktor/aktris utama yang diambil. Spasi dihapus dari nama mereka (misal, "Tom Hanks" menjadi "tomhanks") untuk menghindari kesamaan yang tidak disengaja dengan nama lain (misal, "Tom Holland").Genres: Semua genre diekstraksi dan diformat dengan cara yang sama.# Contoh logika ekstraksi
df['cast'] = df['cast'].apply(lambda x: [i.strip().lower().replace(' ', '') for i in x.split(',')[:3]])
df['genres'] = df['listed_in'].fillna('').apply(lambda x: [g.strip().lower().replace(' ', '') for g in x.split(',')])
df['director'] = df['director'].apply(lambda x: [x.strip().lower().replace(' ', '')] if x else [])
3. "Content Soup" CreationFitur-fitur yang telah dibersihkan (genres, cast, director, type) digabungkan menjadi satu string besar yang disebut "content soup". String ini mewakili semua atribut konten dari sebuah judul dalam satu dokumen.# 'similarity' adalah "content soup"
def create_similarity(x):
    return ' '.join(x['genres'] + x['cast'] + x['director']) + ' ' + x['type']

df['similarity'] = df.apply(create_similarity, axis=1)
4. Vectorization & Similarity Model"Content soup" dari setiap judul kemudian diubah menjadi vektor numerik menggunakan TfidfVectorizer dari Scikit-learn, yang berfokus pada 5.000 kata kunci (fitur) yang paling penting.Setelah semua judul direpresentasikan sebagai vektor, matriks kesamaan (similarity matrix) dihitung menggunakan linear_kernel (sebuah cara cepat untuk menghitung Cosine Similarity), yang membandingkan setiap judul dengan setiap judul lainnya.from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Inisialisasi TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['similarity'])

# Hitung Cosine Similarity Matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
🚀 Application Architecture (Streamlit)Aplikasi web ini dibangun menggunakan Streamlit (inferencing.py) dan menggunakan class-based model (modelling.py) untuk menjaga kode tetap bersih dan terorganisir.modelling.py: Berisi class NetflixRecommender yang menangani semua logika pemuatan data, preprocessing, dan modelling.inferencing.py: Berisi logika UI Streamlit.netflix_df.pkl: Sebuah file pickle dari DataFrame yang telah diproses (setelah langkah 1 & 2) untuk mempercepat waktu startup aplikasi.Saat startup, aplikasi memuat netflix_df.pkl, kemudian class NetflixRecommender membangun model TF-IDF dan similarity matrix di dalam memori.# Logika inti di inferencing.py
@st.cache_resource
def load_model():
    # Load DataFrame yang sudah diproses
    with open("netflix_df.pkl", "rb") as f:
        df = pickle.load(f)
    
    # Inisialisasi model dan inject df
    model = NetflixRecommender("dummy.csv")
    model.df = df
    model.build_model() # Membangun TF-IDF & Cosine Sim
    return model

recommender = load_model()

# UI
if st.button("Show recommendation") and selected_title:
    # Panggil model untuk mendapatkan rekomendasi
    result = recommender.get_recommendations(selected_title, topn=5)
🛠️ Tech StackPythonPandas: Untuk manipulasi dan preprocessing data.Scikit-learn: Untuk TfidfVectorizer dan linear_kernel (Cosine Similarity).Streamlit: Untuk membangun dan menayangkan antarmuka web interaktif.Pickle: Untuk serialisasi dan desrialisasi DataFrame yang telah diproses.🏁 How to Run LocallyClone the repository:git clone [https://github.com/CresenshiaHB/model-deployment-project.git](https://github.com/CresenshiaHB/model-deployment-project.git)
cd model-deployment-project
Create a virtual environment (recommended):python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the dependencies:pip install -r requirements.txt
Run the Streamlit app:streamlit run inferencing.py
Buka browser Anda dan pergi ke http://localhost:8501.
