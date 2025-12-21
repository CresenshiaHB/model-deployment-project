import streamlit as st
import pickle
from modelling import NetflixRecommender

# --- Load preprocessed dataframe from pickle ---
@st.cache_resource
def load_model():
    # Load DataFrame dari pickle
    with open("netflix_df.pkl", "rb") as f:
        df = pickle.load(f)
    
    # Buat objek NetflixRecommender dan inject df
    model = NetflixRecommender("dummy.csv")  # path tidak dipakai
    model.df = df
    model.build_model()
    return model

recommender = load_model()

# --- Streamlit UI ---
st.title("🎬 Netflix Content-Based Recommender")

# Film Titles
movie_titles = recommender.df['title'].values

# Input Box
selected_title = st.selectbox(
    "Type to search movie:",
    options=movie_titles,
    index=None
)

# Button to Show The Output
if st.button("Show recommendation") and selected_title:
    result = recommender.get_recommendations(selected_title, topn=5)

    if isinstance(result, str):
        st.warning(result)  # kalau judul tidak ditemukan
    else:
        st.success(f"You might like this also")
        for _, row in result.iterrows():
            st.markdown(f"### 🎥 {row['title']} ({row['release_year']})")
            st.markdown(f"*Type: {row['type']}  \nGenre*: {', '.join(row['genres'])}")
            st.markdown(f"{row['description']}")

            st.markdown("---")

