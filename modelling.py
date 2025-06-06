import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class NetflixRecommender:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.cosine_sim = None
        self.indices = None

    def load_and_preprocess_data(self):
        self.df = pd.read_csv(self.file_path)
        self.df['description'] = self.df['description'].fillna('')
        self.df['cast'] = self.df['cast'].fillna('')
        self.df['director'] = self.df['director'].fillna('')
        self.df['cast'] = self.df['cast'].apply(lambda x: [i.strip().lower().replace(' ', '') for i in x.split(',')[:3]])
        self.df['genres'] = self.df['listed_in'].fillna('').apply(lambda x: [g.strip().lower().replace(' ', '') for g in x.split(',')])
        self.df['director'] = self.df['director'].apply(lambda x: [x.strip().lower().replace(' ', '')] if x else [])

    def create_similarity(self, x):
        return ' '.join(x['genres'] + x['cast'] + x['director']) + ' ' + x['type']

    def build_model(self):
        if self.df is None:
            self.load_and_preprocess_data()
        
        self.df['similarity'] = self.df.apply(self.create_similarity, axis=1)
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(self.df['similarity'])
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        self.indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates()

    def get_recommendations(self, title, topn=5):
        if self.cosine_sim is None:
            self.build_model()
            
        idx = self.indices.get(title)
        if idx is None:
            return f"{title}** tidak ditemukan"
        
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:topn+1]
        rec_idxs = [i[0] for i in sim_scores]
        
        return self.df[['title', 'type', 'release_year', 'description', 'genres']].iloc[rec_idxs]

    def save_df_to_pickle(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.df, file)


if __name__ == "__main__":   
    recommender = NetflixRecommender("C:/Users/cheryl/OneDrive - Bina Nusantara/Documents/sem 4/modep/Final Project/netflix_titles.csv")
    recommender.build_model()
    
    recommendations = recommender.get_recommendations('Stranger Things', topn=5)
    print("Rekomendasi untuk 'Stranger Things':")
    print(recommendations)
    
    recommender.save_df_to_pickle('netflix_df.pkl')