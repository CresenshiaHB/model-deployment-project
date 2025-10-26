🎬 Netflix Content-Based Recommendation System

This project is a content-based recommendation system for Netflix movies and TV shows, built with Python and deployed as an interactive web application using Streamlit.

Given a movie or TV show title, the model will recommend 5 other titles from the Netflix dataset based on content similarity (genres, cast, director, and description).

🚀 Features

Content-Based Filtering: Recommends items by finding similarities in their content (text-based attributes).

Interactive Web UI: Built with Streamlit for an easy-to-use interface.

Real-time Recommendations: Type in a title and get recommendations instantly.

Modular Code: Logic is separated into modelling.py (for the recommendation logic) and inferencing.py (for the Streamlit app).

🛠️ Tech Stack

Python

Pandas: For data loading and preprocessing.

Scikit-learn: For TfidfVectorizer (to convert text to a matrix) and linear_kernel (to compute similarity scores).

Streamlit: To build and serve the interactive web application.

Pickle: For saving and loading the preprocessed DataFrame.

📂 File Structure

.
├── inferencing.py       # Main Streamlit application file
├── modelling.py         # Contains the NetflixRecommender class
├── netflix_df.pkl       # Preprocessed DataFrame
├── netflix_titles.csv   # The original raw dataset
├── requirements.txt     # Python dependencies
└── README.md            # You are here!


🏁 How to Run Locally

Clone the repository:

git clone [https://github.com/CresenshiaHB/model-deployment-project.git](https://github.com/CresenshiaHB/model-deployment-project.git)
cd model-deployment-project


Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install the dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run inferencing.py


Open your browser and go to http://localhost:8501.
