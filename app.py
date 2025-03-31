import streamlit as st
import pickle
import sqlite3
import pandas as pd
import numpy as np
import re
import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import hashlib
import os
from datetime import datetime

# Download NLTK resources if not already present
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

download_nltk_resources()

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Function for text cleaning
def clean_text(text, remove_stopwords=True, use_stemming=False, use_lemmatization=True):
    text = text.lower() # to lowercase
    text = re.sub(r'http\S+', '', text) # remove URLs
    text = re.sub(r'www\S+', '', text) # remove URLs
    text = re.sub(r'\s+', ' ', text) # remove extra spaces
    text = re.sub(r'[^\w\s]', '', text) # remove punctuation
    text = re.sub(r'\d+', '', text) # remove numbers
    
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
    
    if use_stemming:
        text = ' '.join([stemmer.stem(word) for word in text.split()])
    
    if use_lemmatization:
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    
    return text

# Function to find the Empirical Proportion
def eval_empirical_proportion(text):
    words = [word for word in text.split() if word.strip()]
    first_letters = [word[0].upper() for word in words if word[0].upper() in list(string.ascii_uppercase)]
    letter_counts = Counter(first_letters)
    total_letters = sum(letter_counts.values())
    
    if total_letters == 0:
        return pd.Series([0] * 26)
    
    letter_proportions = {letter: 0 for letter in string.ascii_uppercase}
    for letter, count in letter_counts.items():
        letter_proportions[letter] = count / total_letters
    
    sorted_proportions = dict(sorted(letter_proportions.items(), key=lambda item: item[1], reverse=True))
    return pd.Series(list(sorted_proportions.values()))

# Database functions
def init_db():
    conn = sqlite3.connect('fake_news_detector.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    ''')
    
    # Create history table
    c.execute('''
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        title TEXT,
        content TEXT,
        prediction REAL,
        date TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create saved articles table
    c.execute('''
    CREATE TABLE IF NOT EXISTS saved_articles (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        title TEXT,
        content TEXT,
        prediction REAL,
        date TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()

def register_user(username, password):
    conn = sqlite3.connect('fake_news_detector.db')
    c = conn.cursor()
    
    # Hash the password
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        result = True
    except sqlite3.IntegrityError:
        result = False
    
    conn.close()
    return result

def login_user(username, password):
    conn = sqlite3.connect('fake_news_detector.db')
    c = conn.cursor()
    
    # Hash the password
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    c.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, hashed_password))
    user = c.fetchone()
    
    conn.close()
    
    if user:
        return user[0]
    else:
        return None

def save_to_history(user_id, title, content, prediction):
    conn = sqlite3.connect('fake_news_detector.db')
    c = conn.cursor()
    
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute("INSERT INTO history (user_id, title, content, prediction, date) VALUES (?, ?, ?, ?, ?)",
              (user_id, title, content, prediction, date))
    
    conn.commit()
    conn.close()

def save_article(user_id, title, content, prediction):
    conn = sqlite3.connect('fake_news_detector.db')
    c = conn.cursor()
    
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute("INSERT INTO saved_articles (user_id, title, content, prediction, date) VALUES (?, ?, ?, ?, ?)",
              (user_id, title, content, prediction, date))
    
    conn.commit()
    conn.close()

def get_history(user_id):
    conn = sqlite3.connect('fake_news_detector.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("SELECT * FROM history WHERE user_id = ? ORDER BY date DESC", (user_id,))
    history = [dict(row) for row in c.fetchall()]
    
    conn.close()
    return history

def get_saved_articles(user_id):
    conn = sqlite3.connect('fake_news_detector.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("SELECT * FROM saved_articles WHERE user_id = ? ORDER BY date DESC", (user_id,))
    articles = [dict(row) for row in c.fetchall()]
    
    conn.close()
    return articles

def delete_saved_article(article_id):
    conn = sqlite3.connect('fake_news_detector.db')
    c = conn.cursor()
    
    c.execute("DELETE FROM saved_articles WHERE id = ?", (article_id,))
    
    conn.commit()
    conn.close()

# Load ML models
@st.cache_resource
def load_models():
    import joblib
    models_path = 'models'
    
    # Check if models directory exists
    if not os.path.exists(models_path):
        st.error(f"Models directory '{models_path}' not found. Please create it and add the required model files.")
        return None, None
    
    try:
        # Load the combined model
        rf_combined_path = os.path.join(models_path, 'rf_combined.joblib')
        if not os.path.exists(rf_combined_path):
            st.error(f"Model file 'rf_combined.joblib' not found in models directory.")
            rf_combined_model = None
        else:
            rf_combined_model = joblib.load(rf_combined_path)
        
        # Load TF-IDF vectorizer
        tfidf_path = os.path.join(models_path, 'tfidf.joblib')
        if not os.path.exists(tfidf_path):
            st.error(f"Model file 'tfidf.joblib' not found in models directory.")
            tfidf_vectorizer = None
        else:
            tfidf_vectorizer = joblib.load(tfidf_path)
        
        return rf_combined_model, tfidf_vectorizer
    
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.info("There was an issue loading the model files. Please ensure they're properly formatted and accessible.")
        return None, None

# Predict function
def predict_fake_news(title, content):
    rf_combined_model, tfidf_vectorizer = load_models()
    
    # Combine title and content
    combined_text = f"{title} {content}"
    
    # Clean the text
    cleaned_text = clean_text(combined_text)
    
    # Get TF-IDF features
    tfidf_features = tfidf_vectorizer.transform([cleaned_text])
    tfidf_features_dense = tfidf_features.toarray()
    
    # Get empirical proportion features
    emp_features = eval_empirical_proportion(cleaned_text).values.reshape(1, -1)
    
    # Combine features
    combined_features = np.hstack((tfidf_features_dense, emp_features))
    
    # Predict using the combined model
    prediction = rf_combined_model.predict_proba(combined_features)[0][1]  # Probability of being fake (class 1)
    
    # Convert to "realness" score (0=fake, 1=real)
    realness_score = prediction
    
    return realness_score

# Logo display function
def display_logo():
    logo_path = "logo.png"
    
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    else:
        st.warning("Logo file not found. Please add 'logo.png' to the root directory.")

# App title
def app_header():
    col1, col2 = st.columns([1, 4])
    with col1:
        display_logo()
    with col2:
        st.title("Fake News Detector")

# Initialize database
init_db()

# Main app
def main():
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = "check_article"
    
    if "current_title" not in st.session_state:
        st.session_state.current_title = ""
    
    if "current_content" not in st.session_state:
        st.session_state.current_content = ""
    
    if "current_prediction" not in st.session_state:
        st.session_state.current_prediction = None
    
    # If user is not logged in, show login/signup page
    if st.session_state.user_id is None:
        login_signup_page()
    else:
        # User is logged in, show main application
        app_header()
        
        # Sidebar for navigation
        with st.sidebar:
            display_logo()
            st.title("Navigation")
            
            if st.button("Check Article", use_container_width=True):
                st.session_state.current_page = "check_article"
                st.rerun()
            
            if st.button("History", use_container_width=True):
                st.session_state.current_page = "history"
                st.rerun()
            
            if st.button("Saved Articles", use_container_width=True):
                st.session_state.current_page = "saved_articles"
                st.rerun()
            
            st.markdown("---")
            if st.button("Sign Out", use_container_width=True):
                st.session_state.user_id = None
                st.session_state.current_page = "check_article"
                st.rerun()
        
        # Main content based on selected page
        if st.session_state.current_page == "check_article":
            check_article_page()
        elif st.session_state.current_page == "history":
            history_page()
        elif st.session_state.current_page == "saved_articles":
            saved_articles_page()

def login_signup_page():
    col1, col2 = st.columns([1, 1])
    
    with col1:
        display_logo()
    
    with col2:
        st.title("Fake News Detector")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="login_button"):
            if username and password:
                user_id = login_user(username, password)
                if user_id:
                    st.session_state.user_id = user_id
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            else:
                st.warning("Please enter both username and password")
    
    with tab2:
        st.subheader("Sign Up")
        new_username = st.text_input("Username", key="signup_username")
        new_password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        if st.button("Sign Up", key="signup_button"):
            if new_username and new_password and confirm_password:
                if new_password == confirm_password:
                    if register_user(new_username, new_password):
                        st.success("Account created successfully! Please login.")
                    else:
                        st.error("Username already exists")
                else:
                    st.error("Passwords do not match")
            else:
                st.warning("Please fill in all fields")

def check_article_page():
    st.header("Check Article")
    
    # Input fields
    title = st.text_input("Article Title", value=st.session_state.current_title)
    content = st.text_area("Article Content", height=200, value=st.session_state.current_content)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Analyze", use_container_width=True):
            if title and content:
                with st.spinner("Analyzing article..."):
                    prediction = predict_fake_news(title, content)
                    st.session_state.current_title = title
                    st.session_state.current_content = content
                    st.session_state.current_prediction = prediction
                    
                    # Save to history
                    save_to_history(st.session_state.user_id, title, content, prediction)
                    
                    st.rerun()
            else:
                st.warning("Please enter both title and content")
    
    with col2:
        if st.session_state.current_prediction is not None:
            if st.button("Save Article", use_container_width=True):
                save_article(
                    st.session_state.user_id,
                    st.session_state.current_title,
                    st.session_state.current_content,
                    st.session_state.current_prediction
                )
                st.success("Article saved successfully!")
    
    # Show prediction results
    if st.session_state.current_prediction is not None:
        st.markdown("---")
        st.subheader("Analysis Results")
        
        score = st.session_state.current_prediction
        
        # Create a colored meter to visualize the score
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Determine color based on score (red for fake, green for real)
            color = f"rgba({int(255 * (1 - score))}, {int(255 * score)}, 0, 0.8)"
            
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="width: 100%; height: 30px; background: linear-gradient(to right, #ff0000, #ffff00, #00ff00); border-radius: 5px;">
                    <div style="position: relative; left: {score * 100}%; transform: translateX(-50%); width: 10px; height: 40px; background-color: black;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                    <span>Fake (0.0)</span>
                    <span>Real (1.0)</span>
                </div>
                <h2 style="margin-top: 10px; color: {color};">Score: {score:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Classification based on score
        st.markdown("### Interpretation")
        if score < 0.3:
            st.error("This article appears to be highly unreliable and likely contains misinformation.")
        elif score < 0.6:
            st.warning("This article contains some questionable elements. Verify with other sources.")
        else:
            st.success("This article appears to be from a reliable source with credible information.")

def history_page():
    st.header("History")
    
    history = get_history(st.session_state.user_id)
    
    if not history:
        st.info("No history found. Check some articles first!")
        return
    
    for item in history:
        with st.expander(f"{item['title']} - Score: {item['prediction']:.2f} - {item['date']}"):
            st.write(f"**Content:** {item['content'][:200]}...")
            
            # Score display
            score = item['prediction']
            color = f"rgba({int(255 * (1 - score))}, {int(255 * score)}, 0, 0.8)"
            
            st.markdown(f"""
            <div style="width: 100%; height: 20px; background: linear-gradient(to right, #ff0000, #ffff00, #00ff00); border-radius: 5px;">
                <div style="position: relative; left: {score * 100}%; transform: translateX(-50%); width: 8px; height: 28px; background-color: black;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                <span style="font-size: 0.8em;">Fake (0.0)</span>
                <span style="font-size: 0.8em;">Real (1.0)</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Button to load article for re-checking
            if st.button("Load this article", key=f"load_{item['id']}"):
                st.session_state.current_title = item['title']
                st.session_state.current_content = item['content']
                st.session_state.current_prediction = None
                st.session_state.current_page = "check_article"
                st.rerun()

def saved_articles_page():
    st.header("Saved Articles")
    
    articles = get_saved_articles(st.session_state.user_id)
    
    if not articles:
        st.info("No saved articles found. Save some articles first!")
        return
    
    for item in articles:
        with st.expander(f"{item['title']} - Score: {item['prediction']:.2f} - {item['date']}"):
            st.write(f"**Content:** {item['content'][:200]}...")
            
            # Score display
            score = item['prediction']
            color = f"rgba({int(255 * (1 - score))}, {int(255 * score)}, 0, 0.8)"
            
            st.markdown(f"""
            <div style="width: 100%; height: 20px; background: linear-gradient(to right, #ff0000, #ffff00, #00ff00); border-radius: 5px;">
                <div style="position: relative; left: {score * 100}%; transform: translateX(-50%); width: 8px; height: 28px; background-color: black;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                <span style="font-size: 0.8em;">Fake (0.0)</span>
                <span style="font-size: 0.8em;">Real (1.0)</span>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Button to load article for re-checking
                if st.button("Load this article", key=f"load_saved_{item['id']}"):
                    st.session_state.current_title = item['title']
                    st.session_state.current_content = item['content']
                    st.session_state.current_prediction = None
                    st.session_state.current_page = "check_article"
                    st.rerun()
            
            with col2:
                # Button to delete saved article
                if st.button("Delete", key=f"delete_{item['id']}"):
                    delete_saved_article(item['id'])
                    st.success("Article deleted successfully!")
                    st.rerun()

if __name__ == "__main__":
    main()
