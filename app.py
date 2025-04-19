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
    
def remove_reuters(text):
  text = re.sub(r'reuters', '', text)
  text = re.sub(r'said', '', text)
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
        rf_path = os.path.join(models_path, 'rf.joblib')
        rf_fll_path = os.path.join(models_path, 'rf_fll.joblib')
        if not os.path.exists(rf_combined_path):
            st.error(f"Model file 'rf_combined.joblib' not found in models directory.")
            rf_combined_model = None
        else:
            rf_combined_model = joblib.load(rf_combined_path)
            
        if not os.path.exists(rf_path):
            st.error(f"Model file 'rf.joblib' not found in models directory.")
            rf_model = None
        else:
            rf_model = joblib.load(rf_path)
        
        if not os.path.exists(rf_combined_path):
            st.error(f"Model file 'rf_fll.joblib' not found in models directory.")
            rf_fll_model = None
        else:
            rf_fll_model = joblib.load(rf_fll_path)
        
        # Load TF-IDF vectorizer
        tfidf_path = os.path.join(models_path, 'tfidf.joblib')
        if not os.path.exists(tfidf_path):
            st.error(f"Model file 'tfidf.joblib' not found in models directory.")
            tfidf_vectorizer = None
        else:
            tfidf_vectorizer = joblib.load(tfidf_path)
        
        return rf_combined_model, rf_model, rf_fll_model, tfidf_vectorizer
    
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.info("There was an issue loading the model files. Please ensure they're properly formatted and accessible.")
        return None, None

# Predict function
def predict_fake_news(title, content):
    rf_combined_model, rf_model, rf_fll_model, tfidf_vectorizer = load_models()
    
    # Combine title and content
    combined_text = f"{title} {content}"
    
    # Clean the text
    cleaned_text = clean_text(combined_text)
    cleaned_text = remove_reuters(cleaned_text)
    
    # Get TF-IDF features
    tfidf_features = tfidf_vectorizer.transform([cleaned_text])
    tfidf_features_dense = tfidf_features.toarray()
    
    # Get empirical proportion features
    emp_features = eval_empirical_proportion(cleaned_text).values.reshape(1, -1)
    
    # Combine features
    combined_features = np.hstack((tfidf_features_dense, emp_features))
    
    # Predict using the combined model
    prediction = 0.3 * rf_combined_model.predict_proba(combined_features)[0][1] + 0.35 * rf_model.predict_proba(tfidf_features)[0][1] + 0.35 * rf_fll_model.predict_proba(emp_features)[0][1]
    
    # Convert to "realness" score (0=fake, 1=real)
    realness_score = prediction
    print(rf_combined_model.predict_proba(combined_features)[0][1], rf_model.predict_proba(tfidf_features)[0][1], rf_fll_model.predict_proba(emp_features)[0][1])
    
    return realness_score

# Custom CSS for the app
def apply_custom_css():
    st.markdown("""
    <style>
    /* Change the focus color for input fields and buttons */
    .stTextInput:focus, .stTextArea:focus, .stButton>button:focus, .stButton>button:hover {
        border-color: #2ecc71 !important;
        box-shadow: 0 0 0 0.2rem rgba(46, 204, 113, 0.25) !important;
    }
    
    .stButton>button {
        background-color: #27ae60;
        color: white;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #2ecc71;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 10px;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #2ecc71;
    }
    
    /* Center align header elements */
    .app-header {
        text-align: center;
    }
    
    /* Card-like containers */
    .content-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* For expanders */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px !important;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #e9ecef;
    }
    
    /* For tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 400;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-bottom: 2px solid #27ae60;
        font-weight: 600;
    }
    
    /* Sidebar adjustments */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        padding-top: 2rem;
    }
    
    section[data-testid="stSidebar"] .stButton>button {
        margin-bottom: 10px;
    }
    
    /* Progress bars and sliders */
    div[role="progressbar"] > div {
        background-color: #27ae60 !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Logo display function
def display_logo(width=1000):
    logo_path = "logo.png"
    
    if os.path.exists(logo_path):
        st.image(logo_path, width=width, use_container_width = False)
    else:
        st.warning("Logo file not found. Please add 'logo.png' to the root directory.")

# App title
def app_header():
    st.markdown('<div class="app-header">', unsafe_allow_html=True)
    display_logo(width=250)
    st.title("Fake News Detector")
    st.markdown("</div>", unsafe_allow_html=True)

# Initialize database
init_db()

# Main app
def main():
    # Apply custom CSS
    apply_custom_css()
    
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
    
    # App background and configuration
    st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # If user is not logged in, show login/signup page
    if st.session_state.user_id is None:
        login_signup_page()
    else:
        # User is logged in, show main application
        app_header()
        
        # Sidebar for navigation
        with st.sidebar:
            st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
            display_logo(width=150)
            st.title("Navigation")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="nav-buttons">', unsafe_allow_html=True)
            if st.button("📰 Check Article", use_container_width=True):
                st.session_state.current_page = "check_article"
                st.rerun()
            
            if st.button("📚 History", use_container_width=True):
                st.session_state.current_page = "history"
                st.rerun()
            
            if st.button("🔖 Saved Articles", use_container_width=True):
                st.session_state.current_page = "saved_articles"
                st.rerun()
            
            st.markdown("---")
            if st.button("🚪 Sign Out", use_container_width=True):
                st.session_state.user_id = None
                st.session_state.current_page = "check_article"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Main content based on selected page
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        if st.session_state.current_page == "check_article":
            check_article_page()
        elif st.session_state.current_page == "history":
            history_page()
        elif st.session_state.current_page == "saved_articles":
            saved_articles_page()
        st.markdown('</div>', unsafe_allow_html=True)

def login_signup_page():
    st.markdown('<div class="app-header">', unsafe_allow_html=True)
    display_logo(width=300)
    st.title("Fake News Detector")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔑 Login", "✏️ Sign Up"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("👤 Username", key="login_username")
        password = st.text_input("🔒 Password", type="password", key="login_password")
        
        if st.button("Login", key="login_button", use_container_width=True):
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
        new_username = st.text_input("👤 Username", key="signup_username")
        new_password = st.text_input("🔒 Password", type="password", key="signup_password")
        confirm_password = st.text_input("🔒 Confirm Password", type="password", key="confirm_password")
        
        if st.button("Sign Up", key="signup_button", use_container_width=True):
            if new_username and new_password and confirm_password:
                if new_password == confirm_password:
                    if register_user(new_username, new_password):
                        st.success("Account created successfully! Please login.")
                    else:
                        st.error("Username already exists")
                else:
                    st.error("Password do not match")
            else:
                st.warning("Please fill in all fields")
    st.markdown('</div>', unsafe_allow_html=True)

def check_article_page():
    st.header("📰 Check Article")
    
    # Input fields
    title = st.text_input("📝 Article Title", value=st.session_state.current_title)
    content = st.text_area("📄 Article Content", height=200, value=st.session_state.current_content)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔍 Analyze", use_container_width=True):
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
            if st.button("💾 Save Article", use_container_width=True):
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
        st.subheader("🔬 Analysis Results")
        
        score = st.session_state.current_prediction
        
        # Create a nicer meter to visualize the score
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Determine color based on score (red for fake, green for real)
            color = f"rgba({int(255 * (1 - score))}, {int(255 * score)}, 0, 0.8)"
            
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="width: 100%; height: 30px; background: linear-gradient(to right, #ff0000, #ffff00, #00ff00); 
                     border-radius: 15px; box-shadow: inset 0 2px 5px rgba(0,0,0,0.1);">
                    <div style="position: relative; left: {score * 100}%; transform: translateX(-50%); 
                         width: 15px; height: 30px; background-color: black; border-radius: 3px;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                    <span style="color: #d63031; font-weight: 500;">Fake (0.0)</span>
                    <span style="color: #27ae60; font-weight: 500;">Real (1.0)</span>
                </div>
                <h2 style="margin-top: 15px; color: {color}; font-weight: 600;">Score: {score:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Classification based on score
        st.markdown("### 💡 Interpretation")
        
        if score < 0.3:
            st.markdown("""
            <div style="background-color: #fad7d7; border-left: 5px solid #d63031; padding: 15px; border-radius: 5px;">
                <h4 style="color: #d63031; margin: 0;">⚠️ Likely Unreliable</h4>
                <p>This article appears to be highly unreliable and likely contains misinformation.</p>
            </div>
            """, unsafe_allow_html=True)
        elif score < 0.6:
            st.markdown("""
            <div style="background-color: #ffeeba; border-left: 5px solid #ffc107; padding: 15px; border-radius: 5px;">
                <h4 style="color: #856404; margin: 0;">⚠️ Exercise Caution</h4>
                <p>This article contains some questionable elements. Verify with other sources.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #d4edda; border-left: 5px solid #27ae60; padding: 15px; border-radius: 5px;">
                <h4 style="color: #155724; margin: 0;">✅ Likely Reliable</h4>
                <p>This article appears to be from a reliable source with credible information.</p>
            </div>
            """, unsafe_allow_html=True)

def history_page():
    st.header("📚 History")
    
    history = get_history(st.session_state.user_id)
    
    if not history:
        st.info("No history found. Check some articles first!")
        return
    
    for i, item in enumerate(history):
        with st.expander(f"📄 {item['title']} - Score: {item['prediction']:.2f} - {item['date']}"):
            st.write(f"**Content Preview:** {item['content'][:200]}...")
            
            # Score display
            score = item['prediction']
            
            # Determine color based on score
            if score < 0.3:
                status_color = "#d63031"
                status_text = "Likely Unreliable"
            elif score < 0.6:
                status_color = "#f39c12"
                status_text = "Questionable"
            else:
                status_color = "#27ae60"
                status_text = "Likely Reliable"
            
            st.markdown(f"""
            <div style="width: 100%; height: 20px; background: linear-gradient(to right, #ff0000, #ffff00, #00ff00); 
                 border-radius: 10px; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);">
                <div style="position: relative; left: {score * 100}%; transform: translateX(-50%); 
                     width: 10px; height: 20px; background-color: black; border-radius: 2px;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 5px; margin-bottom: 15px;">
                <span style="font-size: 0.8em;">Fake (0.0)</span>
                <span style="font-size: 0.8em; color: {status_color}; font-weight: 600;">{status_text}</span>
                <span style="font-size: 0.8em;">Real (1.0)</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Button to load article for re-checking
            if st.button("📋 Load this article", key=f"load_{item['id']}"):
                st.session_state.current_title = item['title']
                st.session_state.current_content = item['content']
                st.session_state.current_prediction = None
                st.session_state.current_page = "check_article"
                st.rerun()

def saved_articles_page():
    st.header("🔖 Saved Articles")
    
    articles = get_saved_articles(st.session_state.user_id)
    
    if not articles:
        st.info("No saved articles found. Save some articles first!")
        return
    
    for item in articles:
        with st.expander(f"📄 {item['title']} - Score: {item['prediction']:.2f} - {item['date']}"):
            st.write(f"**Content Preview:** {item['content'][:200]}...")
            
            # Score display
            score = item['prediction']
            
            # Determine color based on score
            if score < 0.3:
                status_color = "#d63031"
                status_text = "Likely Unreliable"
            elif score < 0.6:
                status_color = "#f39c12"
                status_text = "Questionable"
            else:
                status_color = "#27ae60"
                status_text = "Likely Reliable"
            
            st.markdown(f"""
            <div style="width: 100%; height: 20px; background: linear-gradient(to right, #ff0000, #ffff00, #00ff00); 
                 border-radius: 10px; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);">
                <div style="position: relative; left: {score * 100}%; transform: translateX(-50%); 
                     width: 10px; height: 30px; background-color: black; border-radius: 2px;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 5px; margin-bottom: 15px;">
                <span style="font-size: 0.8em;">Fake (0.0)</span>
                <span style="font-size: 0.8em; color: {status_color}; font-weight: 600;">{status_text}</span>
                <span style="font-size: 0.8em;">Real (1.0)</span>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Button to load article for re-checking
                if st.button("📋 Load this article", key=f"load_saved_{item['id']}"):
                    st.session_state.current_title = item['title']
                    st.session_state.current_content = item['content']
                    st.session_state.current_prediction = None
                    st.session_state.current_page = "check_article"
                    st.rerun()
            
            with col2:
                # Button to delete saved article
                if st.button("🗑️ Delete", key=f"delete_{item['id']}"):
                    delete_saved_article(item['id'])
                    st.success("Article deleted successfully!")
                    st.rerun()

if __name__ == "__main__":
    main()
