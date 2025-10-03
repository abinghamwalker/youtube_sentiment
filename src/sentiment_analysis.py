"""
Detailed Sentiment Analysis using Random Forest
- Reads config and data from their respective directories in the project root.
- Saves all analysis artifacts, including a final PDF summary, to the 'sentiment_files/' directory.
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import re
import os
import string
import configparser

# --- NLP & Preprocessing ---
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect, LangDetectException

# --- Machine Learning ---
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- API & Reporting ---
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from wordcloud import WordCloud

# ============================================================================
# 2. PATHS & GLOBAL CONFIGURATION
# ============================================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(ROOT_DIR, 'config.ini')
CSV_INPUT_DIR = os.path.join(ROOT_DIR, 'csv_files')
SENTIMENT_ARTIFACTS_DIR = os.path.join(ROOT_DIR, 'sentiment_files')
OUTPUT_ARTIFACTS_DIR = os.path.join(ROOT_DIR,'display_files')

# ============================================================================
# 3. UTILITY & PREPROCESSING FUNCTIONS (DEFINED AT TOP LEVEL)
# ============================================================================
def load_config():
    """Loads config from the project root."""
    config = configparser.ConfigParser()
    if not os.path.exists(CONFIG_FILE):
        print(f"ERROR: config.ini not found at '{CONFIG_FILE}'")
        return None, None
    config.read(CONFIG_FILE)
    try:
        api_key = config['youtube_api']['api_key']
        video_id = config['video_details']['video_id']
        return api_key, video_id
    except KeyError as e:
        print(f"ERROR: Missing key in config.ini: {e}")
        return None, None

def get_video_title(api_key, video_id):
    """Fetches the video title from the YouTube API."""
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.videos().list(part='snippet', id=video_id)
        response = request.execute()
        if response['items']: return response['items'][0]['snippet']['title']
    except HttpError as e:
        print(f"✗ WARNING: Could not fetch video title: {e}")
    return None

def create_base_filename(video_title, video_id):
    """Creates a clean, file-safe base name from the video title and ID."""
    if video_title:
        clean_title = re.sub(r'[^\w\s-]', '', video_title).strip().lower()
        clean_title = re.sub(r'[-\s]+', '_', clean_title)
        return f"{clean_title}_{video_id}"
    return video_id

def find_latest_comments_file(directory, prefix):
    """Finds the most recent file in a specific directory."""
    try:
        if not os.path.isdir(directory): return None
        files = os.listdir(directory)
        candidate_files = [f for f in files if f.startswith(prefix) and f.endswith('.csv')]
        if not candidate_files: return None
        return max(candidate_files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    except Exception as e:
        print(f"An error occurred while searching for the data file: {e}")
        return None

def is_english(text):
    """Checks if a given text is in English."""
    try: return detect(text) == 'en'
    except LangDetectException: return False

def preprocess_text(text):
    """Cleans and preprocesses text for NLP."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def get_textblob_sentiment(text):
    """Gets sentiment polarity using TextBlob."""
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1: return 'positive'
    elif polarity < -0.1: return 'negative'
    else: return 'neutral'

# ============================================================================
# 4. PDF REPORTING CLASS
# ============================================================================
class PDF(FPDF):
    def __init__(self, video_title, video_id):
        super().__init__()
        self.video_title = video_title
        self.video_id = video_id

    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, f'Sentiment Analysis Report: {self.video_title}', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.set_font('Helvetica', '', 8)
        self.cell(0, 5, f'Video ID: {self.video_id}', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.ln(5)

    def chapter_body(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            body_text = f.read()
        self.set_font('Courier', '', 10)
        self.multi_cell(0, 5, body_text)
        self.ln()
        
    def add_full_width_image(self, image_path, title):
        self.add_page()
        self.chapter_title(title)
        self.image(image_path, w=190)

def create_pdf_report(summary_file, dashboard_img, wordcloud_img, video_title, video_id, pdf_report_path):
    """Generates a PDF report from the analysis artifacts."""
    print("\n--- 8. Generating PDF Report ---")
    
    pdf = PDF(video_title=video_title, video_id=video_id)
    
    pdf.add_page()
    pdf.chapter_title('Analysis Summary & Model Performance')
    pdf.chapter_body(summary_file)
    
    pdf.add_full_width_image(dashboard_img, 'Analysis Dashboard')
    pdf.add_full_width_image(wordcloud_img, 'Sentiment Word Clouds')
    
    pdf.output(pdf_report_path) 
    
    print(f"✓ PDF report saved to '{pdf_report_path}'")

# ============================================================================
# 5. Dashboard and Wordcloud Generation
# ============================================================================

def generate_dashboard(df, top_features, output_path):
    """
    Generates a 2x2 dashboard with key sentiment analysis insights and saves it to a file.
    
    Args:
        df (pd.DataFrame): The DataFrame with analysis results.
        top_features (pd.DataFrame): DataFrame with the model's top features.
        output_path (str): The path to save the output PNG file.
    """
    print(f"--- 6. Generating Analysis Dashboard ---")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('YouTube Comment Sentiment Analysis Dashboard', fontsize=20, weight='bold')

    # --- Plot 1: Overall Sentiment Distribution ---
    sns.countplot(ax=axes[0, 0], data=df, x='predicted_sentiment', 
                  order=['positive', 'neutral', 'negative'], 
                  palette={'positive': 'mediumseagreen', 'neutral': 'lightgray', 'negative': 'lightcoral'})
    axes[0, 0].set_title('Overall Sentiment Distribution', fontsize=14, weight='bold')
    axes[0, 0].set_xlabel('Predicted Sentiment', fontsize=12)
    axes[0, 0].set_ylabel('Number of Comments', fontsize=12)

    # --- Plot 2: Top 20 Most Important Words/Phrases ---
    sns.barplot(ax=axes[0, 1], data=top_features, x='importance', y='feature', palette='viridis')
    axes[0, 1].set_title('Top 20 Most Important Features (TF-IDF)', fontsize=14, weight='bold')
    axes[0, 1].set_xlabel('Importance Score', fontsize=12)
    axes[0, 1].set_ylabel('Feature (Word/N-gram)', fontsize=12)

    # --- Plot 3: Sentiment Over Time ---
    sentiment_over_time = df.set_index('comment_date').resample('D')['predicted_sentiment'].value_counts().unstack().fillna(0)
    sentiment_over_time.plot(kind='line', ax=axes[1, 0], 
                             color={'positive': 'mediumseagreen', 'neutral': 'gray', 'negative': 'lightcoral'}, marker='o')
    axes[1, 0].set_title('Sentiment Volume Over Time', fontsize=14, weight='bold')
    axes[1, 0].set_xlabel('Date', fontsize=12)
    axes[1, 0].set_ylabel('Number of Comments', fontsize=12)
    axes[1, 0].legend(title='Sentiment')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # --- Plot 4: Likes Distribution by Sentiment ---
    sns.boxplot(ax=axes[1, 1], data=df, x='predicted_sentiment', y='likes', 
                order=['positive', 'neutral', 'negative'],
                palette={'positive': 'mediumseagreen', 'neutral': 'lightgray', 'negative': 'lightcoral'})
    axes[1, 1].set_title('Likes Distribution by Sentiment', fontsize=14, weight='bold')
    axes[1, 1].set_xlabel('Predicted Sentiment', fontsize=12)
    axes[1, 1].set_ylabel('Number of Likes (Log Scale)', fontsize=12)
    axes[1, 1].set_yscale('log') # Use log scale for likes, as they can vary wildly
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to make room for suptitle
    plt.savefig(output_path)
    plt.close(fig)
    print(f"✓ Dashboard saved to '{output_path}'")


def generate_wordclouds(df, output_path):
    """
    Generates a 1x3 plot of word clouds for positive, neutral, and negative sentiment.
    
    Args:
        df (pd.DataFrame): The DataFrame with analysis results.
        output_path (str): The path to save the output PNG file.
    """
    print(f"--- 7. Generating Sentiment Word Clouds ---")
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor='black')
    sentiments = ['positive', 'neutral', 'negative']
    colors = ['Greens', 'Greys', 'Reds']
    
    for i, sentiment in enumerate(sentiments):
        text = ' '.join(df[df['predicted_sentiment'] == sentiment]['cleaned_text'])
        
        ax = axes[i]
        ax.set_title(f'{sentiment.capitalize()} Comments', fontsize=20, color='white', weight='bold')
        
        if text:
            wordcloud = WordCloud(width=800, height=800, 
                                  background_color='black', colormap=colors[i],
                                  min_font_size=10).generate(text)
            ax.imshow(wordcloud, interpolation='bilinear')
        else:
            ax.text(0.5, 0.5, 'No comments found\nfor this sentiment', 
                    color='white', ha='center', va='center', fontsize=18)
            
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, facecolor='black')
    plt.close(fig)
    print(f"✓ Word clouds saved to '{output_path}'")

# ============================================================================
# 6. MAIN ANALYSIS WORKFLOW
# ============================================================================
def main():
    """Executes the entire sentiment analysis pipeline by calling top-level functions."""
    # --- NLTK Downloads ---
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    # --- Setup ---
    API_KEY, VIDEO_ID = load_config()
    if not API_KEY or not VIDEO_ID: return

    VIDEO_TITLE = get_video_title(API_KEY, VIDEO_ID)
    BASE_FILENAME = create_base_filename(VIDEO_TITLE, VIDEO_ID)
    print(f"--- Analysis configured for Video ID: {VIDEO_ID} ---")
    
    os.makedirs(SENTIMENT_ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_ARTIFACTS_DIR,exist_ok=True)
    OUTPUT_CSV_FILE = os.path.join(SENTIMENT_ARTIFACTS_DIR, f"{BASE_FILENAME}_annotated_comments.csv")
    SUMMARY_FILE = os.path.join(SENTIMENT_ARTIFACTS_DIR, f"{BASE_FILENAME}_summary_report.txt")
    VISUALIZATION_FILE = os.path.join(SENTIMENT_ARTIFACTS_DIR, f"{BASE_FILENAME}_dashboard.png")
    WORDCLOUD_FILE = os.path.join(SENTIMENT_ARTIFACTS_DIR, f"{BASE_FILENAME}_wordclouds.png")
    PDF_REPORT_FILE = os.path.join(OUTPUT_ARTIFACTS_DIR, f"{BASE_FILENAME}_sentiment_report.pdf")
    
    # --- 1. Data Loading and Preprocessing ---
    print("\n--- 1. Loading and Preprocessing Data ---")
    SEARCH_PREFIX = f"{BASE_FILENAME}_comments_"
    INPUT_FILENAME = find_latest_comments_file(directory=CSV_INPUT_DIR, prefix=SEARCH_PREFIX)
    if INPUT_FILENAME is None:
        print(f"✗ ERROR: No data file found in '{CSV_INPUT_DIR}' with the prefix '{SEARCH_PREFIX}'.")
        return
    
    INPUT_FILEPATH = os.path.join(CSV_INPUT_DIR, INPUT_FILENAME)
    df = pd.read_csv(INPUT_FILEPATH)
    df.dropna(subset=['comment_text'], inplace=True)
    
    df = df[df['comment_text'].str.len() > 15].copy()
    df['is_english'] = df['comment_text'].apply(is_english)
    df = df[df['is_english']].copy()
    
    df['cleaned_text'] = df['comment_text'].apply(preprocess_text)
    df['initial_sentiment'] = df['comment_text'].apply(get_textblob_sentiment)
    df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
    print(f"✓ Final clean dataset size: {len(df)}")
    
    # --- 2. Feature Engineering & Splitting ---
    print("\n--- 2. Vectorizing Text and Splitting Data ---")
    tfidf_vectorizer = TfidfVectorizer(max_features=2500, min_df=3, max_df=0.8, ngram_range=(1, 2))
    X = tfidf_vectorizer.fit_transform(df['cleaned_text'])
    y = df['initial_sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"✓ Data split into training ({X_train.shape[0]}) and testing ({X_test.shape[0]}) sets.")
    
    # --- 3. Model Training & Optimization ---
    print("\n--- 3. Optimizing Random Forest Hyperparameters ---")
    param_dist = {
        'n_estimators': [100, 200, 300, 400], 'max_depth': [10, 20, 30, 40, None],
        'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'], 'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=3, verbose=2, random_state=42, n_jobs=-1, scoring='f1_weighted')
    random_search.fit(X_train, y_train)
    
    # --- 4. Final Prediction & Evaluation ---
    print("\n--- 4. Evaluating Final Optimized Model ---")
    final_rf_model = random_search.best_estimator_
    y_pred = final_rf_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # --- 5. Results Analysis & Export ---
    print("\n--- 5. Analyzing and Exporting Results ---")
    df['predicted_sentiment'] = final_rf_model.predict(X)
    df['sentiment_confidence'] = final_rf_model.predict_proba(X).max(axis=1)
    
    feature_importances = pd.DataFrame({'feature': tfidf_vectorizer.get_feature_names_out(), 'importance': final_rf_model.feature_importances_}).sort_values('importance', ascending=False)
    top_20_features = feature_importances.head(20)
    
    df['comment_date'] = pd.to_datetime(df['comment_date'])
    # No need for df_sorted here unless used elsewhere
    
    df_to_save = df[['comment_text', 'comment_date', 'likes', 'replies', 'predicted_sentiment', 'sentiment_confidence']]
    df_to_save.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"✓ Annotated data saved to '{OUTPUT_CSV_FILE}'")
    
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write(f"RANDOM FOREST SENTIMENT ANALYSIS REPORT: {VIDEO_TITLE}\n")
        f.write("="*50 + "\n")
        f.write("Model Performance:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\n" + "="*50 + "\n")
        f.write("Top 20 Most Important Features:\n")
        f.write(top_20_features.to_string())
        
    generate_dashboard(df, top_20_features, VISUALIZATION_FILE)
    generate_wordclouds(df, WORDCLOUD_FILE)
    
    # --- 8. Generate Final PDF Report ---
    create_pdf_report(SUMMARY_FILE, VISUALIZATION_FILE, WORDCLOUD_FILE, VIDEO_TITLE, VIDEO_ID, PDF_REPORT_FILE)

# ============================================================================
# 7. SCRIPT EXECUTION GUARD
# ============================================================================
if __name__ == "__main__":
    main()
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE. ALL DELIVERABLES GENERATED.")
    print("="*60)