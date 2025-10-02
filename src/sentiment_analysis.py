# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import warnings
import re
import os
import string
import configparser

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect, LangDetectException
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from wordcloud import WordCloud
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --- DEFINE FOLDER STRUCTURE ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(ROOT_DIR, 'config.ini')
CSV_INPUT_DIR = os.path.join(ROOT_DIR, 'csv_files')
SENTIMENT_ARTIFACTS_DIR = os.path.join(ROOT_DIR, 'sentiment_files')

# --- NLTK Downloads ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ============================================================================
# 2. DYNAMIC CONFIGURATION & FILENAME SETUP
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
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.videos().list(part='snippet', id=video_id)
        response = request.execute()
        if response['items']: return response['items'][0]['snippet']['title']
    except HttpError as e:
        print(f"✗ WARNING: Could not fetch video title: {e}")
    return None

def create_base_filename(video_title, video_id):
    if video_title:
        clean_title = re.sub(r'[^\w\s-]', '', video_title).strip().lower()
        clean_title = re.sub(r'[-\s]+', '_', clean_title)
        return f"{clean_title}_{video_id}"
    return video_id

# --- Main Setup Execution ---
API_KEY, VIDEO_ID = load_config()
if not API_KEY or not VIDEO_ID: exit()

VIDEO_TITLE = get_video_title(API_KEY, VIDEO_ID)
BASE_FILENAME = create_base_filename(VIDEO_TITLE, VIDEO_ID)
print(f"--- Analysis configured for Video ID: {VIDEO_ID} ---")
print(f"--- Output files will be prefixed with: '{BASE_FILENAME}' ---")

# --- Ensure output directory exists and define full output paths ---
os.makedirs(SENTIMENT_ARTIFACTS_DIR, exist_ok=True)
OUTPUT_CSV_FILE = os.path.join(SENTIMENT_ARTIFACTS_DIR, f"{BASE_FILENAME}_annotated_comments.csv")
SUMMARY_FILE = os.path.join(SENTIMENT_ARTIFACTS_DIR, f"{BASE_FILENAME}_summary_report.txt")
VISUALIZATION_FILE = os.path.join(SENTIMENT_ARTIFACTS_DIR, f"{BASE_FILENAME}_dashboard.png")
WORDCLOUD_FILE = os.path.join(SENTIMENT_ARTIFACTS_DIR, f"{BASE_FILENAME}_wordclouds.png")

# ============================================================================
# 3. DATA LOADING AND PREPARATION
# ============================================================================
def find_latest_comments_file(directory, prefix):
    """Finds the most recent file in a specific directory."""
    try:
        files = os.listdir(directory)
        candidate_files = [f for f in files if f.startswith(prefix) and f.endswith('.csv')]
        if not candidate_files: return None
        return max(candidate_files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    except Exception as e:
        print(f"An error occurred while searching for the data file: {e}")
        return None

print("\n--- 1. Loading and Preprocessing Data ---")

# --- Search for input file in the correct directory ---
SEARCH_PREFIX = f"{BASE_FILENAME}_comments_"
INPUT_FILENAME = find_latest_comments_file(directory=CSV_INPUT_DIR, prefix=SEARCH_PREFIX)

if INPUT_FILENAME is None:
    print(f"✗ ERROR: No data file found in '{CSV_INPUT_DIR}' with the prefix '{SEARCH_PREFIX}'.")
    print("Please run the data collection script first.")
    exit()

INPUT_FILEPATH = os.path.join(CSV_INPUT_DIR, INPUT_FILENAME)
print(f"Found latest data file: '{INPUT_FILEPATH}'")
try:
    df = pd.read_csv(INPUT_FILEPATH)
    df.dropna(subset=['comment_text'], inplace=True)
except Exception as e:
    print(f"✗ ERROR: Could not read the input file '{INPUT_FILEPATH}'. Error: {e}")
    exit()


print(f"Initial raw comments loaded: {len(df)}")
df = df[df['comment_text'].str.len() > 15].copy()
print(f"Comments after length filtering (> 15 chars): {len(df)}")
def is_english(text):
    try: return detect(text) == 'en'
    except LangDetectException: return False
df['is_english'] = df['comment_text'].apply(is_english)
df = df[df['is_english']].copy()
print(f"Comments after filtering for English: {len(df)}")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)
df['cleaned_text'] = df['comment_text'].apply(preprocess_text)
def get_textblob_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1: return 'positive'
    elif polarity < -0.1: return 'negative'
    else: return 'neutral'
df['initial_sentiment'] = df['comment_text'].apply(get_textblob_sentiment)
df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
print(f"✓ Final clean dataset size: {len(df)}")

print("\n--- 2. Vectorizing Text and Splitting Data ---")
tfidf_vectorizer = TfidfVectorizer(max_features=2500, min_df=3, max_df=0.8, ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(df['cleaned_text'])
y = df['initial_sentiment'].values
print(f"✓ Feature matrix created with shape: {X.shape}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"✓ Data split into training ({X_train.shape[0]}) and testing ({X_test.shape[0]}) sets.")

print("\n--- 3. Optimizing Random Forest Hyperparameters ---")
param_dist = {
    'n_estimators': [100, 200, 300, 400], 'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'], 'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample']
}
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=3, verbose=2, random_state=42, n_jobs=-1, scoring='f1_weighted')
start_time = time()
random_search.fit(X_train, y_train)
print(f"✓ Randomized search completed in {time() - start_time:.2f} seconds.")
print("Best parameters found:")
print(random_search.best_params_)

print("\n--- 4. Training Final Optimized Model ---")
final_rf_model = random_search.best_estimator_
final_rf_model.fit(X_train, y_train)
print("\n--- Performance on Test Set ---")
y_pred = final_rf_model.predict(X_test)
print(classification_report(y_test, y_pred))
print("\n--- Predicting Sentiment for Entire Dataset ---")
df['predicted_sentiment'] = final_rf_model.predict(X)
df['sentiment_confidence'] = final_rf_model.predict_proba(X).max(axis=1)
print("✓ Final sentiment labels generated.")

print("\n--- 5. Analyzing Results ---")
feature_importances = pd.DataFrame({'feature': tfidf_vectorizer.get_feature_names_out(), 'importance': final_rf_model.feature_importances_}).sort_values('importance', ascending=False)
top_20_features = feature_importances.head(20)
df['comment_date'] = pd.to_datetime(df['comment_date'])
df['day'] = df['comment_date'].dt.date
df_sorted = df.sort_values('comment_date').reset_index(drop=True)
df_sorted['is_positive'] = (df_sorted['predicted_sentiment'] == 'positive').astype(int)
df_sorted['positive_rolling_avg'] = df_sorted['is_positive'].rolling(window=100, min_periods=10).mean() * 100
daily_sentiment = df.groupby(['day', 'predicted_sentiment']).size().unstack(fill_value=0)
daily_sentiment_pct = daily_sentiment.div(daily_sentiment.sum(axis=1), axis=0) * 100
print("✓ Temporal features and feature importances calculated.")

print("\n--- 6. Generating Visualizations ---")
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(20, 18))
plot_title = f"Random Forest Sentiment Analysis: {VIDEO_TITLE}" if VIDEO_TITLE else f"Sentiment Analysis: {VIDEO_ID}"
fig.suptitle(plot_title, fontsize=24, fontweight='bold')
gs = fig.add_gridspec(3, 2)
ax1 = fig.add_subplot(gs[0, 0])
sns.barplot(x='importance', y='feature', data=top_20_features, ax=ax1, palette='viridis', hue='feature', legend=False)
ax1.set_title('Top 20 Key Phrases (Feature Importance)', fontsize=16, fontweight='bold')
ax2 = fig.add_subplot(gs[0, 1])
sentiment_counts = df['predicted_sentiment'].value_counts()
ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66c2a5', '#fc8d62', '#8da0cb'])
ax2.set_title('Overall Sentiment Distribution', fontsize=16, fontweight='bold')
ax3 = fig.add_subplot(gs[1, :])
daily_sentiment_pct.plot(ax=ax3, marker='.', linestyle='-', linewidth=2)
ax3.set_title('Daily Sentiment Percentage Over Time', fontsize=16, fontweight='bold')
ax4 = fig.add_subplot(gs[2, :])
ax4.plot(df_sorted['comment_date'], df_sorted['positive_rolling_avg'], color='#66c2a5', linewidth=2.5)
ax4.set_title('Positive Sentiment (100-Comment Rolling Average)', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(VISUALIZATION_FILE, dpi=300)
print(f"✓ Main dashboard saved to '{VISUALIZATION_FILE}'")
fig_wc, axes = plt.subplots(1, 3, figsize=(24, 7))
sentiments = ['positive', 'negative', 'neutral']
colors = ['Greens', 'Reds', 'Blues']
for idx, (sentiment, cmap) in enumerate(zip(sentiments, colors)):
    text = ' '.join(df[df['predicted_sentiment'] == sentiment]['cleaned_text'])
    if text:
        wc = WordCloud(width=800, height=500, background_color='white', colormap=cmap).generate(text)
        axes[idx].imshow(wc, interpolation='bilinear')
        axes[idx].set_title(f'{sentiment.upper()} Comments', fontsize=18, fontweight='bold')
    axes[idx].axis('off')
plt.tight_layout()
plt.savefig(WORDCLOUD_FILE, dpi=300)
print(f"✓ Word clouds saved to '{WORDCLOUD_FILE}'")

print("\n--- 7. Exporting Final Results ---")
df_to_save = df[['comment_text', 'comment_date', 'likes', 'replies', 'predicted_sentiment', 'sentiment_confidence']]
df_to_save.to_csv(OUTPUT_CSV_FILE, index=False)
print(f"✓ Annotated data saved to '{OUTPUT_CSV_FILE}'")
with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
    f.write(f"RANDOM FOREST SENTIMENT ANALYSIS REPORT: {VIDEO_TITLE}\n" + "="*60 + "\n\n")
    f.write(f"Video ID: {VIDEO_ID}\n")
    f.write(f"Total Comments Analyzed (English only): {len(df)}\n")
    f.write(f"Model: Optimized Random Forest Classifier\n\n")
    f.write("--- Model Performance (on Test Set) ---\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\n--- Best Hyperparameters ---\n")
    f.write(str(random_search.best_params_))
    f.write("\n\n--- Top 20 Most Influential Phrases ---\n")
    f.write(top_20_features.to_string(index=False))
print(f"✓ Summary report saved to '{SUMMARY_FILE}'")
print("\n" + "="*60)
print("ANALYSIS COMPLETE. ALL DELIVERABLES GENERATED.")
print("="*60)