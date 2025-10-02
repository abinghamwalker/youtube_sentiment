
import configparser
import os
import re
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime

# --- DEFINE FOLDER STRUCTURE ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(ROOT_DIR, 'config.ini')
CSV_OUTPUT_DIR = os.path.join(ROOT_DIR, 'csv_files')

# ============================================================================
# DYNAMIC CONFIGURATION & FILENAME SETUP
# ============================================================================

def load_config():
    """Loads API key and Video ID from config.ini in the project root."""
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
        if response['items']:
            return response['items'][0]['snippet']['title']
    except HttpError as e:
        print(f"✗ WARNING: Could not fetch video title: {e}")
    return None

def create_base_filename(video_title, video_id):
    """Creates a clean, file-safe base name."""
    if video_title:
        clean_title = re.sub(r'[^\w\s-]', '', video_title).strip().lower()
        clean_title = re.sub(r'[-\s]+', '_', clean_title)
        return f"{clean_title}_{video_id}"
    return video_id

# --- Main Setup Execution ---
API_KEY, VIDEO_ID = load_config()
if not API_KEY or not VIDEO_ID:
    exit()

VIDEO_TITLE = get_video_title(API_KEY, VIDEO_ID)
BASE_FILENAME = create_base_filename(VIDEO_TITLE, VIDEO_ID)
MAX_COMMENTS = 5000

# ============================================================================
# YOUTUBE API FUNCTIONS (No changes needed here)
# ============================================================================
def get_comments(youtube, video_id, max_results):
    comments = []
    next_page_token = None
    try:
        while len(comments) < max_results:
            request = youtube.commentThreads().list(
                part='snippet', videoId=video_id, maxResults=min(100, max_results - len(comments)),
                pageToken=next_page_token, textFormat='plainText', order='time'
            )
            response = request.execute()
            for item in response['items']:
                comment_snippet = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'comment_id': item['snippet']['topLevelComment']['id'],
                    'comment_text': comment_snippet['textDisplay'],
                    'author': comment_snippet['authorDisplayName'],
                    'comment_date': comment_snippet['publishedAt'],
                    'updated_date': comment_snippet['updatedAt'],
                    'likes': comment_snippet['likeCount'],
                    'replies': item['snippet']['totalReplyCount'],
                    'is_reply': False, 'parent_id': None
                })
            if 'nextPageToken' in response:
                next_page_token = response['nextPageToken']
            else:
                break
            print(f"Collected {len(comments)} comments so far...")
    except HttpError as e:
        print(f"An HTTP error occurred: {e}")
    return comments

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("="*50)
    print("YOUTUBE COMMENTS SCRAPER")
    print("="*50)
    print(f"\nConfigured for Video ID: {VIDEO_ID}")
    print(f"Saving files with base name: '{BASE_FILENAME}'")
    
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    comments = get_comments(youtube, VIDEO_ID, MAX_COMMENTS)
    
    if not comments:
        print("\n✗ No comments collected.")
        return
        
    print(f"\n✓ Successfully collected {len(comments)} comments!")
    df = pd.DataFrame(comments)
    
    # --- MODIFIED: Ensure output directory exists and save file there ---
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{BASE_FILENAME}_comments_{timestamp}.csv"
    output_filepath = os.path.join(CSV_OUTPUT_DIR, filename)
    
    df.to_csv(output_filepath, index=False, encoding='utf-8')
    print(f"✓ Data saved to: {output_filepath}")
    print("\nNext step: Run the analysis script.")

if __name__ == "__main__":
    main()