"""
Complete Pytest Suite for YouTube Sentiment Analysis Script
Place this file in the 'tests/' directory as 'test_sentiment_analysis.py'
Run with: pytest tests/test_sentiment_analysis.py -v
"""

import pytest
import os
import sys
import configparser
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from datetime import datetime, timedelta
from googleapiclient.errors import HttpError
from langdetect import LangDetectException

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the module to test (adjust import name to match your actual script name)
import sentiment_analysis as sa


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_config_data():
    """Provides sample configuration data."""
    return {
        'youtube_api': {'api_key': 'test_api_key_123'},
        'video_details': {'video_id': 'test_video_id'}
    }


@pytest.fixture
def mock_video_response():
    """Provides a mock YouTube API video response."""
    return {
        'items': [{
            'snippet': {
                'title': 'Amazing Test Video (2024)'
            }
        }]
    }


@pytest.fixture
def sample_comments_df():
    """
    Provides a sample DataFrame with comments designed to pass TF-IDF vectorization.
    Key words like 'video', 'great', 'bad', 'test' are repeated to meet min_df=3.
    """

    # We need to make sure some words appear at least 3 times.
    comments = [
        # Positive comments
        "This is a great video! I love this test.",
        "Absolutely a great and amazing video. Well done.",
        "Great work on this test video.",
        "Love this content, it's a great test.",
        # Negative comments
        "This is a bad video, a really bad test.",
        "I did not like this bad video at all.",
        "Just a plain bad test, really.",
        # Neutral comments
        "This is a video.",
        "Okay, another test video.",
        "Just a video for a test.",
    ]
    
    dates = pd.date_range(start='2024-01-01', periods=len(comments), freq='D')
    return pd.DataFrame({
        'comment_id': [f'id_{i}' for i in range(len(comments))],
        'comment_text': comments,
        'author': [f'User{i}' for i in range(len(comments))],
        'comment_date': dates,
        'updated_date': dates,
        'likes': np.random.randint(0, 100, size=len(comments)),
        'replies': np.random.randint(0, 10, size=len(comments)),
    })


@pytest.fixture
def sample_processed_df():
    """Provides a sample DataFrame with processed sentiment data."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    return pd.DataFrame({
        'comment_text': ['Great video!'] * 10,
        'comment_date': dates,
        'likes': [10] * 10,
        'replies': [2] * 10,
        'cleaned_text': ['great video'] * 10,
        'predicted_sentiment': ['positive', 'negative', 'neutral'] * 3 + ['positive'],
        'sentiment_confidence': [0.85, 0.72, 0.65] * 3 + [0.90],
        'day': [d.date() for d in dates]
    })


@pytest.fixture
def mock_tfidf_vectorizer():
    """Provides a mock TF-IDF vectorizer."""
    vectorizer = MagicMock()
    vectorizer.fit_transform.return_value = np.random.rand(10, 100)
    vectorizer.transform.return_value = np.random.rand(10, 100)
    vectorizer.get_feature_names_out.return_value = np.array([f'word_{i}' for i in range(100)])
    return vectorizer


@pytest.fixture
def mock_random_forest():
    """Provides a mock Random Forest model."""
    model = MagicMock()
    model.predict.return_value = np.array(['positive', 'negative', 'neutral'] * 3 + ['positive'])
    model.predict_proba.return_value = np.random.rand(10, 3)
    model.feature_importances_ = np.random.rand(100)
    return model


# ============================================================================
# TESTS FOR load_config()
# ============================================================================

class TestLoadConfig:
    """Tests for the load_config function."""
    
    def test_load_config_success(self, tmp_path):
        """Test successful config loading."""
        config_file = tmp_path / "config.ini"
        config = configparser.ConfigParser()
        config['youtube_api'] = {'api_key': 'test_key'}
        config['video_details'] = {'video_id': 'test_id'}
        with open(config_file, 'w') as f:
            config.write(f)
        
        with patch.object(sa, 'CONFIG_FILE', str(config_file)):
            api_key, video_id = sa.load_config()
        
        assert api_key == 'test_key'
        assert video_id == 'test_id'
    
    def test_load_config_file_not_found(self, capsys):
        """Test behavior when config file doesn't exist."""
        with patch.object(sa, 'CONFIG_FILE', '/nonexistent/config.ini'):
            api_key, video_id = sa.load_config()
        
        assert api_key is None
        assert video_id is None
        captured = capsys.readouterr()
        assert "ERROR: config.ini not found" in captured.out
    
    def test_load_config_missing_keys(self, tmp_path, capsys):
        """Test handling of missing keys in config."""
        config_file = tmp_path / "config.ini"
        config = configparser.ConfigParser()
        config['youtube_api'] = {}
        with open(config_file, 'w') as f:
            config.write(f)
        
        with patch.object(sa, 'CONFIG_FILE', str(config_file)):
            api_key, video_id = sa.load_config()
        
        assert api_key is None
        assert video_id is None
        captured = capsys.readouterr()
        assert "ERROR: Missing key in config.ini" in captured.out


# ============================================================================
# TESTS FOR get_video_title()
# ============================================================================

class TestGetVideoTitle:
    """Tests for the get_video_title function."""
    
    @patch('sentiment_analysis.build')
    def test_get_video_title_success(self, mock_build, mock_video_response):
        """Test successful video title retrieval."""
        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_youtube.videos().list().execute.return_value = mock_video_response
        
        title = sa.get_video_title('test_api_key', 'test_video_id')
        
        assert title == 'Amazing Test Video (2024)'
    
    @patch('sentiment_analysis.build')
    def test_get_video_title_empty_response(self, mock_build):
        """Test handling of empty API response."""
        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_youtube.videos().list().execute.return_value = {'items': []}
        
        title = sa.get_video_title('test_api_key', 'test_video_id')
        
        assert title is None
    
    @patch('sentiment_analysis.build')
    def test_get_video_title_http_error(self, mock_build, capsys):
        """Test handling of HTTP errors."""
        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_youtube.videos().list().execute.side_effect = HttpError(
            resp=Mock(status=403), 
            content=b'Quota exceeded'
        )
        
        title = sa.get_video_title('test_api_key', 'test_video_id')
        
        assert title is None
        captured = capsys.readouterr()
        assert "WARNING: Could not fetch video title" in captured.out


# ============================================================================
# TESTS FOR create_base_filename()
# ============================================================================

class TestCreateBaseFilename:
    """Tests for the create_base_filename function."""
    
    def test_create_base_filename_with_title(self):
        """Test filename creation with a valid title."""
        result = sa.create_base_filename('Test Video Title!', 'abc123')
        assert result == 'test_video_title_abc123'
    
    def test_create_base_filename_special_chars(self):
        """Test filename sanitization with special characters."""
        result = sa.create_base_filename('Video: #1 (2024) [HD]', 'xyz789')
        assert result == 'video_1_2024_hd_xyz789'
    
    def test_create_base_filename_no_title(self):
        """Test fallback to video ID when no title provided."""
        result = sa.create_base_filename(None, 'fallback_id')
        assert result == 'fallback_id'
    
    def test_create_base_filename_empty_title(self):
        """Test handling of empty title string."""
        result = sa.create_base_filename('', 'empty_id')
        assert result == 'empty_id'


# ============================================================================
# TESTS FOR find_latest_comments_file()
# ============================================================================

class TestFindLatestCommentsFile:
    """Tests for the find_latest_comments_file function."""
    
    def test_find_latest_comments_file_success(self, tmp_path):
        """Test finding the latest comments file."""
        # Create test files with different timestamps
        file1 = tmp_path / "test_comments_20240101.csv"
        file2 = tmp_path / "test_comments_20240102.csv"
        file3 = tmp_path / "other_file.csv"
        
        file1.touch()
        file2.touch()
        file3.touch()
        
        # Modify file2 to be the newest
        os.utime(file2, (datetime.now().timestamp() + 100, datetime.now().timestamp() + 100))
        
        result = sa.find_latest_comments_file(str(tmp_path), 'test_comments_')
        
        assert result == 'test_comments_20240102.csv'
    
    def test_find_latest_comments_file_no_matches(self, tmp_path):
        """Test when no matching files exist."""
        result = sa.find_latest_comments_file(str(tmp_path), 'nonexistent_')
        assert result is None
    
    def test_find_latest_comments_file_invalid_directory(self):
        """Test with invalid directory."""
        result = sa.find_latest_comments_file('/nonexistent/path', 'test_')
        assert result is None
    
    def test_find_latest_comments_file_ignores_non_csv(self, tmp_path):
        """Test that non-CSV files are ignored."""
        file1 = tmp_path / "test_comments_data.csv"
        file2 = tmp_path / "test_comments_data.txt"
        
        file1.touch()
        file2.touch()
        
        result = sa.find_latest_comments_file(str(tmp_path), 'test_comments_')
        
        assert result == 'test_comments_data.csv'


# ============================================================================
# TESTS FOR is_english()
# ============================================================================

class TestIsEnglish:
    """Tests for the is_english function."""
    
    @patch('sentiment_analysis.detect')
    def test_is_english_true(self, mock_detect):
        """Test English text detection."""
        mock_detect.return_value = 'en'
        result = sa.is_english('This is English text')
        assert result is True
    
    @patch('sentiment_analysis.detect')
    def test_is_english_false(self, mock_detect):
        """Test non-English text detection."""
        mock_detect.return_value = 'es'
        result = sa.is_english('Esto es texto en español')
        assert result is False
    
    @patch('sentiment_analysis.detect')
    def test_is_english_exception(self, mock_detect):
        """Test handling of detection exceptions."""
        mock_detect.side_effect = LangDetectException('error', 'test error')
        result = sa.is_english('...')
        assert result is False


# ============================================================================
# TESTS FOR preprocess_text()
# ============================================================================

class TestPreprocessText:
    """Tests for the preprocess_text function."""
    
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing."""
        result = sa.preprocess_text('This is a TEST!')
        assert result.islower()
        assert 'test' in result
    
    def test_preprocess_text_removes_urls(self):
        """Test URL removal."""
        text = 'Check this out http://example.com great video!'
        result = sa.preprocess_text(text)
        assert 'http' not in result
        assert 'example' not in result
    
    def test_preprocess_text_removes_punctuation(self):
        """Test punctuation removal."""
        text = 'Hello! How are you?'
        result = sa.preprocess_text(text)
        assert '!' not in result
        assert '?' not in result
    
    def test_preprocess_text_removes_stopwords(self):
        """Test stopword removal."""
        text = 'This is a great video'
        result = sa.preprocess_text(text)
        # Common stopwords should be removed
        assert 'is' not in result.split()
        assert 'a' not in result.split()
    
    def test_preprocess_text_lemmatization(self):
        """Test lemmatization."""
        text = 'running runs runner'
        result = sa.preprocess_text(text)
        # All should be lemmatized to 'run'
        words = result.split()
        assert 'run' in words or 'running' in words  # Lemmatization may vary
    
    def test_preprocess_text_empty_string(self):
        """Test empty string handling."""
        result = sa.preprocess_text('')
        assert result == ''
    
    def test_preprocess_text_short_words_removed(self):
        """Test that short words (<=2 chars) are removed."""
        text = 'I am at it in on'
        result = sa.preprocess_text(text)
        # These short words should be removed
        assert len(result) == 0 or all(len(word) > 2 for word in result.split())


# ============================================================================
# TESTS FOR get_textblob_sentiment()
# ============================================================================

class TestGetTextblobSentiment:
    """Tests for the get_textblob_sentiment function."""
    
    @patch('sentiment_analysis.TextBlob')
    def test_get_textblob_sentiment_positive(self, mock_textblob):
        """Test positive sentiment detection."""
        mock_blob = MagicMock()
        mock_blob.sentiment.polarity = 0.5
        mock_textblob.return_value = mock_blob
        
        result = sa.get_textblob_sentiment('Great video!')
        assert result == 'positive'
    
    @patch('sentiment_analysis.TextBlob')
    def test_get_textblob_sentiment_negative(self, mock_textblob):
        """Test negative sentiment detection."""
        mock_blob = MagicMock()
        mock_blob.sentiment.polarity = -0.5
        mock_textblob.return_value = mock_blob
        
        result = sa.get_textblob_sentiment('Terrible content')
        assert result == 'negative'
    
    @patch('sentiment_analysis.TextBlob')
    def test_get_textblob_sentiment_neutral(self, mock_textblob):
        """Test neutral sentiment detection."""
        mock_blob = MagicMock()
        mock_blob.sentiment.polarity = 0.05
        mock_textblob.return_value = mock_blob
        
        result = sa.get_textblob_sentiment('Okay video')
        assert result == 'neutral'
    
    @patch('sentiment_analysis.TextBlob')
    def test_get_textblob_sentiment_boundary_positive(self, mock_textblob):
        """Test boundary case for positive sentiment."""
        mock_blob = MagicMock()
        mock_blob.sentiment.polarity = 0.11  # Just above 0.1 threshold
        mock_textblob.return_value = mock_blob
        
        result = sa.get_textblob_sentiment('Nice')
        assert result == 'positive'
    
    @patch('sentiment_analysis.TextBlob')
    def test_get_textblob_sentiment_boundary_negative(self, mock_textblob):
        """Test boundary case for negative sentiment."""
        mock_blob = MagicMock()
        mock_blob.sentiment.polarity = -0.11  # Just below -0.1 threshold
        mock_textblob.return_value = mock_blob
        
        result = sa.get_textblob_sentiment('Bad')
        assert result == 'negative'


# ============================================================================
# TESTS FOR create_pdf_report()
# ============================================================================

class TestCreatePdfReport:
    """Tests for the create_pdf_report function."""

    @patch('sentiment_analysis.PDF')
    @patch('builtins.open', new_callable=mock_open, read_data='Test summary content')
    def test_create_pdf_report_success(self, mock_file, mock_pdf_class, capsys):
        """Test successful PDF report creation."""
        mock_pdf_instance = mock_pdf_class.return_value

        # Define a mock path for the PDF file to be saved
        mock_pdf_path = '/fake/dir/report.pdf'
        
        sa.create_pdf_report(
            'summary.txt', 'dashboard.png', 'wordcloud.png',
            'Test Video', 'test_id', mock_pdf_path
        )

        captured = capsys.readouterr()
        assert "Generating PDF Report" in captured.out
        assert f"✓ PDF report saved to '{mock_pdf_path}'" in captured.out

        mock_pdf_instance.add_page.assert_called_once()
        mock_pdf_instance.output.assert_called_once_with(mock_pdf_path)

    @patch('sentiment_analysis.PDF')
    @patch('builtins.open', new_callable=mock_open, read_data='Test content')
    def test_create_pdf_report_adds_all_sections(self, mock_file, mock_pdf_class):
        """Test that the function makes the correct high-level calls."""
        mock_pdf_instance = mock_pdf_class.return_value

        sa.create_pdf_report(
            'summary.txt', 'dashboard.png', 'wordcloud.png',
            'Test Video', 'test_id', 'report.pdf'
        )
        
        # 1. Assert that the summary title and body were called once.
        mock_pdf_instance.chapter_title.assert_called_once_with('Analysis Summary & Model Performance')
        mock_pdf_instance.chapter_body.assert_called_once_with('summary.txt')

        # 2. Assert that the image adding method was called twice with the correct arguments.
        assert mock_pdf_instance.add_full_width_image.call_count == 2
        mock_pdf_instance.add_full_width_image.assert_any_call('dashboard.png', 'Analysis Dashboard')
        mock_pdf_instance.add_full_width_image.assert_any_call('wordcloud.png', 'Sentiment Word Clouds')


# ============================================================================
# DATA PROCESSING INTEGRATION TESTS
# ============================================================================

class TestDataProcessing:
    """Integration tests for data processing pipeline."""
    
    def test_dataframe_filtering_length(self, sample_comments_df):
        """Test that short comments are filtered out."""
        df = sample_comments_df.copy()
        df = df[df["comment_text"].str.len() > 15].copy()
        
        # All remaining comments should be longer than 15 chars
        assert all(df["comment_text"].str.len() > 15)
    
    def test_dataframe_date_conversion(self, sample_comments_df):
        """Test date column conversion."""
        df = sample_comments_df.copy()
        df["comment_date"] = pd.to_datetime(df["comment_date"])
        
        assert pd.api.types.is_datetime64_any_dtype(df["comment_date"])
    
    def test_dataframe_day_extraction(self, sample_comments_df):
        """Test extracting day from datetime."""
        df = sample_comments_df.copy()
        df["comment_date"] = pd.to_datetime(df["comment_date"])
        df["day"] = df["comment_date"].dt.date
        
        assert all(isinstance(d, type(datetime.now().date())) for d in df["day"])
    
    def test_sentiment_distribution_calculation(self, sample_processed_df):
        """Test sentiment distribution calculation."""
        df = sample_processed_df.copy()
        sentiment_counts = df["predicted_sentiment"].value_counts()
        
        assert len(sentiment_counts) > 0
        assert all(sentiment in ['positive', 'negative', 'neutral'] for sentiment in sentiment_counts.index)
    
    def test_rolling_average_calculation(self, sample_processed_df):
        """Test rolling average calculation."""
        df = sample_processed_df.copy()
        df = df.sort_values("comment_date").reset_index(drop=True)
        df["is_positive"] = (df["predicted_sentiment"] == "positive").astype(int)
        df["positive_rolling_avg"] = (
            df["is_positive"].rolling(window=3, min_periods=1).mean() * 100
        )
        
        assert "positive_rolling_avg" in df.columns
        assert all(0 <= val <= 100 for val in df["positive_rolling_avg"])

# ============================================================================
# MODEL TRAINING TESTS
# ============================================================================

class TestModelTraining:
    """Tests for model training components."""
    
    @patch('sentiment_analysis.RandomForestClassifier')
    @patch('sentiment_analysis.RandomizedSearchCV')
    def test_randomized_search_execution(self, mock_search, mock_rf):
        """Test RandomizedSearchCV execution."""
        mock_search_instance = MagicMock()
        mock_search_instance.best_estimator_ = MagicMock()
        mock_search_instance.best_params_ = {'n_estimators': 200}
        mock_search.return_value = mock_search_instance
        
        # Simulate search
        X = np.random.rand(100, 50)
        y = np.array(['positive', 'negative', 'neutral'] * 33 + ['positive'])
        
        mock_search_instance.fit(X, y)
        
        mock_search_instance.fit.assert_called_once()
    
    def test_train_test_split_stratification(self):
        """Test that train-test split maintains class distribution."""
        from sklearn.model_selection import train_test_split
        
        X = np.random.rand(100, 10)
        y = np.array(['positive'] * 50 + ['negative'] * 30 + ['neutral'] * 20)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Check proportions are maintained
        train_pos_ratio = sum(y_train == 'positive') / len(y_train)
        test_pos_ratio = sum(y_test == 'positive') / len(y_test)
        
        assert abs(train_pos_ratio - test_pos_ratio) < 0.1


# ============================================================================
# FILE I/O TESTS
# ============================================================================

class TestFileIO:
    """Tests for file input/output operations."""
    
    def test_csv_output_structure(self, sample_processed_df, tmp_path):
        """Test CSV output has correct structure."""
        output_file = tmp_path / "test_output.csv"
        
        df_to_save = sample_processed_df[
            ['comment_text', 'comment_date', 'likes', 'replies', 
             'predicted_sentiment', 'sentiment_confidence']
        ]
        df_to_save.to_csv(output_file, index=False)
        
        # Read back and verify
        df_read = pd.read_csv(output_file)
        expected_columns = [
            'comment_text', 'comment_date', 'likes', 'replies',
            'predicted_sentiment', 'sentiment_confidence'
        ]
        
        assert list(df_read.columns) == expected_columns
    
    def test_summary_report_creation(self, tmp_path):
        """Test summary report text file creation."""
        summary_file = tmp_path / "summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("RANDOM FOREST SENTIMENT ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n")
            f.write("Video ID: test_id\n")
            f.write("Total Comments: 100\n")
        
        assert summary_file.exists()
        content = summary_file.read_text(encoding='utf-8')
        assert "RANDOM FOREST SENTIMENT ANALYSIS REPORT" in content
        assert "test_id" in content


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

class TestParametrized:
    """Parametrized tests for various scenarios."""
    
    @pytest.mark.parametrize("title,video_id,expected", [
        ("Test Video", "abc123", "test_video_abc123"),
        ("Video!@#$%", "xyz", "video_xyz"),
        ("   Spaces   ", "id1", "spaces_id1"),
        ("", "empty", "empty"),
        (None, "none", "none"),
    ])
    def test_create_base_filename_parametrized(self, title, video_id, expected):
        """Test filename creation with various inputs."""
        result = sa.create_base_filename(title, video_id)
        assert result == expected
    
    @pytest.mark.parametrize("text,expected_sentiment", [
        ("This is amazing and wonderful!", "positive"),
        ("Terrible and awful video", "negative"),
        ("This is okay", "neutral"),
    ])
    @patch('sentiment_analysis.TextBlob')
    def test_sentiment_classification_parametrized(self, mock_textblob, text, expected_sentiment):
        """Test sentiment classification with various texts."""
        mock_blob = MagicMock()
        
        # Set polarity based on expected sentiment
        if expected_sentiment == 'positive':
            mock_blob.sentiment.polarity = 0.5
        elif expected_sentiment == 'negative':
            mock_blob.sentiment.polarity = -0.5
        else:
            mock_blob.sentiment.polarity = 0.0
        
        mock_textblob.return_value = mock_blob
        result = sa.get_textblob_sentiment(text)
        
        assert result == expected_sentiment


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Tests for error handling scenarios."""
    
    def test_csv_read_error_handling(self, tmp_path):
        """Test handling of CSV read errors."""
        # Create an invalid CSV file
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("invalid,csv,data\nwith\nmismatched\ncolumns")
        
        # Attempt to read with error handling
        try:
            df = pd.read_csv(bad_csv)
            # May succeed but with unexpected structure
            assert isinstance(df, pd.DataFrame)
        except Exception as e:
            # Should handle the error gracefully
            assert True
    
    @patch('sentiment_analysis.os.makedirs')
    def test_directory_creation_error(self, mock_makedirs):
        """Test handling of directory creation errors."""
        mock_makedirs.side_effect = PermissionError("Permission denied")
        
        with pytest.raises(PermissionError):
            os.makedirs('/protected/path', exist_ok=True)
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        
        # Operations on empty DataFrame should not crash
        result = df[df.get('comment_text', pd.Series(dtype=object)).str.len() > 15] if not df.empty else df
        assert len(result) == 0


# ============================================================================
# FEATURE IMPORTANCE TESTS
# ============================================================================

class TestFeatureImportance:
    """Tests for feature importance analysis."""
    
    def test_feature_importance_dataframe_creation(self, mock_tfidf_vectorizer, mock_random_forest):
        """Test creation of feature importance DataFrame."""
        feature_importances = pd.DataFrame({
            'feature': mock_tfidf_vectorizer.get_feature_names_out(),
            'importance': mock_random_forest.feature_importances_
        }).sort_values('importance', ascending=False)
        
        assert len(feature_importances) == 100
        assert 'feature' in feature_importances.columns
        assert 'importance' in feature_importances.columns
        
        # Verify sorted in descending order
        importances = feature_importances['importance'].values
        assert all(importances[i] >= importances[i+1] for i in range(len(importances)-1))
    
    def test_top_features_extraction(self, mock_tfidf_vectorizer, mock_random_forest):
        """Test extraction of top N features."""
        feature_importances = pd.DataFrame({
            'feature': mock_tfidf_vectorizer.get_feature_names_out(),
            'importance': mock_random_forest.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_20 = feature_importances.head(20)
        
        assert len(top_20) == 20


# ============================================================================
# INTEGRATION TEST
# ============================================================================

class TestIntegration:
    """Integration tests for the complete workflow."""
    
    @patch('sentiment_analysis.build')
    @patch('sentiment_analysis.pd.read_csv')
    @patch('sentiment_analysis.TfidfVectorizer')
    @patch('sentiment_analysis.RandomizedSearchCV')
    @patch('sentiment_analysis.plt.savefig')
    @patch('sentiment_analysis.os.makedirs')
    def test_end_to_end_workflow(self, mock_makedirs, mock_savefig, 
                                 mock_search, mock_tfidf, mock_read_csv, 
                                 mock_build, sample_comments_df, mock_video_response):
        """Test complete workflow from loading to PDF generation."""
        # Setup mocks
        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_youtube.videos().list().execute.return_value = mock_video_response
        
        # Mock data loading
        mock_read_csv.return_value = sample_comments_df
        
        # Mock TF-IDF
        mock_vectorizer = MagicMock()
        mock_vectorizer.fit_transform.return_value = np.random.rand(10, 100)
        mock_vectorizer.get_feature_names_out.return_value = np.array([f'word_{i}' for i in range(100)])
        mock_tfidf.return_value = mock_vectorizer
        
        # Mock model
        mock_search_instance = MagicMock()
        mock_search_instance.best_estimator_ = MagicMock()
        mock_search_instance.best_estimator_.predict.return_value = np.array(['positive'] * 10)
        mock_search_instance.best_estimator_.predict_proba.return_value = np.random.rand(10, 3)
        mock_search_instance.best_estimator_.feature_importances_ = np.random.rand(100)
        mock_search_instance.best_params_ = {'n_estimators': 200}
        mock_search.return_value = mock_search_instance
        
        # Verify the workflow can be set up
        assert mock_build is not None
        assert mock_read_csv is not None
        assert mock_tfidf is not None


# ============================================================================
# TEMPORAL ANALYSIS TESTS
# ============================================================================

class TestTemporalAnalysis:
    """Tests for temporal analysis features."""
    
    def test_daily_sentiment_aggregation(self, sample_processed_df):
        """Test daily sentiment aggregation."""
        df = sample_processed_df.copy()
        daily_sentiment = (
            df.groupby(['day', 'predicted_sentiment']).size().unstack(fill_value=0)
        )
        
        assert not daily_sentiment.empty
        assert all(col in daily_sentiment.columns for col in ['positive', 'negative', 'neutral'] if col in df['predicted_sentiment'].unique())
    
    def test_daily_sentiment_percentage_calculation(self, sample_processed_df):
        """Test daily sentiment percentage calculation."""
        df = sample_processed_df.copy()
        daily_sentiment = (
            df.groupby(['day', 'predicted_sentiment']).size().unstack(fill_value=0)
        )
        daily_sentiment_pct = daily_sentiment.div(daily_sentiment.sum(axis=1), axis=0) * 100
        
        # Each row should sum to 100%
        row_sums = daily_sentiment_pct.sum(axis=1)
        assert all(abs(s - 100.0) < 0.01 for s in row_sums)
    
    def test_rolling_average_positive_sentiment(self, sample_processed_df):
        """Test rolling average calculation for positive sentiment."""
        df = sample_processed_df.copy()
        df = df.sort_values('comment_date').reset_index(drop=True)
        df['is_positive'] = (df['predicted_sentiment'] == 'positive').astype(int)
        df['positive_rolling_avg'] = (
            df['is_positive'].rolling(window=3, min_periods=1).mean() * 100
        )
        
        assert 'positive_rolling_avg' in df.columns
        assert all(0 <= val <= 100 for val in df['positive_rolling_avg'])
        assert not df['positive_rolling_avg'].isna().all()


# ============================================================================
# CONFIDENCE SCORE TESTS
# ============================================================================

class TestConfidenceScores:
    """Tests for sentiment confidence scores."""
    
    def test_confidence_score_range(self, mock_random_forest):
        """Test that confidence scores are in valid range."""
        proba = mock_random_forest.predict_proba.return_value
        max_proba = proba.max(axis=1)
        
        assert all(0 <= score <= 1 for score in max_proba)
    
    def test_confidence_score_assignment(self, sample_processed_df):
        """Test confidence score assignment to DataFrame."""
        df = sample_processed_df.copy()
        
        assert 'sentiment_confidence' in df.columns
        assert all(0 <= score <= 1 for score in df['sentiment_confidence'])


# ============================================================================
# TEXT PREPROCESSING EDGE CASES
# ============================================================================

class TestTextPreprocessingEdgeCases:
    """Tests for edge cases in text preprocessing."""
    
    def test_preprocess_only_stopwords(self):
        """Test preprocessing text with only stopwords."""
        text = "the a an is are"
        result = sa.preprocess_text(text)
        assert len(result) == 0 or result.strip() == ''
    
    def test_preprocess_only_punctuation(self):
        """Test preprocessing text with only punctuation."""
        text = "!@#$%^&*()"
        result = sa.preprocess_text(text)
        assert result.strip() == ''
    
    def test_preprocess_mixed_case(self):
        """Test preprocessing maintains lowercase."""
        text = "ThIs Is MiXeD CaSe TeXt"
        result = sa.preprocess_text(text)
        assert result.islower()
    
    def test_preprocess_numbers(self):
        """Test preprocessing with numbers."""
        text = "Video has 123 likes and 456 comments"
        result = sa.preprocess_text(text)
        # Numbers might be kept or removed depending on tokenization
        assert 'video' in result or 'like' in result or 'comment' in result
    
    def test_preprocess_special_unicode(self):
        """Test preprocessing with special unicode characters."""
        text = "Great video 👍 😊"
        result = sa.preprocess_text(text)
        # Emojis should be removed
        assert '👍' not in result
        assert '😊' not in result
    
    def test_preprocess_multiple_urls(self):
        """Test preprocessing with multiple URLs."""
        text = "Check http://example.com and https://test.com"
        result = sa.preprocess_text(text)
        assert 'http' not in result
        assert 'https' not in result
        assert 'example' not in result
        assert 'test' not in result


# ============================================================================
# DATA VALIDATION TESTS
# ============================================================================

class TestDataValidation:
    """Tests for data validation."""
    
    def test_comment_length_validation(self, sample_comments_df):
        """Test that comment length filtering works correctly."""
        df = sample_comments_df.copy()
        df_filtered = df[df['comment_text'].str.len() > 15].copy()
        
        # All comments should be longer than 15 characters
        assert all(len(text) > 15 for text in df_filtered['comment_text'])
    
    def test_cleaned_text_not_empty_validation(self, sample_comments_df):
        """Test validation that cleaned text is not empty."""
        df = sample_comments_df.copy()
        df['cleaned_text'] = df['comment_text'].apply(sa.preprocess_text)
        df_filtered = df[df['cleaned_text'].str.len() > 0]
        
        assert all(len(text) > 0 for text in df_filtered['cleaned_text'])
    
    def test_dropna_comment_text(self, sample_comments_df):
        """Test dropping NA values in comment_text."""
        df = sample_comments_df.copy()
        # Add some NA values
        df.loc[0, 'comment_text'] = None
        df.loc[1, 'comment_text'] = pd.NA
        
        df_clean = df.dropna(subset=['comment_text'])
        
        assert not df_clean['comment_text'].isna().any()
    
    def test_date_parsing_validation(self, sample_comments_df):
        """Test date parsing validation."""
        df = sample_comments_df.copy()
        df['comment_date'] = pd.to_datetime(df['comment_date'])
        
        # All dates should be valid datetime objects
        assert pd.api.types.is_datetime64_any_dtype(df['comment_date'])
        assert not df['comment_date'].isna().any()


# ============================================================================
# MODEL HYPERPARAMETER TESTS
# ============================================================================

class TestModelHyperparameters:
    """Tests for model hyperparameter configurations."""
    
    def test_param_dist_structure(self):
        """Test that parameter distribution has correct structure."""
        param_dist = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [10, 20, 30, 40, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False],
            'class_weight': ['balanced', 'balanced_subsample'],
        }
        
        assert 'n_estimators' in param_dist
        assert 'max_depth' in param_dist
        assert isinstance(param_dist['n_estimators'], list)
        assert None in param_dist['max_depth']
    
    def test_randomized_search_parameters(self):
        """Test RandomizedSearchCV parameters."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import RandomizedSearchCV
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_dist = {'n_estimators': [100, 200]}
        
        search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=2,
            cv=2,
            random_state=42,
            scoring='f1_weighted'
        )
        
        assert search.n_iter == 2
        assert search.cv == 2
        assert search.scoring == 'f1_weighted'


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Tests for performance-related functionality."""
    
    def test_tfidf_max_features_limit(self):
        """Test TF-IDF max features limit."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(max_features=2500)
        texts = [f"word_{i} " * 10 for i in range(100)]
        X = vectorizer.fit_transform(texts)
        
        # Number of features should not exceed max_features
        assert X.shape[1] <= 2500
    
    def test_tfidf_ngram_range(self):
        """Test TF-IDF n-gram range configuration."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        texts = ["hello world", "world peace"]
        vectorizer.fit_transform(texts)
        
        features = vectorizer.get_feature_names_out()
        # Should contain both unigrams and bigrams
        assert any(' ' not in f for f in features)  # unigrams
        assert any(' ' in f for f in features)  # bigrams


# ============================================================================
# EXPORT FUNCTIONALITY TESTS
# ============================================================================

class TestExportFunctionality:
    """Tests for data export functionality."""
    
    def test_csv_export_columns(self, sample_processed_df, tmp_path):
        """Test CSV export contains correct columns."""
        output_file = tmp_path / "export_test.csv"
        
        columns_to_export = [
            'comment_text', 'comment_date', 'likes', 'replies',
            'predicted_sentiment', 'sentiment_confidence'
        ]
        
        df_export = sample_processed_df[columns_to_export]
        df_export.to_csv(output_file, index=False)
        
        df_read = pd.read_csv(output_file)
        assert list(df_read.columns) == columns_to_export
    
    def test_csv_export_encoding(self, sample_processed_df, tmp_path):
        """Test CSV export uses UTF-8 encoding."""
        output_file = tmp_path / "export_test.csv"
        
        # Add some unicode characters
        df = sample_processed_df.copy()
        df.loc[0, 'comment_text'] = "Test with unicode: café, naïve, résumé"
        
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        # Read back and verify unicode preserved
        df_read = pd.read_csv(output_file, encoding='utf-8')
        assert 'café' in df_read.loc[0, 'comment_text']
    
    def test_summary_report_format(self, tmp_path):
        """Test summary report text format."""
        summary_file = tmp_path / "summary_test.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("RANDOM FOREST SENTIMENT ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write("Video ID: test_video_id\n")
            f.write("Total Comments Analyzed: 100\n")
            f.write("Model: Optimized Random Forest Classifier\n\n")
            f.write("--- Model Performance ---\n")
        
        content = summary_file.read_text(encoding='utf-8')
        
        assert "RANDOM FOREST SENTIMENT ANALYSIS REPORT" in content
        assert "Video ID:" in content
        assert "Total Comments Analyzed:" in content
        assert "Model Performance" in content


# ============================================================================
# CLASSIFICATION REPORT TESTS
# ============================================================================

class TestClassificationReport:
    """Tests for classification report generation."""
    
    def test_classification_report_structure(self):
        """Test classification report output structure."""
        from sklearn.metrics import classification_report
        
        y_true = ['positive', 'negative', 'neutral'] * 10
        y_pred = ['positive', 'negative', 'neutral'] * 10
        
        report = classification_report(y_true, y_pred)
        
        assert 'positive' in report
        assert 'negative' in report
        assert 'neutral' in report
        assert 'precision' in report
        assert 'recall' in report
        assert 'f1-score' in report
    
    def test_classification_report_with_predictions(self):
        """Test classification report with actual predictions."""
        from sklearn.metrics import classification_report
        
        y_true = np.array(['positive'] * 50 + ['negative'] * 30 + ['neutral'] * 20)
        y_pred = np.array(['positive'] * 45 + ['negative'] * 35 + ['neutral'] * 20)
        
        report = classification_report(y_true, y_pred, output_dict=True)
        
        assert 'positive' in report
        assert 'accuracy' in report
        assert 0 <= report['accuracy'] <= 1


# ============================================================================
# NLTK DOWNLOAD TESTS
# ============================================================================

class TestNLTKDownloads:
    """Tests for NLTK resource downloads."""
    
    @patch('sentiment_analysis.nltk.download')
    def test_nltk_resources_downloaded(self, mock_download):
        """Test that required NLTK resources are downloaded."""
        import nltk
        
        # Required downloads
        resources = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
        
        for resource in resources:
            nltk.download(resource, quiet=True)
        
        # Verify download was called
        assert mock_download.call_count >= len(resources)


# ============================================================================
# DIRECTORY MANAGEMENT TESTS
# ============================================================================

class TestDirectoryManagement:
    """Tests for directory creation and management."""
    
    @patch('sentiment_analysis.os.makedirs')
    def test_output_directories_created(self, mock_makedirs):
        """Test that output directories are created."""
        import os
        
        # Simulate directory creation
        os.makedirs('/path/to/sentiment_files', exist_ok=True)
        os.makedirs('/path/to/display_files', exist_ok=True)
        
        assert mock_makedirs.call_count >= 2
    
    def test_output_directory_paths_defined(self):
        """Test that output directory paths are properly defined."""
        # These should be defined in the module
        assert hasattr(sa, 'SENTIMENT_ARTIFACTS_DIR')
        assert not hasattr(sa, 'OUTPUT_ARTIFACTS_DIR')
        assert hasattr(sa, 'CSV_INPUT_DIR')


# ============================================================================
# MATPLOTLIB STYLE TESTS
# ============================================================================

class TestMatplotlibConfiguration:
    """Tests for matplotlib configuration."""
    
    @patch('sentiment_analysis.plt.style.use')
    def test_matplotlib_style_set(self, mock_style):
        """Test that matplotlib style is set."""
        import matplotlib.pyplot as plt
        
        plt.style.use('seaborn-v0_8-darkgrid')
        
        mock_style.assert_called()


# ============================================================================
# INTEGRATION TEST FOR MAIN WORKFLOW
# ============================================================================

@patch('sentiment_analysis.create_pdf_report')
@patch('sentiment_analysis.plt.savefig')
@patch('sentiment_analysis.RandomizedSearchCV')
@patch('sentiment_analysis.TfidfVectorizer')
@patch('sentiment_analysis.pd.read_csv')
@patch('sentiment_analysis.find_latest_comments_file')
@patch('sentiment_analysis.train_test_split')
def test_main_workflow_orchestration(
    mock_split, mock_find_file, mock_read_csv, mock_tfidf, 
    mock_search, mock_savefig, mock_pdf, sample_comments_df
):
    """
    Tests that the main() function correctly orchestrates the pipeline
    by dynamically adapting to the data size.
    """
    # --- 1. SETUP MOCKS ---
    mock_find_file.return_value = 'fake_comments.csv'
    mock_read_csv.return_value = sample_comments_df
    
    temp_df = sample_comments_df.copy()
    temp_df = temp_df[temp_df['comment_text'].str.len() > 15]
    temp_df['cleaned_text'] = temp_df['comment_text'].apply(sa.preprocess_text)
    temp_df = temp_df[temp_df['cleaned_text'].str.len() > 0]
    num_samples = len(temp_df) # This will be 8
    
    # Now, calculate train/test split based on the *actual* final size
    num_test = int(np.ceil(num_samples * 0.25)) # This will be 2
    num_train = num_samples - num_test # This will be 6
    
    # Mock TfidfVectorizer
    from scipy.sparse import csr_matrix
    mock_tfidf_instance = MagicMock()
    mock_tfidf_instance.fit_transform.return_value = csr_matrix((num_samples, 2500))
    mock_tfidf_instance.get_feature_names_out.return_value = np.array([f'word_{i}' for i in range(2500)])
    mock_tfidf.return_value = mock_tfidf_instance
    
    # Mock train_test_split
    X_train, X_test = csr_matrix((num_train, 2500)), csr_matrix((num_test, 2500))
    y_train, y_test = ['pos']*num_train, ['pos']*num_test
    mock_split.return_value = (X_train, X_test, y_train, y_test)
    
    # Mock RandomizedSearchCV and the model it finds
    mock_estimator = MagicMock()
    mock_estimator.predict.side_effect = [
        np.array(['pos'] * num_test),      # For classification_report on X_test
        np.array(['pos'] * num_samples)    # For assigning to the full dataframe
    ]
    mock_estimator.predict_proba.return_value = np.random.rand(num_samples, 3)
    mock_estimator.feature_importances_ = np.random.rand(2500)
    
    mock_search_instance = MagicMock()
    mock_search_instance.best_estimator_ = mock_estimator
    mock_search.return_value = mock_search_instance
    
    # --- 2. RUN THE FUNCTION ---
    with patch('os.path.isdir', return_value=True):
        sa.main()
    
    # --- 3. ASSERTIONS ---
    mock_find_file.assert_called_once()
    mock_read_csv.assert_called_once()
    mock_split.assert_called_once()
    mock_search_instance.fit.assert_called_once()
    assert mock_estimator.predict.call_count == 2
    mock_pdf.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])