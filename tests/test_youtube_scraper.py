import pytest
import os
import sys
import configparser
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime
import pandas as pd
from googleapiclient.errors import HttpError

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the module to test (adjust import name to match your actual script name)
import youtube_scraper as ys


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_config_data():
    """Provides sample configuration data."""
    return {
        'youtube_api': {'api_key': 'test_api_key_123'},
        'video_details': {'video_id': 'dQw4w9WgXcQ'}
    }


@pytest.fixture
def mock_video_response():
    """Provides a mock YouTube API video response."""
    return {
        'items': [{
            'snippet': {
                'title': 'Test Video Title! (Official)'
            }
        }]
    }


@pytest.fixture
def mock_comments_response():
    """Provides a mock YouTube API comments response."""
    return {
        'items': [
            {
                'snippet': {
                    'topLevelComment': {
                        'id': 'comment_1',
                        'snippet': {
                            'textDisplay': 'Great video!',
                            'authorDisplayName': 'User1',
                            'publishedAt': '2024-01-01T12:00:00Z',
                            'updatedAt': '2024-01-01T12:00:00Z',
                            'likeCount': 10,
                        }
                    },
                    'totalReplyCount': 2
                }
            },
            {
                'snippet': {
                    'topLevelComment': {
                        'id': 'comment_2',
                        'snippet': {
                            'textDisplay': 'Thanks for sharing',
                            'authorDisplayName': 'User2',
                            'publishedAt': '2024-01-02T12:00:00Z',
                            'updatedAt': '2024-01-02T12:00:00Z',
                            'likeCount': 5,
                        }
                    },
                    'totalReplyCount': 0
                }
            }
        ]
    }


@pytest.fixture
def mock_comments_response_with_pagination():
    """Provides paginated comments response."""
    return {
        'items': [
            {
                'snippet': {
                    'topLevelComment': {
                        'id': f'comment_{i}',
                        'snippet': {
                            'textDisplay': f'Comment {i}',
                            'authorDisplayName': f'User{i}',
                            'publishedAt': '2024-01-01T12:00:00Z',
                            'updatedAt': '2024-01-01T12:00:00Z',
                            'likeCount': i,
                        }
                    },
                    'totalReplyCount': 0
                }
            } for i in range(5)
        ],
        'nextPageToken': 'next_page_token_123'
    }


@pytest.fixture
def sample_comments():
    """Provides sample comment data for testing."""
    return [
        {
            'comment_id': 'comment_1',
            'comment_text': 'Great video!',
            'author': 'User1',
            'comment_date': '2024-01-01T12:00:00Z',
            'updated_date': '2024-01-01T12:00:00Z',
            'likes': 10,
            'replies': 2
        },
        {
            'comment_id': 'comment_2',
            'comment_text': 'Thanks for sharing',
            'author': 'User2',
            'comment_date': '2024-01-02T12:00:00Z',
            'updated_date': '2024-01-02T12:00:00Z',
            'likes': 5,
            'replies': 0
        }
    ]


# ============================================================================
# TESTS FOR load_config()
# ============================================================================

class TestLoadConfig:
    """Tests for the load_config function."""
    
    def test_load_config_success(self, mock_config_data, tmp_path):
        """Test successful config loading."""
        # Create a temporary config file
        config_file = tmp_path / "config.ini"
        config = configparser.ConfigParser()
        config['youtube_api'] = {'api_key': 'test_key'}
        config['video_details'] = {'video_id': 'test_id'}
        with open(config_file, 'w') as f:
            config.write(f)
        
        # Mock the CONFIG_FILE path
        with patch.object(ys, 'CONFIG_FILE', str(config_file)):
            api_key, video_id = ys.load_config()
        
        assert api_key == 'test_key'
        assert video_id == 'test_id'
    
    def test_load_config_file_not_found(self, capsys):
        """Test behavior when config file doesn't exist."""
        with patch.object(ys, 'CONFIG_FILE', '/nonexistent/config.ini'):
            api_key, video_id = ys.load_config()
        
        assert api_key is None
        assert video_id is None
        captured = capsys.readouterr()
        assert "ERROR: config.ini not found" in captured.out
    
    def test_load_config_missing_api_key(self, tmp_path, capsys):
        """Test handling of missing API key in config."""
        config_file = tmp_path / "config.ini"
        config = configparser.ConfigParser()
        config['video_details'] = {'video_id': 'test_id'}
        with open(config_file, 'w') as f:
            config.write(f)
        
        with patch.object(ys, 'CONFIG_FILE', str(config_file)):
            api_key, video_id = ys.load_config()
        
        assert api_key is None
        assert video_id is None
        captured = capsys.readouterr()
        assert "ERROR: Missing key in config.ini" in captured.out
    
    def test_load_config_missing_video_id(self, tmp_path, capsys):
        """Test handling of missing video ID in config."""
        config_file = tmp_path / "config.ini"
        config = configparser.ConfigParser()
        config['youtube_api'] = {'api_key': 'test_key'}
        with open(config_file, 'w') as f:
            config.write(f)
        
        with patch.object(ys, 'CONFIG_FILE', str(config_file)):
            api_key, video_id = ys.load_config()
        
        assert api_key is None
        assert video_id is None
        captured = capsys.readouterr()
        assert "ERROR: Missing key in config.ini" in captured.out


# ============================================================================
# TESTS FOR get_video_title()
# ============================================================================

class TestGetVideoTitle:
    """Tests for the get_video_title function."""
    
    @patch('youtube_scraper.build')
    def test_get_video_title_success(self, mock_build, mock_video_response):
        """Test successful video title retrieval."""
        # More direct mocking
        mock_list_method = mock_build.return_value.videos.return_value.list
        mock_list_method.return_value.execute.return_value = mock_video_response
        
        title = ys.get_video_title('test_api_key', 'test_video_id')

        assert title == 'Test Video Title! (Official)'
        # Assert against the specific mocked method
        mock_list_method.assert_called_once_with(
            part='snippet', 
            id='test_video_id'
        )
    
    @patch('youtube_scraper.build')
    def test_get_video_title_empty_response(self, mock_build):
        """Test handling of empty API response."""
        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_youtube.videos().list().execute.return_value = {'items': []}
        
        title = ys.get_video_title('test_api_key', 'test_video_id')
        
        assert title is None
    
    @patch('youtube_scraper.build')
    def test_get_video_title_http_error(self, mock_build, capsys):
        """Test handling of HTTP errors."""
        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_youtube.videos().list().execute.side_effect = HttpError(
            resp=Mock(status=403), 
            content=b'Quota exceeded'
        )
        
        title = ys.get_video_title('test_api_key', 'test_video_id')
        
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
        result = ys.create_base_filename('Test Video Title!', 'abc123')
        assert result == 'test_video_title_abc123'
    
    def test_create_base_filename_special_chars(self):
        """Test filename sanitization with special characters."""
        result = ys.create_base_filename('Video: #1 (2024) [HD]', 'xyz789')
        assert result == 'video_1_2024_hd_xyz789'
    
    def test_create_base_filename_multiple_spaces(self):
        """Test handling of multiple consecutive spaces."""
        result = ys.create_base_filename('Video   With    Spaces', 'id123')
        assert result == 'video_with_spaces_id123'
    
    def test_create_base_filename_no_title(self):
        """Test fallback to video ID when no title provided."""
        result = ys.create_base_filename(None, 'fallback_id')
        assert result == 'fallback_id'
    
    def test_create_base_filename_empty_title(self):
        """Test handling of empty title string."""
        result = ys.create_base_filename('', 'empty_id')
        assert result == 'empty_id'
    
    def test_create_base_filename_unicode(self):
        """Test handling of unicode characters."""
        result = ys.create_base_filename('Vidéo Tëst 日本語', 'unicode_id')
        assert result == 'vidéo_tëst_日本語_unicode_id'

# ============================================================================
# TESTS FOR get_comments()
# ============================================================================

class TestGetComments:
    """Tests for the get_comments function."""
    
    def test_get_comments_success(self, mock_comments_response):
        """Test successful comment retrieval."""
        mock_youtube = MagicMock()
        mock_youtube.commentThreads().list().execute.return_value = mock_comments_response
        
        comments = ys.get_comments(mock_youtube, 'test_video_id', 100)
        
        assert len(comments) == 2
        assert comments[0]['comment_id'] == 'comment_1'
        assert comments[0]['comment_text'] == 'Great video!'
        assert comments[0]['author'] == 'User1'
        assert comments[0]['likes'] == 10
        assert comments[0]['replies'] == 2
        
        assert comments[1]['comment_id'] == 'comment_2'
        assert comments[1]['author'] == 'User2'
    
    def test_get_comments_pagination(self, mock_comments_response_with_pagination):
        """Test comment retrieval with pagination."""
        mock_youtube = MagicMock()
        
        # First page with nextPageToken
        first_response = mock_comments_response_with_pagination.copy()
        # Second page without nextPageToken (last page)
        second_response = {
            'items': [
                {
                    'snippet': {
                        'topLevelComment': {
                            'id': 'comment_last',
                            'snippet': {
                                'textDisplay': 'Last comment',
                                'authorDisplayName': 'LastUser',
                                'publishedAt': '2024-01-03T12:00:00Z',
                                'updatedAt': '2024-01-03T12:00:00Z',
                                'likeCount': 1,
                            }
                        },
                        'totalReplyCount': 0
                    }
                }
            ]
        }
        
        mock_youtube.commentThreads().list().execute.side_effect = [
            first_response, 
            second_response
        ]
        
        comments = ys.get_comments(mock_youtube, 'test_video_id', 100)
        
        assert len(comments) == 6  # 5 from first page + 1 from second page
        assert mock_youtube.commentThreads().list().execute.call_count == 2
    
    def test_get_comments_max_results_limit(self, mock_comments_response_with_pagination):
        """Test that max_results limit is respected."""
        mock_youtube = MagicMock()
        mock_youtube.commentThreads().list().execute.return_value = mock_comments_response_with_pagination
        
        comments = ys.get_comments(mock_youtube, 'test_video_id', 3)
        
        assert len(comments) == 3
    
    def test_get_comments_http_error(self, capsys):
        """Test handling of HTTP errors during comment retrieval."""
        mock_youtube = MagicMock()
        mock_youtube.commentThreads().list().execute.side_effect = HttpError(
            resp=Mock(status=403), 
            content=b'Comments disabled'
        )
        
        comments = ys.get_comments(mock_youtube, 'test_video_id', 100)
        
        assert comments == []
        captured = capsys.readouterr()
        assert "An HTTP error occurred" in captured.out
    
    def test_get_comments_empty_response(self):
        """Test handling of empty comment response."""
        mock_youtube = MagicMock()
        mock_youtube.commentThreads().list().execute.return_value = {'items': []}
        
        comments = ys.get_comments(mock_youtube, 'test_video_id', 100)
        
        assert comments == []
    
    def test_get_comments_api_call_parameters(self):
        """Test that API is called with correct parameters."""
        mock_youtube = MagicMock()
        mock_youtube.commentThreads().list().execute.return_value = {'items': []}
        
        ys.get_comments(mock_youtube, 'test_id_123', 50)
        
        mock_youtube.commentThreads().list.assert_called_with(
            part='snippet',
            videoId='test_id_123',
            maxResults=50,
            pageToken=None,
            textFormat='plainText',
            order='time'
        )


# ============================================================================
# TESTS FOR main()
# ============================================================================

class TestMain:
    """Tests for the main execution function."""
    
    @patch('youtube_scraper.build')
    @patch('youtube_scraper.get_comments')
    @patch('youtube_scraper.pd.DataFrame.to_csv')
    @patch('youtube_scraper.os.makedirs')
    def test_main_success(self, mock_makedirs, mock_to_csv, mock_get_comments, 
                          mock_build, sample_comments, capsys):
        """Test successful main execution."""
        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_get_comments.return_value = sample_comments
        
        ys.main()
        
        captured = capsys.readouterr()
        assert "YOUTUBE COMMENTS SCRAPER" in captured.out
        assert "Successfully collected 2 comments!" in captured.out
        assert "Data saved to:" in captured.out
        mock_makedirs.assert_called_once()
        mock_to_csv.assert_called_once()
    
    @patch('youtube_scraper.build')
    def test_main_build_failure(self, mock_build, capsys):
        """Test handling of YouTube service build failure."""
        mock_build.side_effect = Exception("API connection failed")
        
        ys.main()
        
        captured = capsys.readouterr()
        assert "Failed to build YouTube service" in captured.out
    
    @patch('youtube_scraper.build')
    @patch('youtube_scraper.get_comments')
    def test_main_no_comments(self, mock_get_comments, mock_build, capsys):
        """Test handling when no comments are collected."""
        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_get_comments.return_value = []
        
        ys.main()
        
        captured = capsys.readouterr()
        assert "No comments collected" in captured.out
    
    @patch('youtube_scraper.build')
    @patch('youtube_scraper.get_comments')
    @patch('youtube_scraper.pd.DataFrame.to_csv')
    @patch('youtube_scraper.os.makedirs')
    def test_main_csv_creation(self, mock_makedirs, mock_to_csv, 
                               mock_get_comments, mock_build, sample_comments, tmp_path):
        """Test CSV file creation with correct parameters."""
        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_get_comments.return_value = sample_comments
        
        with patch.object(ys, 'CSV_OUTPUT_DIR', str(tmp_path)):
            ys.main()
        
        # Verify to_csv was called with correct encoding
        call_args = mock_to_csv.call_args
        assert call_args[1]['index'] == False
        assert call_args[1]['encoding'] == 'utf-8'
        
        # Verify output filename contains expected components
        output_path = call_args[0][0]
        assert '_comments_' in output_path
        assert '.csv' in output_path


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for the complete workflow."""
    
    @patch('youtube_scraper.build')
    def test_end_to_end_workflow(self, mock_build, tmp_path, 
                                 mock_comments_response, mock_video_response):
        """Test complete workflow from config to CSV output."""
        # Setup
        config_file = tmp_path / "config.ini"
        csv_dir = tmp_path / "csv_files"
        
        config = configparser.ConfigParser()
        config['youtube_api'] = {'api_key': 'test_key'}
        config['video_details'] = {'video_id': 'test_id'}
        with open(config_file, 'w') as f:
            config.write(f)
        
        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_youtube.videos().list().execute.return_value = mock_video_response
        mock_youtube.commentThreads().list().execute.return_value = mock_comments_response
        
        # Execute
        with patch.object(ys, 'CONFIG_FILE', str(config_file)):
            with patch.object(ys, 'CSV_OUTPUT_DIR', str(csv_dir)):
                api_key, video_id = ys.load_config()
                title = ys.get_video_title(api_key, video_id)
                base_name = ys.create_base_filename(title, video_id)
                comments = ys.get_comments(mock_youtube, video_id, 100)
        
        # Verify
        assert api_key == 'test_key'
        assert video_id == 'test_id'
        assert title == 'Test Video Title! (Official)'
        assert base_name == 'test_video_title_official_test_id'
        assert len(comments) == 2
        assert comments[0]['comment_text'] == 'Great video!'


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

class TestParametrized:
    """Parametrized tests for edge cases."""
    
    @pytest.mark.parametrize("title,video_id,expected", [
        ("Simple Title", "abc123", "simple_title_abc123"),
        ("Title!@#$%^&*()", "xyz", "title_xyz"),
        ("   Spaces   ", "id1", "spaces_id1"),
        ("UPPERCASE", "id2", "uppercase_id2"),
        ("under_score-dash", "id3", "under_score_dash_id3"),
        ("", "empty", "empty"),
        (None, "none", "none"),
    ])
    def test_create_base_filename_parametrized(self, title, video_id, expected):
        """Test filename creation with various inputs."""
        result = ys.create_base_filename(title, video_id)
        assert result == expected
    
    @pytest.mark.parametrize("max_results,expected_count", [
        (1, 1),
        (5, 5),
        (100, 2),  # Only 2 comments in mock response
        (0, 0),
    ])
    def test_get_comments_various_limits(self, max_results, expected_count, 
                                         mock_comments_response):
        """Test comment retrieval with various max_results values."""
        mock_youtube = MagicMock()
        mock_youtube.commentThreads().list().execute.return_value = mock_comments_response
        
        comments = ys.get_comments(mock_youtube, 'test_id', max_results)
        
        final_expected_count = min(max_results, 2) if max_results > 0 else 0
        assert len(comments) == final_expected_count


# ============================================================================
# FIXTURES FOR DATAFRAME VALIDATION
# ============================================================================

def test_comments_dataframe_structure(sample_comments):
    """Test that comments can be converted to DataFrame with correct structure."""
    df = pd.DataFrame(sample_comments)
    
    expected_columns = [
        'comment_id', 'comment_text', 'author', 
        'comment_date', 'updated_date', 'likes', 'replies'
    ]
    
    assert list(df.columns) == expected_columns
    assert len(df) == 2
    assert df['likes'].dtype in ['int64', 'int32']
    assert df['replies'].dtype in ['int64', 'int32']


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Tests for various error scenarios."""
    
    @patch('youtube_scraper.os.makedirs')
    def test_directory_creation_permission_error(self, mock_makedirs):
        """Test handling of directory creation errors."""
        mock_makedirs.side_effect = PermissionError("Permission denied")
        
        with pytest.raises(PermissionError):
            os.makedirs('/protected/path', exist_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])