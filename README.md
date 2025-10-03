# YouTube Comment Sentiment Analysis Pipeline

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a complete, end-to-end pipeline for conducting a detailed sentiment analysis on YouTube comments using a Random Forest classifier. The system is fully automated: it scrapes comments from a target video, preprocesses the text, trains and optimizes a machine learning model, and generates a suite of analytical artifacts, including a final multi-page PDF report summarizing all findings.

The entire workflow is designed to be modular, reusable, and configurable, making it easy to analyze any YouTube video with a predominantly English-speaking audience.

## Key Features

- ⚙️ **Dynamic & Reusable**: Target any YouTube video by changing a single ID in the `config.ini` file. All output filenames are generated dynamically based on the video's title.
- 🚀 **Fully Automated Pipeline**: A simple, two-script process (`scrape` then `analyze`) handles everything from data collection to final report generation.
- 📂 **Organized File Structure**: All raw data, analysis artifacts, and source code are automatically saved to their own dedicated directories (`csv_files/`, `sentiment_files/`, `display_files/`,`src/`), keeping the project workspace clean and predictable.
- 🤖 **Optimized Machine Learning**: Utilizes `RandomizedSearchCV` to perform hyperparameter tuning, ensuring the best possible Random Forest model is trained for each specific dataset.
- 📊 **Robust Data Preprocessing**: The pipeline automatically filters for English-language comments, removes noise, and applies standard NLP techniques (lemmatization, stopword removal) for high-quality model training.
- 📄 **Automated PDF Reporting**: The final and most powerful feature—the analysis script automatically generates a multi-page PDF report summarizing the entire analysis, complete with tables and visualizations.

## Project Structure

The repository is organized into a clean and logical structure:

```
.
├── config.ini          # <-- Configuration file for API key and Video ID
├── csv_files/          # <-- All raw scraped comment data is saved here
├── sentiment_files/    # <-- All final analysis artifacts are saved here
├── display_files/      # <-- All final pdf artifacts are saved here
├── src/                # <-- All Python source code lives here
│   ├── youtube_scraper.py
│   └── sentiment_analysis.py
├── .gitignore          # <-- Specifies which files/folders to ignore in version control
├── pyproject.toml      # <-- Project metadata and dependencies for uv/pip
└── README.md           # <-- You are here
```

## Setup and Installation

Follow these steps to set up the project environment. This project uses `uv` for fast and efficient package management.

**1. Clone the Repository**

```bash
git clone https://github.com/abinghamwalker/youtube_sentiment.git
cd youtube_sentiment
```

**2. Create and Sync the Virtual Environment**
This command creates a virtual environment named `.venv`

```bash
uv venv
```

**3. Activate Virtual Environment**
This command activates the environment.

_On Linux/macOS:_

```bash
source .venv/bin/activate
```

_On Windows:_

```powershell
.venv\Scripts\Activate.ps1
```

**4. Install Dependencies**
This command installs the dependencies from the `pyproject.toml` file

```bash
uv pip sync
```

**5. Configure API Key and Video ID**
The scraper needs a YouTube Data API v3 key to function.

- Create a file named `config.ini` in the root of the project directory.
- Copy the following template into your `config.ini` and fill in your details. **This file is listed in `.gitignore` and will not be committed.**

  ```ini
  [youtube_api]
  api_key = YOUR_YOUTUBE_API_KEY_HERE

  [video_details]
  # Example: Demon Slayer Infinity Castle Trailer
  video_id = x7uLutVRBfI
  ```

**6. NLTK Data**
The analysis script will automatically download the necessary NLTK data packages (`punkt`, `stopwords`, etc.) on its first run. No manual action is needed.

## How to Use the Pipeline

The workflow is a simple two-step process executed from the project's root directory.

### Step 1: Scrape the YouTube Comments

Run the scraper script. It will read your `config.ini`, fetch the video's title to create a unique filename, and save the raw comments to the `csv_files/` directory.

```bash
python src/youtube_scraper.py
```

**Expected Output:**

```
==================================================
YOUTUBE COMMENTS SCRAPER
==================================================
Configured for Video ID: x7uLutVRBfI
Saving files with base name: 'demon_slayer_...'
...
✓ Data saved to: csv_files/demon_slayer_..._comments_20231028_100000.csv
```

### Step 2: Run the Sentiment Analysis

Once the data is scraped, run the analysis script. It will automatically find the latest data file in `csv_files/`, process it, train the model, and save all analysis artifacts—including the final PDF report—to the `sentiment_files/` directory.

```bash
python src/sentiment_analysis.py
```

**Expected Output:**

```
--- Analysis configured for Video ID: x7uLutVRBfI ---
...
--- 3. Optimizing Random Forest Hyperparameters ---
Fitting 3 folds for each of 50 candidates...
...
--- 8. Generating PDF Report ---
✓ PDF report saved to 'sentiment_files/demon_slayer_..._full_report.pdf'

============================================================
ANALYSIS COMPLETE. ALL DELIVERABLES GENERATED.
============================================================
```

## Deliverables & Outputs

After running the full pipeline, the `sentiment_files/` directory will contain a complete set of dynamically named deliverables:

1.  **Full PDF Report** (`..._full_report.pdf`)

    - A multi-page, automatically generated PDF containing the summary report, the main analysis dashboard, and the sentiment word clouds. This is the primary deliverable.

2.  **Annotated CSV File** (`..._annotated_comments.csv`)

    - The final dataset with predicted sentiment labels (`positive`, `negative`, `neutral`) and confidence scores for every comment.

3.  **Analysis Dashboard** (`..._dashboard.png`)

    - A comprehensive plot visualizing:
      - Top 20 most important words/phrases influencing sentiment.# Setup Instructions

4.  **Word Clouds** (`..._wordclouds.png`)

    - Visual representations of the most frequent words found in positive, negative, and neutral comments.

5.  **Summary Report** (`..._summary_report.txt`)
    - A text file containing the model's performance metrics (precision, recall, f1-score), best hyperparameters, and the top 20 influential phrases.

## Methodology

The pipeline follows a standard machine learning workflow:

1.  **Data Collection**: The `youtube_scraper.py` script uses the YouTube Data API v3 to collect comments.
2.  **Preprocessing & Cleaning**: The analysis script performs several crucial cleaning steps:
    - Filters out non-English comments using `langdetect`.
    - Removes short, noisy comments.
    - Normalizes text to lowercase, removes URLs and punctuation, tokenizes, removes stopwords, and applies lemmatization.
3.  **Initial Labeling**: The rule-based `TextBlob` library provides initial sentiment labels, which serve as the ground truth for training the model.
4.  **Feature Engineering**: The cleaned text is converted into a numerical format using `TfidfVectorizer`, representing the importance of words and 2-word phrases.
5.  **Modeling & Optimization**: A `RandomForestClassifier` is trained. `RandomizedSearchCV` is used to efficiently find the optimal set of hyperparameters, maximizing the model's performance (measured by weighted F1-score).
6.  **Analysis & Reporting**: The final, optimized model is used to predict sentiment on the entire dataset. The model's feature importances are extracted to identify key phrases, and temporal features are analyzed. All findings are then programmatically assembled into a final PDF report.

## Code Quality and Tooling

This project adheres to modern, professional Python development standards to ensure code quality, consistency, and maintainability.

- **Dependency Management**: All project dependencies are managed using `uv` with a `pyproject.toml` file, ensuring a reproducible and fast environment setup.
- **Code Formatting**: The entire codebase is automatically formatted using **`black`**, the uncompromising code formatter. This ensures a consistent style across all modules.
- **Linting & Error Checking**: The code is statically analyzed using **`ruff`**, an extremely fast Python linter. `ruff` helps catch potential bugs, stylistic errors, and unused imports before the code is even run.

**Check for any linting issues**

```bash
ruff check src/
```

**Auto-format all code in the src directory**

```bash
black src/
```

## Testing and Code Verification

This project is rigorously tested to ensure the reliability and correctness of its core logic. The test suite is built using `pytest` and achieves high unit test coverage on critical components.

- **Unit Tests**: The `tests/` directory contains a comprehensive suite of unit tests that verify each function in isolation. This includes testing utility functions, data preprocessing logic, and error handling. The core data scraping logic in `src/youtube_scraper.py` maintains **98% test coverage**.
- **Code Coverage**: Test coverage is measured using the `coverage` package. While the overall coverage for `src/sentiment_analysis.py` is ~52%, this is by design. The unit tests focus on the complex helper functions (e.g., `preprocess_text`, `is_english`). The remaining uncovered lines reside within the main execution workflow (`main()` function), which handles the time-consuming model training and plotting. This part of the pipeline is tested via end-to-end execution rather than unit tests.
- **Code Quality**: The entire codebase is formatted with `black` and linted with `ruff` to maintain consistency and prevent common errors.

To run the full test suite and view the coverage report:

```bash
# Run all tests
python -m pytest

# Run tests and generate a coverage report
coverage run -m pytest
coverage report -m
```
