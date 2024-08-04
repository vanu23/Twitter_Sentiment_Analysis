# Twitter_Sentiment_Analysis

# Table of Contents
1. Introduction
2. Features
3. Installation
4. Usage
5. Dataset
6. Model
7. Results


# Introduction
    Twitter Sentiment Analysis is a project that aims to analyze the sentiment of tweets. This project uses Natural Language Processing (NLP) techniques and machine learning algorithms to classify the sentiment of tweets as positive, negative, or neutral.

# Features
1.Preprocessing of tweets (tokenization, removing stop words, etc.)
2.Sentiment classification using machine learning models
3.Visualization of sentiment distribution
4.Easy-to-use interface for analyzing new tweets

# Installation
  1. Clone the repository:
        git clone https://github.com/vanu23/twitter-sentiment-analysis.git
  2. Change to the project directory:
        cd twitter-sentiment-analysis
  3. Create a virtual environment:
        python -m venv venv
  4. Activate the virtual environment:
        On Windows:
            venv\Scripts\activate
        On macOS/Linux:
            source venv/bin/activate
  5. Install the required packages:
        pip install -r requirements.txt

# Usage
  1. Obtain Twitter API credentials and update the config.py file with your API keys.
  2. Run the sentiment analysis script:
        python analyze.py
  3. To analyze a specific tweet, use the following command:
        python analyze.py --tweet "Your tweet here"

# Dataset
    The dataset used for training the sentiment analysis model is the Sentiment140 dataset, which contains 1.6 million tweets labeled as positive, negative, or neutral.

# Model
    The sentiment analysis model is built using a combination of NLP techniques and machine learning algorithms. The current implementation uses a Logistic Regression model trained on the Sentiment140 dataset.

# Results
    The model achieves an accuracy of approximately 80% on the test dataset. The following chart shows the distribution of sentiments in the analyzed tweets:
