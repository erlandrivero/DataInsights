"""
Text Mining and Sentiment Analysis utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import re
import warnings
warnings.filterwarnings('ignore')

# NLP imports
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.sentiment import SentimentIntensityAnalyzer
    
    # Download required NLTK data
    for resource in ['punkt', 'punkt_tab', 'stopwords', 'vader_lexicon']:
        try:
            if resource == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif resource == 'punkt_tab':
                nltk.data.find('tokenizers/punkt_tab')
            elif resource == 'stopwords':
                nltk.data.find('corpora/stopwords')
            else:
                nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download(resource, quiet=True)
        
except Exception as e:
    print(f"NLTK initialization warning: {e}")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except:
    TextBlob = None
    TEXTBLOB_AVAILABLE = False

from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64

class TextAnalyzer:
    """Handles text mining and NLP operations."""
    
    def __init__(self, text_series: pd.Series):
        """
        Initialize TextAnalyzer with a text column.
        
        Args:
            text_series: Pandas Series containing text data
        """
        self.text_series = text_series.dropna().astype(str)
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def get_sentiment_analysis(self) -> pd.DataFrame:
        """
        Perform sentiment analysis using VADER and TextBlob.
        
        Returns:
            DataFrame with sentiment scores and classification
        """
        results = []
        
        for text in self.text_series:
            # VADER sentiment
            vader_scores = self.sia.polarity_scores(text)
            
            # TextBlob sentiment (if available)
            if TEXTBLOB_AVAILABLE:
                try:
                    blob = TextBlob(text)
                    textblob_polarity = blob.sentiment.polarity
                    textblob_subjectivity = blob.sentiment.subjectivity
                except:
                    textblob_polarity = 0
                    textblob_subjectivity = 0
            else:
                textblob_polarity = 0
                textblob_subjectivity = 0
            
            # Classification
            compound = vader_scores['compound']
            if compound >= 0.05:
                sentiment = 'Positive'
            elif compound <= -0.05:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
            
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'vader_compound': vader_scores['compound'],
                'vader_pos': vader_scores['pos'],
                'vader_neg': vader_scores['neg'],
                'vader_neu': vader_scores['neu'],
                'textblob_polarity': textblob_polarity,
                'textblob_subjectivity': textblob_subjectivity,
                'sentiment': sentiment
            })
        
        return pd.DataFrame(results)
    
    def get_word_frequency(self, n_words: int = 50) -> pd.DataFrame:
        """
        Get word frequency counts.
        
        Args:
            n_words: Number of top words to return
            
        Returns:
            DataFrame with words and their frequencies
        """
        all_words = []
        
        for text in self.text_series:
            cleaned = self.clean_text(text)
            tokens = word_tokenize(cleaned)
            tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
            all_words.extend(tokens)
        
        word_freq = pd.Series(all_words).value_counts().head(n_words)
        
        return pd.DataFrame({
            'word': word_freq.index,
            'frequency': word_freq.values
        })
    
    def get_topic_modeling(self, num_topics: int = 5, n_words: int = 10) -> Dict[int, List[str]]:
        """
        Perform topic modeling using LDA.
        
        Args:
            num_topics: Number of topics to extract
            n_words: Number of words per topic
            
        Returns:
            Dictionary mapping topic number to top words
        """
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        
        cleaned_texts = [self.clean_text(text) for text in self.text_series]
        
        vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english',
            min_df=2
        )
        
        doc_term_matrix = vectorizer.fit_transform(cleaned_texts)
        
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            n_jobs=-1
        )
        lda.fit(doc_term_matrix)
        
        feature_names = vectorizer.get_feature_names_out()
        topics = {}
        
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topics[topic_idx] = top_words
        
        return topics
    
    def create_wordcloud(self, max_words: int = 100) -> Any:
        """Generate word cloud image."""
        all_text = ' '.join([self.clean_text(text) for text in self.text_series])
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=max_words,
            stopwords=self.stop_words,
            colormap='viridis'
        ).generate(all_text)
        
        return wordcloud
    
    def create_sentiment_plot(self, sentiment_df: pd.DataFrame) -> go.Figure:
        """Create sentiment distribution plot."""
        sentiment_counts = sentiment_df['sentiment'].value_counts()
        
        colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
        
        fig = go.Figure(data=[go.Bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            marker_color=[colors.get(s, 'blue') for s in sentiment_counts.index]
        )])
        
        fig.update_layout(
            title='Sentiment Distribution',
            xaxis_title='Sentiment',
            yaxis_title='Count',
            height=400
        )
        
        return fig
    
    def create_word_frequency_plot(self, word_freq_df: pd.DataFrame) -> go.Figure:
        """Create word frequency bar chart."""
        fig = go.Figure(data=[go.Bar(
            x=word_freq_df['frequency'],
            y=word_freq_df['word'],
            orientation='h',
            marker_color='lightblue'
        )])
        
        fig.update_layout(
            title='Top Words by Frequency',
            xaxis_title='Frequency',
            yaxis_title='Word',
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
