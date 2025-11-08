"""Text Mining and Sentiment Analysis utilities for DataInsights.

This module provides comprehensive text mining capabilities including sentiment
analysis, word frequency analysis, topic modeling, and word cloud generation.

Author: DataInsights Team
Phase 2 Enhancement: Oct 23, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import re
import warnings
warnings.filterwarnings('ignore')

# Lazy loading for sklearn
from utils.lazy_loader import LazyModuleLoader

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
    """Handles text mining and natural language processing operations.
    
    This class provides comprehensive text analysis capabilities including
    sentiment analysis using VADER, word frequency analysis, topic modeling
    with LDA, and word cloud generation.
    
    Attributes:
        text_series (pd.Series): Clean text data (non-null, string type)
        stop_words (set): English stopwords from NLTK
        sia (SentimentIntensityAnalyzer): VADER sentiment analyzer
    
    Example:
        >>> # Basic text analysis workflow
        >>> df = pd.read_csv('reviews.csv')
        >>> analyzer = TextAnalyzer(df['review_text'])
        >>> 
        >>> # Sentiment analysis
        >>> sentiments = analyzer.get_sentiment_analysis(max_samples=1000)
        >>> print(sentiments['sentiment'].value_counts())
        >>> 
        >>> # Word frequency
        >>> word_freq = analyzer.get_word_frequency(n_words=20)
        >>> st.dataframe(word_freq)
        >>> 
        >>> # Topic modeling
        >>> topics = analyzer.get_topic_modeling(num_topics=5)
        >>> for topic_id, words in topics.items():
        >>>     print(f"Topic {topic_id}: {', '.join(words)}")
    
    Note:
        - Automatically downloads required NLTK data on first use
        - Uses sampling for large datasets (>5000 texts) for performance
        - VADER is optimized for social media and short texts
    """
    
    def __init__(self, text_series: pd.Series):
        """Initialize TextAnalyzer with a text column.
        
        Prepares text data for analysis by removing nulls and converting
        to strings. Initializes NLTK stopwords and VADER analyzer.
        
        Args:
            text_series: Pandas Series containing text data
                        (nulls will be removed, all values converted to str)
        
        Example:
            >>> # Initialize with DataFrame column
            >>> analyzer = TextAnalyzer(df['comments'])
            >>> 
            >>> # Or with Series
            >>> reviews = pd.Series(['Great!', 'Bad product', 'Average'])
            >>> analyzer = TextAnalyzer(reviews)
        
        Note:
            - Non-null values are automatically retained
            - All values converted to strings
            - English stopwords loaded from NLTK
        """
        self.text_series: pd.Series = text_series.dropna().astype(str)
        self.stop_words: set = set(stopwords.words('english'))
        self.sia: SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis.
        
        Performs comprehensive text cleaning including:
        - Lowercase conversion
        - URL removal
        - Email removal
        - Non-alphabetic character removal
        - Whitespace normalization
        
        Args:
            text: Raw text string to clean
        
        Returns:
            Cleaned text with lowercase, no URLs/emails/special chars
        
        Example:
            >>> analyzer = TextAnalyzer(df['text'])
            >>> cleaned = analyzer.clean_text("Check out https://example.com! Email: test@test.com")
            >>> print(cleaned)
            check out  email
        
        Note:
            - Preserves spaces between words
            - Removes all punctuation and numbers
            - Does NOT remove stopwords (use separate filtering)
        """
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def get_sentiment_analysis(self, max_samples: int = 5000) -> pd.DataFrame:
        """Perform sentiment analysis using VADER algorithm.
        
        Analyzes text sentiment using VADER (Valence Aware Dictionary and
        sEntiment Reasoner), which is optimized for social media text and
        performs well on short texts without training data.
        
        Args:
            max_samples: Maximum number of texts to analyze for performance
                        (default: 5000). Larger datasets are randomly sampled.
        
        Returns:
            DataFrame with columns:
                - text (str): First 100 chars of original text
                - compound (float): Overall sentiment score (-1 to +1)
                - positive (float): Positive sentiment proportion (0 to 1)
                - negative (float): Negative sentiment proportion (0 to 1)
                - neutral (float): Neutral sentiment proportion (0 to 1)
                - sentiment (str): Classification ('Positive', 'Negative', 'Neutral')
        
        Example:
            >>> sentiments = analyzer.get_sentiment_analysis(max_samples=1000)
            >>> 
            >>> # View distribution
            >>> print(sentiments['sentiment'].value_counts())
            >>> 
            >>> # Find most positive
            >>> most_positive = sentiments.nlargest(5, 'compound')
            >>> print(most_positive[['text', 'compound']])
        
        Note:
            - Compound >= 0.05: Positive
            - Compound <= -0.05: Negative
            - Between -0.05 and 0.05: Neutral
            - VADER doesn't require training data
            - Uses sampling for datasets >5000 texts
        """
        # Sample if dataset is large
        if len(self.text_series) > max_samples:
            text_sample = self.text_series.sample(n=max_samples, random_state=42)
            print(f"Analyzing {max_samples} sampled texts out of {len(self.text_series)} total")
        else:
            text_sample = self.text_series
        
        # Use pandas apply for better performance
        def analyze_text(text: str) -> pd.Series:
            scores = self.sia.polarity_scores(str(text))
            compound = scores['compound']
            
            # Classification based on compound score
            if compound >= 0.05:
                sentiment = 'Positive'
            elif compound <= -0.05:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
            
            return pd.Series({
                'text': text[:100] + '...' if len(str(text)) > 100 else text,
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'sentiment': sentiment
            })
        
        # Apply analysis (vectorized, much faster than loop)
        results = text_sample.apply(analyze_text)
        
        return results
    
    def get_word_frequency(
        self, 
        n_words: int = 50, 
        max_samples: int = 5000
    ) -> pd.DataFrame:
        """Calculate word frequency counts for most common words.
        
        Extracts and counts the most frequently occurring words after
        cleaning and filtering stopwords.
        
        Args:
            n_words: Number of top words to return (default: 50)
            max_samples: Maximum texts to process for performance (default: 5000)
        
        Returns:
            DataFrame with columns:
                - word (str): The word
                - frequency (int): Number of occurrences
            Sorted by frequency (descending)
        
        Example:
            >>> word_freq = analyzer.get_word_frequency(n_words=20)
            >>> 
            >>> # Create visualization
            >>> fig = analyzer.create_word_frequency_plot(word_freq)
            >>> st.plotly_chart(fig)
            >>> 
            >>> # Find specific words
            >>> product_words = word_freq[word_freq['word'].str.contains('product')]
        
        Note:
            - Automatically removes stopwords (the, a, is, etc.)
            - Filters words shorter than 3 characters
            - Uses sampling for large datasets
            - Case-insensitive (all lowercase)
        """
        # Sample if dataset is large
        if len(self.text_series) > max_samples:
            text_sample = self.text_series.sample(n=max_samples, random_state=42)
            print(f"Analyzing {max_samples} sampled texts for word frequency")
        else:
            text_sample = self.text_series
        
        # Vectorized text cleaning and tokenization
        all_text = ' '.join(text_sample.astype(str))
        cleaned = self.clean_text(all_text)
        
        # Tokenize all at once (much faster than per-text)
        tokens = word_tokenize(cleaned)
        
        # Filter stopwords and short words
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        # Get frequency counts
        word_freq = pd.Series(tokens).value_counts().head(n_words)
        
        return pd.DataFrame({
            'word': word_freq.index,
            'frequency': word_freq.values
        })
    
    def get_topic_modeling(
        self, 
        num_topics: int = 5, 
        n_words: int = 10, 
        max_samples: int = 3000
    ) -> Dict[int, List[str]]:
        """Perform topic modeling using Latent Dirichlet Allocation (LDA).
        
        Discovers hidden topics in the text corpus by identifying groups of
        words that frequently occur together.
        
        Args:
            num_topics: Number of topics to extract (default: 5)
            n_words: Number of top words per topic (default: 10)
            max_samples: Maximum texts to process (default: 3000)
        
        Returns:
            Dictionary mapping topic ID (int) to list of top words (List[str])
        
        Example:
            >>> topics = analyzer.get_topic_modeling(num_topics=3, n_words=10)
            >>> 
            >>> # Display topics
            >>> for topic_id, words in topics.items():
            >>>     st.write(f"**Topic {topic_id}:** {', '.join(words)}")
            >>> 
            >>> # Interpret results
            >>> # Topic 0: ['price', 'value', 'cost', ...] → Price-related
            >>> # Topic 1: ['shipping', 'delivery', 'fast', ...] → Delivery-related
            >>> # Topic 2: ['quality', 'product', 'good', ...] → Quality-related
        
        Note:
            - Uses scikit-learn's LDA implementation
            - Requires at least 10-20 documents for meaningful results
            - More topics = more granular but potentially less coherent
            - Uses all CPU cores (n_jobs=-1) for faster computation
            - Sampling recommended for datasets >3000 texts
        """
        # Lazy load sklearn modules
        feature_extraction_text = LazyModuleLoader.load_module('sklearn.feature_extraction.text')
        decomposition = LazyModuleLoader.load_module('sklearn.decomposition')
        
        CountVectorizer = getattr(feature_extraction_text, 'CountVectorizer')
        LatentDirichletAllocation = getattr(decomposition, 'LatentDirichletAllocation')
        
        # Sample if dataset is large
        if len(self.text_series) > max_samples:
            text_sample = self.text_series.sample(n=max_samples, random_state=42)
            print(f"Using {max_samples} sampled texts for topic modeling")
        else:
            text_sample = self.text_series
        
        cleaned_texts = [self.clean_text(text) for text in text_sample]
        
        # Create document-term matrix
        vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english',
            min_df=2  # Word must appear in at least 2 documents
        )
        
        doc_term_matrix = vectorizer.fit_transform(cleaned_texts)
        
        # Fit LDA model
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        lda.fit(doc_term_matrix)
        
        # Extract top words per topic
        feature_names = vectorizer.get_feature_names_out()
        topics = {}
        
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topics[topic_idx] = top_words
        
        return topics
    
    def create_wordcloud(self, max_words: int = 100) -> Any:
        """Generate word cloud image for visualization.
        
        Creates a visual representation where word size represents frequency.
        
        Args:
            max_words: Maximum number of words to include (default: 100)
        
        Returns:
            WordCloud object (can be displayed with matplotlib or converted to image)
        
        Example:
            >>> import matplotlib.pyplot as plt
            >>> 
            >>> wordcloud = analyzer.create_wordcloud(max_words=50)
            >>> 
            >>> # Display with matplotlib
            >>> plt.figure(figsize=(10, 5))
            >>> plt.imshow(wordcloud, interpolation='bilinear')
            >>> plt.axis('off')
            >>> st.pyplot(plt)
            >>> 
            >>> # Or save as image
            >>> wordcloud.to_file('wordcloud.png')
        
        Note:
            - Automatically removes stopwords
            - Uses 'viridis' colormap for better visibility
            - Size 800x400 pixels
            - White background
        """
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
        """Create interactive bar chart of sentiment distribution.
        
        Visualizes the breakdown of positive, negative, and neutral sentiments.
        
        Args:
            sentiment_df: DataFrame from get_sentiment_analysis()
        
        Returns:
            Plotly Figure object ready for display
        
        Example:
            >>> sentiments = analyzer.get_sentiment_analysis()
            >>> fig = analyzer.create_sentiment_plot(sentiments)
            >>> st.plotly_chart(fig, use_container_width=True)
        
        Note:
            - Positive: Green
            - Negative: Red
            - Neutral: Gray
        """
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
        """Create horizontal bar chart of word frequencies.
        
        Visualizes the most common words in a clear, sorted format.
        
        Args:
            word_freq_df: DataFrame from get_word_frequency()
        
        Returns:
            Plotly Figure object with horizontal bars
        
        Example:
            >>> word_freq = analyzer.get_word_frequency(n_words=30)
            >>> fig = analyzer.create_word_frequency_plot(word_freq)
            >>> st.plotly_chart(fig, use_container_width=True)
        
        Note:
            - Horizontal layout for better readability
            - Sorted by frequency (most common at top)
            - 600px height accommodates many words
        """
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
