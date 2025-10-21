---

## 💬 Text Mining & Sentiment Analysis Module - Windsurf Prompts

**Total Time:** 4-5 hours

**Features:**
- Text data upload and cleaning
- Sentiment analysis (Positive, Negative, Neutral)
- Word frequency and word cloud visualization
- Topic modeling (LDA) to discover themes
- Named Entity Recognition (NER)
- AI-powered summarization and insights

---

### **PROMPT 1: Setup & Text Analysis Utility** (1 hour)

**Goal:** Install necessary NLP libraries and create a backend utility for text processing.

**Windsurf Prompt:**

```
Upgrade the DataInsights app. First, add `nltk`, `textblob`, and `spacy` to `requirements.txt`. Then, create a new utility file at `utils/text_mining.py`.

In this file, create a class called `TextAnalyzer`. The `__init__` method should take a pandas Series (a text column) as input. Before initializing, the class should ensure the necessary NLTK and Spacy models are downloaded (`punkt`, `stopwords`, `wordnet`, `vader_lexicon`, and `en_core_web_sm`).

Implement the following methods:

1.  `get_sentiment_analysis(self)`: Uses TextBlob or NLTK's VADER to calculate polarity and subjectivity for each text entry. It should return a dataframe with these scores and a classification (Positive, Negative, Neutral).
2.  `get_word_frequency(self, n_words)`: Cleans the text (lowercase, remove stopwords, punctuation), and returns a dataframe of the top `n_words` and their frequencies.
3.  `get_named_entities(self)`: Uses SpaCy to perform Named Entity Recognition (NER) and returns a dataframe of entities, their labels (e.g., PERSON, ORG, GPE), and their frequencies.

Ensure all methods have robust error handling and clear docstrings.
```

**Testing Checklist:**
- [ ] Verify `requirements.txt` is updated.
- [ ] Check that `utils/text_mining.py` is created.
- [ ] Ensure the NLTK/Spacy model download process is handled correctly.
- [ ] Test `get_sentiment_analysis` and check the output dataframe.
- [ ] Test `get_word_frequency` with different `n_words` values.
- [ ] Test `get_named_entities` and verify the entity extraction.

---

### **PROMPT 2: Add Text Mining Page to App** (45 min)

**Goal:** Create the main UI for the Text Mining module in the Streamlit app.

**Windsurf Prompt:**

```
Upgrade the DataInsights app. In `app.py`, add a new page to the main navigation called "Text Mining & NLP".

On this new page, create the following UI:

1.  **Title:** "Text Mining & Sentiment Analysis"
2.  **Data Upload:** Use the existing data uploader component.
3.  **Column Selection:** Add a selectbox for the user to choose the text column they want to analyze from the uploaded dataframe.
4.  **Analysis Tabs:** Once a column is selected, display the results in a tabbed interface:
    - **Sentiment Analysis**
    - **Word Frequency**
    - **Named Entities**

Instantiate the `TextAnalyzer` class from `utils/text_mining.py` once the user selects a column. Use loading spinners for analysis processes. The initial view should be clean and guide the user to select a column.
```

**Testing Checklist:**
- [ ] Verify "Text Mining & NLP" appears in the navigation.
- [ ] Test data upload and text column selection.
- [ ] Check that the tabbed interface appears after column selection.
- [ ] Ensure the app structure is clean and ready for results to be displayed.

---

### **PROMPT 3: Display Sentiment & Word Frequency Results** (45 min)

**Goal:** Visualize the sentiment and word frequency analysis results.

**Windsurf Prompt:**

```
Upgrade the "Text Mining & NLP" page in `app.py`.

**In the "Sentiment Analysis" tab:**
1.  Display summary metrics: Overall Sentiment (the mode of the classifications), percentage of Positive, Negative, and Neutral texts.
2.  Show a pie chart or bar chart visualizing the distribution of sentiments.
3.  Display the dataframe with the text and its calculated sentiment scores.

**In the "Word Frequency" tab:**
1.  Add a slider to control the number of top words to display (e.g., 10 to 100).
2.  Display a bar chart of the top N most frequent words.
3.  Generate and display a Word Cloud visualization based on the word frequencies. Use the `wordcloud` library (add it to `requirements.txt`).
```

**Testing Checklist:**
- [ ] Verify the sentiment summary metrics are correct.
- [ ] Check that the sentiment distribution chart is clear.
- [ ] Test the word frequency slider and ensure the bar chart updates.
- [ ] Ensure the Word Cloud is generated and displayed correctly.
- [ ] Test with different text datasets.

---

### **PROMPT 4: Display NER & Implement Topic Modeling** (1 hour)

**Goal:** Show Named Entities and add a new Topic Modeling feature.

**Windsurf Prompt:**

```
Upgrade the `utils/text_mining.py` file. Add a new method to the `TextAnalyzer` class called `get_topic_modeling(self, num_topics)`. This method will:
1.  Use `sklearn.feature_extraction.text.CountVectorizer` and `sklearn.decomposition.LatentDirichletAllocation`.
2.  Perform LDA to extract the specified number of topics.
3.  Return the top words for each topic.

Now, upgrade the "Text Mining & NLP" page in `app.py`:

**In the "Named Entities" tab:**
1.  Display a table of the most common entities and their labels (PERSON, ORG, etc.).
2.  Add a filter to view entities by label.

**Add a new tab: "Topic Modeling"**
1.  Add a slider to select the number of topics to discover (e.g., 2 to 10).
2.  Display the top N words for each discovered topic in a clean format (e.g., using `st.expander` for each topic).
```

**Testing Checklist:**
- [ ] Verify the Named Entities table and filter work correctly.
- [ ] Check that the new "Topic Modeling" tab is present.
- [ ] Test the topic number slider and ensure the LDA model runs.
- [ ] Review the topic words to ensure they are relevant and make sense.

---

### **PROMPT 5: AI-Powered Summarization & Insights** (45 min)

**Goal:** Use AI to provide a high-level summary and actionable insights from the text data.

**Windsurf Prompt:**

```
Upgrade the "Text Mining & NLP" page in `app.py`. Add a new main section at the top of the results area called "AI-Powered Summary".

Add a button labeled "Generate AI Summary & Insights". When clicked, this button will:
1.  Gather the key findings: overall sentiment, top 5 words, top 3 named entities, and top 2 topics.
2.  Send this context to the OpenAI API.
3.  Ask the AI to act as a business analyst and provide:
    - A concise executive summary of the text data.
    - Key positive and negative themes.
    - Actionable business recommendations based on the findings (e.g., "The frequent mention of 'slow support' and negative sentiment suggests a need to improve customer service response times.").

Display the AI's response in a well-formatted `st.info` or `st.container` with a header.
```

**Testing Checklist:**
- [ ] Verify the "Generate AI Summary" button works.
- [ ] Check the quality of the context sent to the AI.
- [ ] Review the AI-generated summary for accuracy and relevance.
- [ ] Assess the business value of the AI-generated recommendations.

---

### **PROMPT 6: Final Polish & Export** (30 min)

**Goal:** Add help text, error handling, and export functionality.

**Windsurf Prompt:**

```
Finalize the "Text Mining & NLP" module in `app.py`.

1.  **Add Help Text:** Create a help section with expanders explaining Sentiment Analysis, Word Frequency, NER, and Topic Modeling.
2.  **Error Handling:** Ensure user-friendly errors appear for common issues, such as selecting a non-text column or providing a dataset with too little text for analysis.
3.  **Export Features:** Add buttons to download:
    - A CSV file with the original text and all its analysis (sentiment, entities).
    - A Markdown file containing the complete analysis report, including the AI summary and all charts/tables.
    - The Word Cloud as a PNG image.

Finally, create a new documentation file `guides/TEXT_MINING_GUIDE.md` explaining how to use the module and interpret the results. Update the main README to link to this new guide.
```

**Testing Checklist:**
- [ ] Check that all help text is clear and useful.
- [ ] Test error handling with invalid data.
- [ ] Verify all download buttons work correctly.
- [ ] Check the content and formatting of the downloaded files.
- [ ] Ensure `TEXT_MINING_GUIDE.md` is created and linked.
- [ ] Perform a full end-to-end test of the module.

---

