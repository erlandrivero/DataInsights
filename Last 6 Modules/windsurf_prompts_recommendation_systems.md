# Windsurf Prompts: Recommendation Systems Module

## Overview

Add a comprehensive **Recommendation Systems** module to DataInsights with collaborative filtering, content-based recommendations, and hybrid approaches.

**Total Time:** ~5 hours  
**Prompts:** 6 detailed prompts  
**Result:** Production-ready recommendation engine with multiple algorithms

---

## PROMPT 1: Create Recommendation Engine Utility (1.5 hours)

### Instructions for Windsurf:

```
Create a new file `utils/recommendation_engine.py` for DataInsights with comprehensive recommendation system functionality.

Requirements:

1. **RecommendationEngine Class** with methods:
   - `prepare_data()` - Prepare user-item interaction matrix
   - `collaborative_filtering()` - User-based and item-based CF
   - `content_based_filtering()` - Recommendations based on item features
   - `hybrid_recommendations()` - Combine multiple approaches
   - `matrix_factorization()` - SVD-based recommendations
   - `evaluate_recommendations()` - Calculate precision, recall, NDCG

2. **Collaborative Filtering:**
   - User-based CF (find similar users)
   - Item-based CF (find similar items)
   - Cosine similarity calculation
   - Pearson correlation option
   - Handle sparse matrices efficiently

3. **Content-Based Filtering:**
   - TF-IDF for text features
   - Feature similarity calculation
   - Category/tag-based recommendations
   - Weighted feature importance

4. **Matrix Factorization:**
   - Use scikit-learn's TruncatedSVD
   - Latent factor discovery
   - Prediction generation
   - Handle cold start problem

5. **Evaluation Metrics:**
   - Precision@K
   - Recall@K
   - NDCG (Normalized Discounted Cumulative Gain)
   - Coverage
   - Diversity

6. **Features:**
   - Top-N recommendations
   - Similarity scores
   - Explanation generation (why recommended)
   - Confidence scores
   - Handle cold start (new users/items)

7. **Data validation:**
   - Check for user-item-rating structure
   - Handle missing values
   - Validate rating scale
   - Check for sufficient data

8. **Error handling:**
   - Try-except for all operations
   - Informative error messages
   - Fallback to simpler methods

9. **Code quality:**
   - Type hints
   - Comprehensive docstrings
   - Clean, efficient code
   - Follow DataInsights patterns

Example usage:
```python
engine = RecommendationEngine(df, user_col='user_id', item_col='product_id', rating_col='rating')
# Collaborative filtering
cf_recs = engine.collaborative_filtering(user_id=123, method='user-based', top_n=10)
# Content-based
cb_recs = engine.content_based_filtering(item_id=456, features=['category', 'brand'], top_n=10)
# Hybrid
hybrid_recs = engine.hybrid_recommendations(user_id=123, top_n=10, cf_weight=0.6, cb_weight=0.4)
```

Use pandas, numpy, scikit-learn (TfidfVectorizer, TruncatedSVD, cosine_similarity). Match style of existing utils files.

Test with sample movie ratings data: user_id, movie_id, rating, genre, title.
```

### Expected Output:
- `utils/recommendation_engine.py` file created
- RecommendationEngine class with 6+ methods
- Multiple recommendation algorithms
- Evaluation metrics
- Error handling

### Testing Checklist:
- [ ] File created in utils folder
- [ ] RecommendationEngine instantiates correctly
- [ ] Collaborative filtering works
- [ ] Content-based filtering works
- [ ] Hybrid recommendations work
- [ ] Evaluation metrics calculate correctly
- [ ] Handles sparse data

---

## PROMPT 2: Add Recommendation Systems Page (1 hour)

### Instructions for Windsurf:

```
Add a new "Recommendation Systems" page to the DataInsights Streamlit app (app.py).

Requirements:

1. **Add to navigation:**
   - Add "Recommendation Systems" option in sidebar
   - Place after "ML Regression"
   - Use 🎯 emoji icon

2. **Page structure:**
   - Title: "🎯 Recommendation Systems"
   - Subtitle: "Personalized recommendations using collaborative and content-based filtering"
   - Four main sections:
     a) Data Configuration
     b) Recommendation Method Selection
     c) Results & Recommendations
     d) Evaluation & Insights

3. **Data Configuration section:**
   - Column selection:
     * User ID column (auto-detect)
     * Item ID column (auto-detect)
     * Rating/Interaction column
     * Feature columns (optional, for content-based)
   
   - Data type selection:
     * Explicit feedback (ratings 1-5)
     * Implicit feedback (clicks, views, purchases)
   
   - Sample data options:
     * Movie ratings
     * E-commerce purchases
     * Article views

4. **Method Selection:**
   - Radio buttons for algorithm:
     * Collaborative Filtering (User-based)
     * Collaborative Filtering (Item-based)
     * Content-Based Filtering
     * Matrix Factorization (SVD)
     * Hybrid (Combine methods)
   
   - Parameters:
     * Number of recommendations (slider 1-20)
     * Similarity metric (cosine, pearson)
     * For hybrid: weight sliders (CF vs CB)
   
   - Input selection:
     * For user-based: Select user ID
     * For item-based: Select item ID
   
   - "Generate Recommendations" button

5. **Results section:**
   - Recommended items table:
     * Rank
     * Item ID/Name
     * Predicted rating/score
     * Similarity score
     * Explanation (why recommended)
   
   - Similar users/items (for CF):
     * Top 5 similar users/items
     * Similarity scores
   
   - Downloadable results (CSV)

6. **UI elements:**
   - Use st.columns() for layout
   - st.tabs() for different recommendation types
   - st.metric() for key statistics
   - st.dataframe() for results
   - st.expander() for help text

7. **Help text:**
   - Explanation of each algorithm
   - When to use which method
   - How to interpret results
   - Business applications

8. **Error handling:**
   - Check if data is uploaded
   - Validate column selections
   - Show helpful error messages
   - Handle insufficient data

Import RecommendationEngine from utils.recommendation_engine. Follow existing page patterns.
```

### Expected Output:
- New "Recommendation Systems" page
- Algorithm selection interface
- Parameter controls
- Results display
- Help documentation

### Testing Checklist:
- [ ] Page appears in navigation
- [ ] Can select user/item/rating columns
- [ ] Algorithm selection works
- [ ] Parameters adjust correctly
- [ ] Generate button triggers recommendations
- [ ] Results display properly
- [ ] Sample data loads

---

## PROMPT 3: Create Recommendation Visualizations (1 hour)

### Instructions for Windsurf:

```
Add comprehensive visualizations to the Recommendation Systems page in DataInsights.

Requirements:

1. **User-Item Interaction Heatmap:**
   - Show user-item rating matrix
   - X-axis: Items (top N most popular)
   - Y-axis: Users (sample of active users)
   - Color: Rating intensity
   - Highlight missing values (white/gray)
   - Interactive hover with details

2. **Similarity Network Graph:**
   - For collaborative filtering
   - Nodes: Users or items
   - Edges: Similarity connections
   - Edge thickness: Similarity strength
   - Highlight target user/item
   - Use Plotly network graph or networkx

3. **Recommendation Distribution:**
   - Bar chart of recommended items
   - X-axis: Item names/IDs
   - Y-axis: Predicted rating/score
   - Color-coded by confidence
   - Show top 10-20 recommendations

4. **Feature Importance (Content-Based):**
   - Bar chart showing which features drive recommendations
   - For content-based filtering
   - X-axis: Feature names
   - Y-axis: Importance score
   - Helps explain recommendations

5. **Coverage & Diversity Analysis:**
   - Pie chart: % of items recommended
   - Bar chart: Diversity score across categories
   - Shows if recommendations are diverse or repetitive

6. **Algorithm Comparison:**
   - Side-by-side comparison of different algorithms
   - Table showing:
     * Algorithm name
     * Top 5 recommendations
     * Precision@5
     * Diversity score
   - Helps choose best method

7. **Rating Distribution:**
   - Histogram of actual ratings
   - Histogram of predicted ratings
   - Compare distributions
   - Identify bias

8. **Layout:**
   - Use tabs:
     * Tab 1: Recommendations
     * Tab 2: Similarity Analysis
     * Tab 3: Performance Metrics
     * Tab 4: Algorithm Comparison

Use Plotly for visualizations. Add interpretation text below each chart.

Reference patterns from Market Basket Analysis network graphs and RFM Analysis visualizations.
```

### Expected Output:
- 6-7 interactive visualizations
- Network graphs for similarity
- Recommendation charts
- Performance metrics
- Tabbed layout

### Testing Checklist:
- [ ] Heatmap displays correctly
- [ ] Network graph shows connections
- [ ] Recommendation bars are clear
- [ ] Charts are interactive
- [ ] Tabs work properly
- [ ] Export functionality works

---

## PROMPT 4: Add Recommendation Evaluation (45 min)

### Instructions for Windsurf:

```
Add comprehensive evaluation metrics to the Recommendation Systems module.

Requirements:

1. **Evaluation Metrics to Calculate:**
   - Precision@K (K=5, 10, 20)
   - Recall@K
   - F1-Score@K
   - NDCG (Normalized Discounted Cumulative Gain)
   - Mean Average Precision (MAP)
   - Coverage (% of items recommended)
   - Diversity (variety in recommendations)
   - Novelty (how surprising recommendations are)

2. **Train/Test Split:**
   - Split data into train (80%) and test (20%)
   - Temporal split if timestamps available
   - Random split otherwise
   - Stratified by user if possible

3. **Evaluation Process:**
   - Train on training set
   - Generate recommendations for test users
   - Compare with actual test interactions
   - Calculate metrics

4. **Display Metrics:**
   - Metrics summary card:
     * Precision@10: {value}
     * Recall@10: {value}
     * NDCG: {value}
     * Coverage: {value}%
     * Diversity: {value}
   
   - Metrics comparison table:
     * Compare different algorithms
     * Highlight best performer
     * Show trade-offs

5. **Visualization:**
   - Precision-Recall curve
   - NDCG@K for different K values
   - Algorithm performance comparison (radar chart)

6. **Cross-Validation:**
   - Optional: K-fold cross-validation
   - Show average metrics across folds
   - Confidence intervals

7. **Business Metrics:**
   - Estimated click-through rate
   - Potential revenue impact
   - User engagement prediction

8. **UI Elements:**
   - "Evaluate Recommendations" button
   - Evaluation results in expander
   - Metrics visualization
   - Downloadable evaluation report

Add evaluation section to the Recommendation Systems page. Use scikit-learn metrics where applicable.

Reference ML Classification evaluation patterns for consistency.
```

### Expected Output:
- Multiple evaluation metrics
- Train/test split functionality
- Metrics visualization
- Performance comparison
- Evaluation report

### Testing Checklist:
- [ ] Train/test split works
- [ ] Metrics calculate correctly
- [ ] Evaluation runs without errors
- [ ] Results display clearly
- [ ] Comparison is meaningful
- [ ] Report downloads successfully

---

## PROMPT 5: Add AI-Powered Recommendation Insights (45 min)

### Instructions for Windsurf:

```
Add AI-powered insights to the Recommendation Systems page using OpenAI GPT-4.

Requirements:

1. **Generate recommendation insights:**
   - After recommendations are generated, create AI insights
   - Use existing ai_helper.py module
   - Pass recommendation results to GPT-4

2. **Insights to generate:**
   - Why these items were recommended
   - User preference patterns identified
   - Recommendation quality assessment
   - Suggestions to improve recommendations
   - Business opportunities

3. **Prompt template for GPT-4:**
```
You are a recommendation systems expert. Analyze the following recommendation results:

User/Item: {target}
Algorithm: {algorithm}
Top Recommendations: {recommendations}
Similarity Scores: {scores}
Evaluation Metrics: {metrics}

Please provide:
1. Explanation of why these items were recommended
2. Patterns in user preferences or item similarities
3. Quality assessment of recommendations
4. Suggestions to improve recommendation accuracy
5. Business opportunities based on these recommendations

Be concise, actionable, and business-focused.
```

4. **Display insights:**
   - Show in expandable section after recommendations
   - Categorize insights:
     * 🎯 Recommendation Explanation
     * 📊 Pattern Analysis
     * ✅ Quality Assessment
     * 💡 Improvement Suggestions
     * 💰 Business Opportunities

5. **Personalized explanations:**
   - For each recommended item, generate brief explanation
   - "Recommended because you liked..."
   - "Similar to items you've rated highly..."
   - "Popular among users like you..."

6. **Error handling:**
   - Handle API failures gracefully
   - Show fallback explanations
   - Cache insights

Use ai_helper.generate_insights(). Follow patterns from other AI-enhanced modules.

Add "Regenerate Insights" button.
```

### Expected Output:
- AI-generated insights
- Personalized explanations
- Categorized findings
- Actionable recommendations
- Professional formatting

### Testing Checklist:
- [ ] AI insights generate automatically
- [ ] Insights are relevant
- [ ] Explanations make sense
- [ ] Formatted clearly
- [ ] Regenerate works
- [ ] Handles errors gracefully

---

## PROMPT 6: Add Export & Documentation (45 min)

### Instructions for Windsurf:

```
Add comprehensive export functionality and documentation to the Recommendation Systems module.

Requirements:

1. **Export options:**
   - Download recommendations (CSV)
   - Download similarity matrix (CSV)
   - Download evaluation metrics (CSV)
   - Download full recommendation report (Markdown/PDF)
   - Download visualizations (PNG bundle)

2. **Recommendation report structure:**
   ```markdown
   # Recommendation Systems Report
   Generated: {date}
   
   ## Executive Summary
   - Algorithm used: {algorithm}
   - Target: {user/item}
   - Recommendations generated: {n}
   - Average confidence: {score}
   
   ## Top Recommendations
   {recommendations_table}
   
   ## Similarity Analysis
   {similarity_info}
   
   ## Evaluation Metrics
   {metrics_table}
   
   ## AI Insights
   {ai_insights}
   
   ## Methodology
   - Algorithm: {algorithm_description}
   - Parameters: {parameters}
   - Data used: {data_summary}
   ```

3. **Export buttons:**
   - Place in sidebar under "Quick Export"
   - "Download Recommendations"
   - "Download Similarity Matrix"
   - "Download Evaluation Report"
   - "Download Full Report"

4. **Documentation:**
   - Add "📚 Recommendation Systems Guide" expander
   - Explain different algorithms:
     * Collaborative Filtering
     * Content-Based Filtering
     * Matrix Factorization
     * Hybrid Methods
   - When to use each
   - Business applications:
     * E-commerce: Product recommendations
     * Streaming: Content suggestions
     * News: Article recommendations
     * Social: Friend suggestions
   - Best practices
   - Common pitfalls

5. **Help tooltips:**
   - Add (?) icons for:
     * Collaborative filtering
     * Content-based filtering
     * Similarity metrics
     * Evaluation metrics
   - Use st.help() or tooltips

6. **Sample datasets:**
   - Movie ratings (MovieLens-style)
   - E-commerce purchases
   - Article views
   - Include README

7. **Integration:**
   - Add to main Reports page
   - Include recommendation analysis in comprehensive reports

Use export_helper.py and advanced_report_exporter.py. Follow patterns from other modules.
```

### Expected Output:
- Multiple export formats
- Comprehensive report
- Documentation
- Sample datasets
- Integration with Reports

### Testing Checklist:
- [ ] All exports work
- [ ] Report includes all sections
- [ ] Documentation is clear
- [ ] Sample data loads
- [ ] Help tooltips work
- [ ] Reports page integration works

---

## 🎯 Implementation Order

1. **PROMPT 1** - Create recommendation engine (foundation)
2. **PROMPT 2** - Add page and UI (interface)
3. **PROMPT 3** - Add visualizations (insights)
4. **PROMPT 4** - Add evaluation (quality assurance)
5. **PROMPT 5** - Add AI insights (intelligence)
6. **PROMPT 6** - Add exports & docs (polish)

---

## 📊 Expected Final Result

After completing all 6 prompts:

✅ **Multiple recommendation algorithms**
✅ **Collaborative & content-based filtering**
✅ **Comprehensive evaluation metrics**
✅ **Interactive visualizations**
✅ **AI-powered insights**
✅ **Professional exports**
✅ **Complete documentation**

**Total time:** ~5 hours  
**Value:** Very High - major differentiator

---

## 🧪 Final Testing Checklist

- [ ] Upload sample movie ratings data
- [ ] Select user_id, movie_id, rating columns
- [ ] Try collaborative filtering (user-based)
- [ ] Try collaborative filtering (item-based)
- [ ] Try content-based filtering
- [ ] Try hybrid recommendations
- [ ] View similarity network graph
- [ ] Check evaluation metrics
- [ ] Read AI insights
- [ ] Download recommendation report
- [ ] Verify recommendations make sense

---

## 💡 Pro Tips

1. **Test with MovieLens data** - Standard benchmark dataset
2. **Verify cold start handling** - New users/items
3. **Check diversity** - Avoid filter bubbles
4. **Review explanations** - Must be understandable
5. **Test performance** - Should be fast even with large data

---

**Ready to implement!** Copy prompts 1-6 into Windsurf in order. 🚀

