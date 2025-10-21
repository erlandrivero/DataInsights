# Market Basket Analysis Module - User Guide

## Overview

The Market Basket Analysis (MBA) module in DataInsight AI allows you to discover hidden patterns in transactional data using the Apriori algorithm.

## Getting Started

### 1. Load Data

**Option A: Sample Dataset**
- Click "Load Groceries Data" to use the sample grocery transactions dataset
- Contains 9,835 transactions with 169 unique items

**Option B: Upload Custom Data**
- Prepare CSV file with columns: `transaction_id`, `item`
- Each row = one item in a transaction
- Example:
  ```csv
  transaction_id,item
  1,bread
  1,milk
  2,eggs
  2,bread
  ```

### 2. Adjust Thresholds

- **Minimum Support** (0.001-0.1): How often itemsets appear
  - Lower = more rules, but less frequent
  - Higher = fewer rules, but more common
  - Recommended: Start with 0.01 (1%)

- **Minimum Confidence** (0.1-1.0): Strength of association
  - Lower = more rules, weaker associations
  - Higher = fewer rules, stronger associations
  - Recommended: Start with 0.2 (20%)

- **Minimum Lift** (1.0-5.0): Correlation strength
  - 1.0 = no correlation
  - >1.0 = positive correlation (items bought together)
  - Recommended: Start with 1.0

### 3. Run Analysis

- Click "Run Market Basket Analysis"
- Wait for processing (usually 5-30 seconds)
- View results in tables and visualizations

### 4. Explore Results

**Association Rules Table:**
- Sort by Lift, Confidence, or Support
- Download as CSV
- Search for specific items

**Visualizations:**
- **Scatter Plot:** See rule strength visually
- **Network Graph:** Understand item relationships
- **Top Items:** Identify most frequent products

**Business Insights:**
- Read AI-generated recommendations
- Review strategic suggestions
- Download full report

## Understanding the Metrics

### Support
**What it means:** How often an itemset appears

**Example:** Support({milk}) = 0.25 means milk appears in 25% of transactions

**Business use:** Identify popular products

### Confidence
**What it means:** Probability of buying B given A was purchased

**Example:** Confidence({milk} → {bread}) = 0.6 means 60% of milk buyers also buy bread

**Business use:** Predict customer behavior

### Lift
**What it means:** How much more likely items are bought together vs. independently

**Example:** Lift({milk} → {bread}) = 2.0 means buying milk makes you 2x more likely to buy bread

**Business use:** Find strong associations

## Business Applications

### Retail
- **Product Placement:** Put associated items near each other
- **Bundling:** Create combo deals from high-confidence rules
- **Promotions:** "Buy X, get Y at discount"

### E-commerce
- **Recommendations:** "Customers who bought X also bought Y"
- **Upselling:** Suggest complementary products
- **Personalization:** Tailor homepage to purchase history

### Inventory
- **Stock Planning:** Order associated items proportionally
- **Demand Forecasting:** Predict consequent sales from antecedent sales
- **Warehouse Layout:** Store related items together

## Tips for Best Results

1. **Start Conservative:** Begin with default thresholds, then adjust
2. **Focus on Lift:** Rules with lift > 2 are usually most actionable
3. **Consider Context:** Not all high-lift rules make business sense
4. **Test Strategies:** Implement recommendations and measure results
5. **Iterate:** Rerun analysis as customer behavior changes

## Troubleshooting

**No rules found:**
- Lower minimum support (try 0.005)
- Lower minimum confidence (try 0.15)
- Check data format is correct

**Too many rules:**
- Raise minimum support (try 0.02)
- Raise minimum confidence (try 0.3)
- Raise minimum lift (try 1.5)

**Analysis is slow:**
- Reduce dataset size
- Increase minimum support
- Use fewer transactions

## Example Workflow

1. Load groceries dataset
2. Set thresholds: support=0.01, confidence=0.2, lift=1.0
3. Run analysis
4. Sort rules by Lift (descending)
5. Review top 10 rules
6. Check network graph for clusters
7. Read business insights
8. Download report
9. Implement recommendations
10. Track results

## Support

For questions or issues:
- Check this guide
- Review the "What is MBA?" section in the app
- Consult the main README.md

---

**Made with ❤️ for Data Mining Module 1 Assignment**
