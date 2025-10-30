# Implementation Guide: Add CSV Exports to 6 New Modules

**Time Required:** 2-3 hours (30 minutes per module)  
**Difficulty:** Easy  
**Priority:** 🔥 CRITICAL  
**Impact:** HIGH - Essential functionality for users

---

## Overview

All 6 new modules currently only have Markdown exports, while older modules (RFM, MBA, Anomaly Detection) have both CSV and Markdown exports. This creates an inconsistency that limits user ability to export numerical results for further analysis.

### Modules to Update:
1. A/B Testing
2. Cohort Analysis
3. Recommendation Systems
4. Geospatial Analysis
5. Survival Analysis
6. Network Analysis

---

## Standard Pattern to Follow

### Reference Implementation (from RFM Analysis, lines 4380-4396)

```python
st.subheader("📥 Export Results")

col1, col2 = st.columns(2)

with col1:
    csv = rfm_segmented.to_csv(index=False)
    st.download_button(
        label="📥 Download RFM Data (CSV)",
        data=csv,
        file_name=f"rfm_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:
    st.download_button(
        label="📥 Download Report (Markdown)",
        data=report,
        file_name=f"rfm_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True
    )
```

### Key Elements:
1. **Section header:** `st.subheader("📥 Export Results")`
2. **Two-column layout:** `col1, col2 = st.columns(2)`
3. **CSV in left column** with clear label
4. **Markdown in right column** with clear label
5. **Timestamp in filename** for uniqueness
6. **Full-width buttons:** `use_container_width=True`

---

## Module 1: A/B Testing

### Current Location (Line 9539)
```python
st.download_button(
    label="📥 Download Test Report (Markdown)",
    data=report,
    file_name=f"ab_test_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
    mime="text/markdown",
    use_container_width=True
)
```

### Replace With:

```python
st.subheader("📥 Export Results")

col1, col2 = st.columns(2)

with col1:
    # Prepare test results CSV
    results_data = {
        'Metric': [
            'Control Sample Size',
            'Treatment Sample Size',
            'Control Rate/Mean',
            'Treatment Rate/Mean',
            'Absolute Lift',
            'Relative Lift (%)',
            'P-Value',
            'Statistically Significant',
            'Confidence Interval Lower',
            'Confidence Interval Upper',
            'Test Type',
            'Alpha Level',
            'Statistical Power'
        ],
        'Value': [
            st.session_state.ab_test_results.get('control_n', 'N/A'),
            st.session_state.ab_test_results.get('treatment_n', 'N/A'),
            f"{st.session_state.ab_test_results.get('control_rate', 0):.4f}",
            f"{st.session_state.ab_test_results.get('treatment_rate', 0):.4f}",
            f"{st.session_state.ab_test_results.get('absolute_lift', 0):.4f}",
            f"{st.session_state.ab_test_results.get('relative_lift', 0):.2f}",
            f"{st.session_state.ab_test_results.get('p_value', 0):.6f}",
            'Yes' if st.session_state.ab_test_results.get('is_significant', False) else 'No',
            f"{st.session_state.ab_test_results.get('ci_lower', 0):.4f}",
            f"{st.session_state.ab_test_results.get('ci_upper', 0):.4f}",
            st.session_state.ab_test_results.get('test_type', 'N/A'),
            f"{st.session_state.ab_test_results.get('alpha', 0.05):.2f}",
            f"{st.session_state.ab_test_results.get('power', 0.80):.2f}"
        ]
    }
    
    results_df = pd.DataFrame(results_data)
    csv = results_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name=f"ab_test_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:
    st.download_button(
        label="📥 Download Report (Markdown)",
        data=report,
        file_name=f"ab_test_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True
    )
```

### Optional: Add Comparison Data CSV

```python
# After the main export section, add a third button for detailed comparison
st.markdown("---")
st.markdown("**📊 Additional Exports:**")

if 'ab_comparison_data' in st.session_state:
    comparison_df = st.session_state.ab_comparison_data
    csv_comparison = comparison_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Detailed Comparison Data (CSV)",
        data=csv_comparison,
        file_name=f"ab_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
```

---

## Module 2: Cohort Analysis

### Current Location (Line 10043)
```python
st.download_button(
    label="📥 Download Cohort Report",
    data=report,
    file_name=f"cohort_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
    mime="text/markdown",
    use_container_width=True
)
```

### Replace With:

```python
st.subheader("📥 Export Results")

col1, col2, col3 = st.columns(3)

with col1:
    # Retention matrix CSV
    if 'cohort_retention' in st.session_state:
        retention_matrix = st.session_state.cohort_retention
        csv_retention = retention_matrix.to_csv()
        
        st.download_button(
            label="📥 Retention Matrix (CSV)",
            data=csv_retention,
            file_name=f"cohort_retention_matrix_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

with col2:
    # Cohort metrics CSV
    if 'cohort_metrics' in st.session_state:
        metrics_df = st.session_state.cohort_metrics
        csv_metrics = metrics_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Cohort Metrics (CSV)",
            data=csv_metrics,
            file_name=f"cohort_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

with col3:
    # Markdown report
    st.download_button(
        label="📥 Full Report (MD)",
        data=report,
        file_name=f"cohort_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True
    )
```

### Note: Ensure cohort_metrics is stored in session state

Add this after cohort analysis is complete:

```python
# After retention matrix is calculated
cohort_metrics = pd.DataFrame({
    'Cohort': retention_matrix.index,
    'Cohort Size': [len(df[df[cohort_col] == cohort]) for cohort in retention_matrix.index],
    'Avg Retention': retention_matrix.mean(axis=1),
    'Period 1 Retention': retention_matrix.iloc[:, 0] if len(retention_matrix.columns) > 0 else None,
    'Period 3 Retention': retention_matrix.iloc[:, 2] if len(retention_matrix.columns) > 2 else None,
    'Period 6 Retention': retention_matrix.iloc[:, 5] if len(retention_matrix.columns) > 5 else None,
    'Churn Rate': 1 - retention_matrix.mean(axis=1)
})

st.session_state.cohort_metrics = cohort_metrics
```

---

## Module 3: Recommendation Systems

### Current Location (Line 10558)
```python
st.download_button(
    label="📥 Download Recommendations Report",
    data=report,
    file_name=f"recommendations_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
    mime="text/markdown",
    use_container_width=True
)
```

### Replace With:

```python
st.subheader("📥 Export Results")

col1, col2, col3 = st.columns(3)

with col1:
    # Recommendations CSV
    if 'rec_recommendations' in st.session_state:
        recommendations_df = st.session_state.rec_recommendations
        csv_recs = recommendations_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Recommendations (CSV)",
            data=csv_recs,
            file_name=f"recommendations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

with col2:
    # Evaluation metrics CSV
    if 'rec_metrics' in st.session_state:
        metrics_df = st.session_state.rec_metrics
        csv_metrics = metrics_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Metrics (CSV)",
            data=csv_metrics,
            file_name=f"rec_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

with col3:
    # Markdown report
    st.download_button(
        label="📥 Full Report (MD)",
        data=report,
        file_name=f"recommendations_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True
    )
```

### Note: Ensure recommendations and metrics are stored

```python
# After generating recommendations
recommendations_list = []
for user_id in test_users:
    recs = engine.get_recommendations(user_id, top_n=10)
    for rank, (item_id, score) in enumerate(recs, 1):
        recommendations_list.append({
            'User ID': user_id,
            'Item ID': item_id,
            'Score': score,
            'Rank': rank
        })

st.session_state.rec_recommendations = pd.DataFrame(recommendations_list)

# Store evaluation metrics
metrics_data = {
    'Metric': ['Precision@5', 'Precision@10', 'Recall@5', 'Recall@10', 'NDCG@5', 'NDCG@10', 'Coverage'],
    'Value': [
        precision_at_5,
        precision_at_10,
        recall_at_5,
        recall_at_10,
        ndcg_at_5,
        ndcg_at_10,
        coverage
    ]
}

st.session_state.rec_metrics = pd.DataFrame(metrics_data)
```

---

## Module 4: Geospatial Analysis

### Current Location (Line 11092)
```python
st.download_button(
    label="📥 Download Geospatial Report",
    data=report,
    file_name=f"geospatial_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
    mime="text/markdown",
    use_container_width=True
)
```

### Replace With:

```python
st.subheader("📥 Export Results")

col1, col2, col3 = st.columns(3)

with col1:
    # Location data with clusters CSV
    if 'geo_data' in st.session_state:
        geo_df = st.session_state.geo_data.copy()
        
        # Add cluster assignments if available
        if 'geo_clusters' in st.session_state:
            geo_df['Cluster'] = st.session_state.geo_clusters
        
        csv_locations = geo_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Location Data (CSV)",
            data=csv_locations,
            file_name=f"geospatial_locations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

with col2:
    # Cluster statistics CSV
    if 'geo_cluster_stats' in st.session_state:
        cluster_stats = st.session_state.geo_cluster_stats
        csv_stats = cluster_stats.to_csv(index=False)
        
        st.download_button(
            label="📥 Cluster Stats (CSV)",
            data=csv_stats,
            file_name=f"geospatial_clusters_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

with col3:
    # Markdown report
    st.download_button(
        label="📥 Full Report (MD)",
        data=report,
        file_name=f"geospatial_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True
    )
```

### Note: Calculate cluster statistics

```python
# After clustering is performed
if 'geo_clusters' in st.session_state:
    cluster_stats_list = []
    
    for cluster_id in sorted(st.session_state.geo_clusters.unique()):
        cluster_data = geo_df[st.session_state.geo_clusters == cluster_id]
        
        cluster_stats_list.append({
            'Cluster ID': cluster_id,
            'Size': len(cluster_data),
            'Centroid Latitude': cluster_data[lat_col].mean(),
            'Centroid Longitude': cluster_data[lon_col].mean(),
            'Avg Value': cluster_data[value_col].mean() if value_col else None,
            'Min Latitude': cluster_data[lat_col].min(),
            'Max Latitude': cluster_data[lat_col].max(),
            'Min Longitude': cluster_data[lon_col].min(),
            'Max Longitude': cluster_data[lon_col].max()
        })
    
    st.session_state.geo_cluster_stats = pd.DataFrame(cluster_stats_list)
```

---

## Module 5: Survival Analysis

### Current Location (Line 11681)
```python
st.download_button(
    label="📥 Download Survival Report",
    data=report,
    file_name=f"survival_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
    mime="text/markdown",
    use_container_width=True
)
```

### Replace With:

```python
st.subheader("📥 Export Results")

col1, col2, col3 = st.columns(3)

with col1:
    # Survival probabilities CSV
    if 'surv_probabilities' in st.session_state:
        surv_probs = st.session_state.surv_probabilities
        csv_probs = surv_probs.to_csv(index=False)
        
        st.download_button(
            label="📥 Survival Curve (CSV)",
            data=csv_probs,
            file_name=f"survival_probabilities_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

with col2:
    # Individual risk scores CSV
    if 'surv_risk_scores' in st.session_state:
        risk_scores = st.session_state.surv_risk_scores
        csv_risks = risk_scores.to_csv(index=False)
        
        st.download_button(
            label="📥 Risk Scores (CSV)",
            data=csv_risks,
            file_name=f"survival_risk_scores_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

with col3:
    # Markdown report
    st.download_button(
        label="📥 Full Report (MD)",
        data=report,
        file_name=f"survival_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True
    )
```

### Note: Extract survival probabilities from Kaplan-Meier

```python
# After Kaplan-Meier fitting
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
kmf.fit(durations=df[duration_col], event_observed=df[event_col])

# Extract survival probabilities
surv_probs = pd.DataFrame({
    'Time': kmf.survival_function_.index,
    'Survival Probability': kmf.survival_function_.values.flatten(),
    'Confidence Interval Lower': kmf.confidence_interval_.iloc[:, 0].values,
    'Confidence Interval Upper': kmf.confidence_interval_.iloc[:, 1].values
})

st.session_state.surv_probabilities = surv_probs

# Calculate individual risk scores (if applicable)
# This would require Cox model or other risk stratification
# For now, can use simple percentile-based risk groups
risk_scores = df[[duration_col, event_col]].copy()
risk_scores['Risk Score'] = 1 - (df[duration_col] / df[duration_col].max())
risk_scores['Risk Group'] = pd.qcut(risk_scores['Risk Score'], q=3, labels=['Low', 'Medium', 'High'])

st.session_state.surv_risk_scores = risk_scores
```

---

## Module 6: Network Analysis

### Current Location (Line 12220)
```python
st.download_button(
    label="📥 Download Network Report",
    data=report,
    file_name=f"network_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
    mime="text/markdown",
    use_container_width=True
)
```

### Replace With:

```python
st.subheader("📥 Export Results")

col1, col2, col3 = st.columns(3)

with col1:
    # Network metrics CSV
    if 'net_metrics' in st.session_state:
        metrics_df = st.session_state.net_metrics
        csv_metrics = metrics_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Node Metrics (CSV)",
            data=csv_metrics,
            file_name=f"network_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

with col2:
    # Edge list CSV
    if 'net_edges' in st.session_state:
        edges_df = st.session_state.net_edges
        csv_edges = edges_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Edge List (CSV)",
            data=csv_edges,
            file_name=f"network_edges_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

with col3:
    # Markdown report
    st.download_button(
        label="📥 Full Report (MD)",
        data=report,
        file_name=f"network_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True
    )

# Optional: Community assignments
st.markdown("---")
st.markdown("**📊 Additional Exports:**")

if 'net_communities' in st.session_state:
    communities_df = st.session_state.net_communities
    csv_communities = communities_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Community Assignments (CSV)",
        data=csv_communities,
        file_name=f"network_communities_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
```

### Note: Calculate and store network metrics

```python
import networkx as nx

# After network is created
G = st.session_state.net_graph

# Calculate centrality metrics
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

# Create metrics DataFrame
metrics_list = []
for node in G.nodes():
    metrics_list.append({
        'Node': node,
        'Degree': G.degree(node),
        'Degree Centrality': degree_centrality[node],
        'Betweenness Centrality': betweenness_centrality[node],
        'Closeness Centrality': closeness_centrality[node],
        'Eigenvector Centrality': eigenvector_centrality[node]
    })

st.session_state.net_metrics = pd.DataFrame(metrics_list)

# Store edge list
edges_list = []
for edge in G.edges(data=True):
    edges_list.append({
        'Source': edge[0],
        'Target': edge[1],
        'Weight': edge[2].get('weight', 1)
    })

st.session_state.net_edges = pd.DataFrame(edges_list)

# Store community assignments (if community detection is performed)
if hasattr(st.session_state, 'net_community_dict'):
    communities_list = []
    for node, community in st.session_state.net_community_dict.items():
        communities_list.append({
            'Node': node,
            'Community': community
        })
    
    st.session_state.net_communities = pd.DataFrame(communities_list)
```

---

## Testing Checklist

After implementing CSV exports for each module, test the following:

### ✅ A/B Testing
- [ ] Run proportion test
- [ ] Verify results CSV downloads
- [ ] Verify markdown report downloads
- [ ] Check CSV contains all metrics
- [ ] Verify timestamp in filename

### ✅ Cohort Analysis
- [ ] Run cohort analysis
- [ ] Verify retention matrix CSV downloads
- [ ] Verify cohort metrics CSV downloads
- [ ] Verify markdown report downloads
- [ ] Check CSV format is correct

### ✅ Recommendation Systems
- [ ] Generate recommendations
- [ ] Verify recommendations CSV downloads
- [ ] Verify metrics CSV downloads
- [ ] Verify markdown report downloads
- [ ] Check CSV contains user, item, score, rank

### ✅ Geospatial Analysis
- [ ] Run geospatial analysis
- [ ] Verify location data CSV downloads
- [ ] Verify cluster stats CSV downloads
- [ ] Verify markdown report downloads
- [ ] Check CSV contains lat, lon, cluster

### ✅ Survival Analysis
- [ ] Run survival analysis
- [ ] Verify survival probabilities CSV downloads
- [ ] Verify risk scores CSV downloads
- [ ] Verify markdown report downloads
- [ ] Check CSV contains time, probability, CI

### ✅ Network Analysis
- [ ] Build network
- [ ] Verify node metrics CSV downloads
- [ ] Verify edge list CSV downloads
- [ ] Verify markdown report downloads
- [ ] Check CSV contains all centrality metrics

---

## Common Issues & Solutions

### Issue 1: KeyError when accessing session_state

**Problem:** `KeyError: 'module_results'`

**Solution:** Ensure results are stored in session_state before export section:
```python
# After analysis is complete
st.session_state.module_results = results_df
```

### Issue 2: CSV download button not appearing

**Problem:** Button doesn't show up

**Solution:** Check that data exists before creating button:
```python
if 'module_results' in st.session_state:
    csv = st.session_state.module_results.to_csv(index=False)
    st.download_button(...)
else:
    st.info("Run analysis first to enable exports")
```

### Issue 3: Timestamp format error

**Problem:** `AttributeError: 'Timestamp' object has no attribute 'strftime'`

**Solution:** Use `pd.Timestamp.now()` instead of `datetime.now()`:
```python
file_name=f"results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
```

### Issue 4: Column layout breaks on mobile

**Problem:** Three columns too narrow on mobile

**Solution:** Use responsive columns:
```python
# For desktop: 3 columns
# For mobile: stack vertically
col1, col2, col3 = st.columns([1, 1, 1])
```

---

## Deployment Checklist

Before deploying to Streamlit Cloud:

- [ ] Test all 6 modules locally
- [ ] Verify CSV downloads work
- [ ] Verify markdown downloads work
- [ ] Check file sizes are reasonable
- [ ] Test with sample data
- [ ] Verify timestamps in filenames
- [ ] Check button labels are consistent
- [ ] Test on mobile browser
- [ ] Commit changes to GitHub
- [ ] Deploy to Streamlit Cloud
- [ ] Test on live app

---

## Estimated Time Breakdown

| Module | Time | Notes |
|--------|------|-------|
| A/B Testing | 30 min | Simple results table |
| Cohort Analysis | 30 min | Need to add metrics calculation |
| Recommendations | 30 min | Need to store recommendations list |
| Geospatial | 30 min | Need to calculate cluster stats |
| Survival | 30 min | Need to extract KM probabilities |
| Network | 30 min | Need to calculate centrality metrics |
| Testing | 30 min | Test all exports |
| **TOTAL** | **3.5 hours** | Including testing |

---

## Success Criteria

✅ All 6 modules have CSV exports  
✅ All CSV exports follow standard pattern  
✅ All button labels are consistent  
✅ All exports include timestamps  
✅ All exports tested and working  
✅ Code is clean and maintainable  
✅ No errors in Streamlit Cloud  

---

## Next Steps After Implementation

1. **Update documentation** - Add export capabilities to README
2. **Add to changelog** - Document new export features
3. **User testing** - Get feedback on export functionality
4. **Consider additional exports** - JSON, Excel formats?
5. **Add export analytics** - Track which exports are most used

---

**Implementation Guide Complete**  
**Ready to implement!** 🚀

