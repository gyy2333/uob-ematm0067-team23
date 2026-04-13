
***

# Visualization Part
Person in charge: Chenyang Yan
---

## 1. Work Contents

The visualization results are based on the work of other members within the team, with the analysis divided into two axes: **text representation** and **comparison method**.

Work completed so far:
1. Predefined the data reception format and strictly standardized the intra-group data exchange format to avoid additional workload.

2. Based on the poor performance of existing results, a dedicated stopword list was constructed to improve the effectiveness of visualization.

---
***
## 2.  Visualizations & Data Dependencies

**For the results of Text Representation**

| Image Name | Core Input Data | Image Purpose |
| :--- | :--- | :--- |
| **01_category_growth.png** | `processed_dataset_final.csv` | Visualizes publication volume trends across AI subfields over time, illustrating macro-level disciplinary expansion and heat. |
| **02_keywords_period_wordcloud.png** | `keywords_by_period.csv` | Highlights the most prominent core academic terms while filtering out noise, revealing the core focus of the dataset. |
| **03_embeddings_clustering.png** | `embeddings.npy`<br>`processed_dataset_final.csv` | Uncovers semantic clustering and the latent feature space distribution of papers across different historical periods. |
| **04_tfidf_trajectory.png** | `tfidf_matrix.npz`<br>`tfidf_vocab.pkl`<br>`processed_dataset_final.csv` | Tracks the overall macroscopic paradigm shift of AI research across multiple eras utilizing SVD on TF-IDF matrices. |
| **05_similarity_heatmap.png** | `similarity_matrix.csv` | Quantifies and displays the similarity matrix, showing the closeness and intersections among different research categories. |
| **06_ngram_density_complexity.png** | `ngram_matrix.npz`<br>`processed_dataset_final.csv` | Assesses academic expression complexity by evaluating the density of multi-word phrases over different periods. |
| **07_category_fingerprint.png** | `keywords_by_category.csv` | Compares exclusive technical foci across distinct CS fields, highlighting their commonalities and individual distinguishing features. |
| **08_keywords_period_evolution.png** | `keywords_by_period.csv` | Provides a timeline view of when specific technologies gained traction or phased out, representing knowledge transition. |
| **09_multi_phrases.png** | `processed_dataset_final.csv` | Provides a highly interpretable list of key N-grams (phrases) representing the exact breakthroughs of each time period. |

**For the results of Comparison Method**

Waiting for data to be received...
***

## 3. **How to Use the Visualization Code**
Due to the large size of the original project dataset, the processing results cannot be uploaded to GitHub for storage. For the visualization part, all received data has been uploaded and stored on Google Drive, and view access has been granted to all team members.

The link to the Drive folder is: https://drive.google.com/drive/folders/1sdKg4s9AUKgr7r1pWjj3E-3te2GETAsf?usp=drive_link

To run the visualization code locally, please download the relevant core dependency data to your local folder.
The current relative path for data access is：
```text
   Visualization/data/
```

## 4. **Environment Setup**

**Required Dependencies:**
* `os`, `pickle`, `typing`
* `numpy`, `pandas`, `scipy`
* `matplotlib`, `seaborn`, `wordcloud`
* `scikit-learn` (`TruncatedSVD`, `TSNE`, `CountVectorizer`)

**Installation:**
```bash
pip install numpy pandas matplotlib seaborn scipy wordcloud scikit-learn
```