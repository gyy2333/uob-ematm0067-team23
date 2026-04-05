"""
Data visualization script for AI/ML/NLP paper abstracts.
"""

import os
from collections import Counter
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
)

# Constants
ACADEMIC_STOPWORDS = [
    'based', 'mechanism', 'mechanisms', 'analysis', 'application', 'research', 
    'review', 'trend', 'features', 'paper', 'propose', 'proposed', 'method', 
    'methods', 'model', 'models', 'results', 'result', 'using', 'approach', 
    'performance', 'data', 'task', 'tasks', 'state', 'art', 'new', 'show', 
    'two', 'problem', 'problems', 'time', 'algorithm', 'algorithms', 'word', 
    'words', 'language', 'languages', 'network', 'networks', 'system', 'systems', 
    'information', 'different', 'well', 'used', 'via', 'work', 'works', 'use', 
    'make', 'study', 'studies', 'set', 'architectures', 'domains', 'representations', 
    'end', 'english', 'number', 'numbers', 'case', 'cases', 'example', 'examples', 
    'type', 'types', 'value', 'values', 'level', 'levels', 'step', 'steps', 
    'function', 'functions', 'process', 'processes', 'term', 'terms', 'dataset', 
    'datasets', 'text', 'texts', 'accuracy', 'parameters', 'parameter', 'experiment', 
    'experiments', 'figure', 'table', 'section', 'article', 'journal', 'conference', 
    'recent', 'current', 'future', 'pre', 'real', 'document'
]

ULTIMATE_STOPWORDS = list(ENGLISH_STOP_WORDS.union(ACADEMIC_STOPWORDS))
N_TOPICS = 5
THEME_COLORS = ['#00798C', '#D1495B', '#30638E', '#EDAE49', '#66A182']


def main() -> None:
    """Main execution block"""
    # Load & Clean Data
    file_path = os.path.join('data', 'processed', 'ai_ml_nlp_clean.csv')
    df = (pd.read_csv(file_path)
          .dropna(subset=['year', 'clean_abstract'])
          .assign(
              year=lambda x: x['year'].astype(int), 
              clean_abstract=lambda x: x['clean_abstract'].astype(str)
          ))
    
    out_dir = os.path.join('data', 'visualization', 'svg')
    os.makedirs(out_dir, exist_ok=True)

    # LDA Topic Evolution
    print("\n--- Generating LDA Topic Evolution ---")
    vectorizer = CountVectorizer(
        stop_words=ULTIMATE_STOPWORDS, max_df=0.8, 
        min_df=5, max_features=1000, ngram_range=(1, 2)
    )
    x_counts = vectorizer.fit_transform(df['clean_abstract'])
    
    feature_names = vectorizer.get_feature_names_out().tolist()

    lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=42)
    topic_cols = [f'Topic {i+1}' for i in range(N_TOPICS)]
    df_topics = pd.DataFrame(
        lda.fit_transform(x_counts), index=df.index, columns=topic_cols
    )
    topic_evo = pd.concat([df[['year']], df_topics], axis=1).groupby('year').mean()

    # Extract keywords & identify overlaps
    topic_words = [
        [str(feature_names[i]) for i in topic.argsort()[:-11:-1]] 
        for topic in lda.components_
    ]
    
    all_words_flat = (w for t in topic_words for w in t)
    repeated = {w for w, c in Counter(all_words_flat).items() if c > 1}

    content_lines = []
    target_pca_words = []
    for i, words in enumerate(topic_words):
        fmt = [f"★ {w.upper()}" if w in repeated else w for w in words]
        content_lines.append(f"Topic {i+1}: {', '.join(fmt)}")
        for w in words:
            if len(w.split()) == 1 and w not in target_pca_words:
                target_pca_words.append(w)
                break

    target_pca_words = target_pca_words[:N_TOPICS]

    # Milestones dict comprehension
    milestones = {
        year: f"{str(name).replace('_', ' ').replace('plus', 'Plus').title()} Era"
        for name, year in df.groupby('period')['year'].min().items()
    }

    # Plot LDA
    fig, ax = plt.subplots(figsize=(14, 9))
    plt.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.28)
    
    ax.stackplot(
        topic_evo.index, topic_evo.T, labels=topic_evo.columns, 
        colors=THEME_COLORS, alpha=0.85
    )

    y_limit_upper = float(ax.get_ylim()[1])
    for yr, lbl in milestones.items():
        if yr in topic_evo.index and yr > df['year'].min():
            ax.axvline(x=yr, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.text(
                yr + 0.1, y_limit_upper * 0.95, lbl, va='top', ha='left', 
                fontsize=11, fontweight='bold', color='#333333',
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9)
            )

    ax.set_title(
        'LDA Topic Evolution', 
        pad=35, fontweight='bold', fontsize=14
    )
    ax.set(xlabel='Year', ylabel='Topic Intensity', xticks=sorted(df['year'].unique()))
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=N_TOPICS, frameon=False)
    
    fig.text(0.5, 0.12, "\n" * 7, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=1.5", fc="whitesmoke", ec="lightgray", alpha=0.8))
    fig.text(0.5, 0.19, "| Core Keywords per Topic (Top 10) |", ha='center', va='center', fontweight='bold')
    fig.text(0.5, 0.17, "★ UPPERCASE = Overlapping words across topics", ha='center', va='center', fontsize=10, alpha=0.7)
    fig.text(0.08, 0.11, "\n".join(content_lines), ha='left', va='center', fontsize=11)
    plt.show()

    # PCA Semantic Drift
    
    print(f"\n--- Generating PCA Drift for Targets: {target_pca_words} ---")
    tfidf = TfidfVectorizer(stop_words=ULTIMATE_STOPWORDS, max_features=2000, min_df=5)
    x_tfidf = tfidf.fit_transform(df['clean_abstract'])

    pca_data = []
    for word in target_pca_words:
        if word not in tfidf.vocabulary_:
            continue
            
        target_idx = int(tfidf.vocabulary_[word])
        doc_indices = x_tfidf.getcol(target_idx).nonzero()[0]

        for year in sorted(df['year'].unique()):
            year_mask = np.where(df['year'] == year)[0]
            valid_docs = np.intersect1d(year_mask, doc_indices)
            
            if len(valid_docs) > 0:
                selected_docs = cast(Any, x_tfidf)[valid_docs]
                flat_vec = np.asarray(selected_docs.mean(axis=0)).flatten()
                
                if flat_vec.sum() > 1e-4:
                    pca_data.append((word, year, flat_vec))

    if len(pca_data) > 2:
        pca = PCA(n_components=2)
        pca_input = np.array([d[2] for d in pca_data])
        pca.fit(pca_input)

        fig, ax = plt.subplots(figsize=(10, 7))

        for c_idx, word in enumerate(target_pca_words):
            word_data = sorted([d for d in pca_data if d[0] == word], key=lambda x: x[1])
            if len(word_data) > 2: 
                years = [d[1] for d in word_data]
                raw_vecs = np.array([d[2] for d in word_data])
                vecs = pca.transform(raw_vecs)
                
                # Smooth trajectories
                smoothed = np.copy(vecs)
                smoothed[1:-1] = (vecs[:-2] + vecs[1:-1] + vecs[2:]) / 3.0
                vecs = smoothed

                color = THEME_COLORS[c_idx]
                lbl = f"[Topic {c_idx+1}] {word} ({years[0]} -> {years[-1]})"

                ax.plot(vecs[:, 0], vecs[:, 1], marker='o', lw=3.0, ms=7, 
                        label=lbl, color=color, alpha=0.8)
                ax.plot(vecs[-1, 0], vecs[-1, 1], marker='*', ms=18, 
                        color=color, mec='black', mew=1.0)

                x_rng = float(vecs[:, 0].max() - vecs[:, 0].min())
                for i in range(len(years) - 1):
                    ax.arrow(
                        vecs[i, 0], vecs[i, 1], vecs[i+1, 0] - vecs[i, 0], 
                        vecs[i+1, 1] - vecs[i, 1], head_width=max(x_rng * 0.015, 0.001), 
                        length_includes_head=True, color=color, alpha=0.9
                    )

        ax.set(title='PCA Semantic Drift (Smoothed Trajectories)', 
               xlabel='PCA Component 1', ylabel='PCA Component 2')
        ax.plot([], [], marker='*', color='gray', ms=14, ls='None', label='★ Latest Year (End)')
        ax.legend(title='Core Concepts', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()