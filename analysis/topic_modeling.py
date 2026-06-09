"""
Topic Modeling Module
======================
Uses Latent Dirichlet Allocation (LDA) to discover latent themes
in employee open-text survey responses.

Author: Anushree Ayyar
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter


# Stop words extended with survey-specific filler words
SURVEY_STOP_WORDS = [
    "i", "my", "me", "we", "our", "the", "a", "an", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "he", "she", "him", "her", "his", "hers", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "up", "out",
    "not", "no", "but", "or", "and", "if", "so", "than", "too", "very",
    "just", "don", "doesn", "didn", "won", "wouldn", "couldn", "shouldn",
    "also", "more", "much", "really", "like", "even", "still", "well",
    "think", "feel", "know", "get", "got", "make", "made", "go", "going",
    "thing", "things", "lot", "way", "want", "need", "see", "come",
    # survey-specific
    "company", "work", "job", "employee", "employees", "working", "overall",
    "generally", "sometimes", "often", "always", "never",
]


def preprocess_for_lda(texts):
    """Preprocess text for topic modeling."""
    processed = []
    for text in texts:
        if pd.isna(text) or text.strip() == "":
            continue
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text.split()) >= 5:  # Only include substantive responses
            processed.append(text)
    return processed


def run_topic_modeling(texts, n_topics=6, n_top_words=10):
    """Run LDA topic modeling on survey responses."""
    print(f"Running LDA topic modeling with {n_topics} topics...")

    processed = preprocess_for_lda(texts)
    print(f"  {len(processed)} substantive responses after filtering")

    # Create document-term matrix
    vectorizer = CountVectorizer(
        max_df=0.85,
        min_df=3,
        stop_words=SURVEY_STOP_WORDS,
        ngram_range=(1, 2),
        max_features=2000,
    )
    doc_term_matrix = vectorizer.fit_transform(processed)
    feature_names = vectorizer.get_feature_names_out()
    print(f"  Vocabulary size: {len(feature_names)} terms")

    # Fit LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=20,
        learning_method="online",
        random_state=42,
        doc_topic_prior=0.1,
        topic_word_prior=0.01,
    )
    doc_topics = lda.fit_transform(doc_term_matrix)

    # Extract topics
    topics = []
    topic_labels = {
        0: "Career Growth & Development",
        1: "Manager Relationship & Support",
        2: "Recognition & Appreciation",
        3: "Team Culture & Collaboration",
        4: "Work-Life Balance & Flexibility",
        5: "Tools, Resources & Infrastructure",
    }

    print(f"\n  Discovered Topics:")
    for topic_idx, topic in enumerate(lda.components_):
        top_word_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_word_indices]
        top_weights = [round(topic[i], 2) for i in top_word_indices]

        label = topic_labels.get(topic_idx, f"Topic {topic_idx + 1}")
        prevalence = (doc_topics[:, topic_idx] > 0.2).mean() * 100

        topics.append({
            "topic_id": topic_idx,
            "label": label,
            "top_words": top_words,
            "top_weights": top_weights,
            "prevalence_pct": round(prevalence, 1),
        })

        print(f"\n    Topic {topic_idx}: {label} ({prevalence:.0f}% of responses)")
        print(f"    Keywords: {', '.join(top_words[:7])}")

    # Assign dominant topic to each document
    dominant_topics = doc_topics.argmax(axis=1)
    topic_distribution = Counter(dominant_topics)

    # Get representative documents for each topic
    representative_docs = {}
    for topic_idx in range(n_topics):
        topic_doc_scores = doc_topics[:, topic_idx]
        top_doc_indices = topic_doc_scores.argsort()[-3:][::-1]
        representative_docs[topic_idx] = [processed[i][:200] for i in top_doc_indices]

    # TF-IDF for additional keyword extraction
    tfidf = TfidfVectorizer(
        max_df=0.8, min_df=3,
        stop_words=SURVEY_STOP_WORDS,
        ngram_range=(1, 2),
        max_features=500,
    )
    tfidf_matrix = tfidf.fit_transform(processed)
    tfidf_features = tfidf.get_feature_names_out()

    # Top TF-IDF terms overall
    mean_tfidf = tfidf_matrix.mean(axis=0).A1
    top_tfidf_indices = mean_tfidf.argsort()[-20:][::-1]
    top_tfidf_terms = [(tfidf_features[i], round(mean_tfidf[i], 4)) for i in top_tfidf_indices]
    print(f"\n  Top TF-IDF terms: {[t[0] for t in top_tfidf_terms[:10]]}")

    # Save results
    results = {
        "topics": topics,
        "representative_docs": representative_docs,
        "top_tfidf_terms": top_tfidf_terms,
        "doc_topic_distribution": doc_topics,
        "processed_texts": processed,
    }

    # Save topic assignments
    topic_df = pd.DataFrame({
        "text": processed,
        "dominant_topic": dominant_topics,
        "topic_label": [topic_labels.get(t, f"Topic {t}") for t in dominant_topics],
        "topic_confidence": [round(doc_topics[i, dominant_topics[i]], 3) for i in range(len(processed))],
    })
    topic_df.to_csv("data/topic_assignments.csv", index=False)

    # Save topic summary
    topic_summary = pd.DataFrame(topics)
    topic_summary["top_words"] = topic_summary["top_words"].apply(lambda x: ", ".join(x[:7]))
    topic_summary.to_csv("data/topic_summary.csv", index=False)

    print(f"\n  Topic assignments and summaries saved to data/")
    return results


if __name__ == "__main__":
    pulse = pd.read_csv("data/pulse_survey_enriched.csv")
    exit_df = pd.read_csv("data/exit_surveys_enriched.csv")

    # Combine all open text
    all_texts = list(pulse["open_text_response"].dropna()) + list(exit_df["exit_comment"].dropna())
    results = run_topic_modeling(all_texts)
