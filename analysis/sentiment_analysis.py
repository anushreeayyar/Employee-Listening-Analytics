"""
NLP Sentiment Analysis Module
==============================
Analyzes open-text survey responses using VADER sentiment analysis.
Classifies responses into sentiment categories and extracts key phrases.

Author: Anushree Ayyar
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def clean_text(text):
    """Clean and preprocess survey text responses."""
    if pd.isna(text) or text.strip() == "":
        return ""
    text = re.sub(r"[^\w\s.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def analyze_sentiment(text, analyzer):
    """Run VADER sentiment analysis on a text response."""
    if not text:
        return {"compound": 0, "pos": 0, "neu": 0, "neg": 0, "category": "no_response"}

    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.35:
        category = "very_positive"
    elif compound >= 0.05:
        category = "positive"
    elif compound >= -0.05:
        category = "neutral"
    elif compound >= -0.35:
        category = "negative"
    else:
        category = "very_negative"

    return {
        "compound": round(compound, 4),
        "pos": round(scores["pos"], 4),
        "neu": round(scores["neu"], 4),
        "neg": round(scores["neg"], 4),
        "category": category,
    }


def extract_key_phrases(texts, top_n=30):
    """Extract frequently occurring meaningful phrases from responses."""
    # Common phrases that indicate themes
    theme_keywords = {
        "career": ["career", "growth", "promotion", "advance", "develop", "path", "opportunity"],
        "recognition": ["recognition", "recognized", "appreciated", "acknowledged", "noticed", "celebrate", "thank"],
        "manager": ["manager", "1:1", "coaching", "feedback", "support", "leadership", "skip-level"],
        "work_life_balance": ["balance", "workload", "overtime", "weekend", "flexible", "hybrid", "remote", "burnout"],
        "collaboration": ["team", "collaboration", "cross-functional", "together", "culture"],
        "tools": ["tools", "tooling", "systems", "tech stack", "infrastructure", "outdated"],
        "compensation": ["compensation", "pay", "salary", "equity", "benefits", "market"],
        "inclusion": ["belong", "inclusion", "diversity", "respect", "safe", "trust"],
    }

    theme_counts = Counter()
    theme_sentiments = {theme: [] for theme in theme_keywords}

    analyzer = SentimentIntensityAnalyzer()

    for text in texts:
        if not text:
            continue
        text_lower = text.lower()
        sentiment = analyzer.polarity_scores(text)["compound"]

        for theme, keywords in theme_keywords.items():
            if any(kw in text_lower for kw in keywords):
                theme_counts[theme] += 1
                theme_sentiments[theme].append(sentiment)

    results = []
    for theme, count in theme_counts.most_common():
        sentiments = theme_sentiments[theme]
        results.append({
            "theme": theme,
            "mention_count": count,
            "avg_sentiment": round(np.mean(sentiments), 3) if sentiments else 0,
            "pct_negative": round(sum(1 for s in sentiments if s < -0.05) / len(sentiments) * 100, 1) if sentiments else 0,
        })

    return results


def run_sentiment_analysis(pulse_path="data/pulse_survey_q1_2026.csv",
                           exit_path="data/exit_surveys_2026.csv",
                           onboarding_path="data/onboarding_surveys_2026.csv"):
    """Run full sentiment analysis pipeline."""
    print("Running sentiment analysis pipeline...")

    analyzer = SentimentIntensityAnalyzer()

    # ── Pulse Survey ──
    pulse = pd.read_csv(pulse_path)
    pulse["clean_text"] = pulse["open_text_response"].apply(clean_text)

    sentiment_results = pulse["clean_text"].apply(lambda t: analyze_sentiment(t, analyzer))
    sentiment_df = pd.DataFrame(sentiment_results.tolist())
    pulse = pd.concat([pulse, sentiment_df], axis=1)

    # Filter to responses with text
    has_text = pulse[pulse["clean_text"] != ""]
    print(f"  Pulse: {len(has_text)} responses with open text analyzed")

    # Sentiment distribution
    dist = has_text["category"].value_counts(normalize=True).round(3)
    print(f"  Sentiment distribution: {dist.to_dict()}")

    # Sentiment by department
    dept_sentiment = has_text.groupby("department")["compound"].agg(["mean", "count"]).round(3)
    dept_sentiment.columns = ["avg_sentiment", "response_count"]
    print(f"\n  Sentiment by department:")
    for dept, row in dept_sentiment.sort_values("avg_sentiment").iterrows():
        emoji = "+" if row["avg_sentiment"] > 0 else ""
        print(f"    {dept:25s} {emoji}{row['avg_sentiment']:.3f}  (n={int(row['response_count'])})")

    # Key phrase themes
    themes = extract_key_phrases(has_text["clean_text"].tolist())
    print(f"\n  Top themes from open-ended responses:")
    for t in themes:
        print(f"    {t['theme']:25s} mentions={t['mention_count']:3d}  sentiment={t['avg_sentiment']:+.3f}  %negative={t['pct_negative']:.0f}%")

    # ── Exit Surveys ──
    exits = pd.read_csv(exit_path)
    exits["clean_text"] = exits["exit_comment"].apply(clean_text)
    exit_sentiments = exits["clean_text"].apply(lambda t: analyze_sentiment(t, analyzer))
    exit_df = pd.DataFrame(exit_sentiments.tolist())
    exits = pd.concat([exits, exit_df], axis=1)
    print(f"\n  Exit surveys: avg sentiment = {exits['compound'].mean():.3f}")

    # ── Onboarding ──
    onboarding = pd.read_csv(onboarding_path)
    onboarding["clean_text"] = onboarding["onboarding_comment"].apply(clean_text)
    onb_sentiments = onboarding["clean_text"].apply(lambda t: analyze_sentiment(t, analyzer))
    onb_df = pd.DataFrame(onb_sentiments.tolist())
    onboarding = pd.concat([onboarding, onb_df], axis=1)
    print(f"  Onboarding surveys: avg sentiment = {onboarding['compound'].mean():.3f}")

    # Save enriched data
    pulse.to_csv("data/pulse_survey_enriched.csv", index=False)
    exits.to_csv("data/exit_surveys_enriched.csv", index=False)
    onboarding.to_csv("data/onboarding_surveys_enriched.csv", index=False)

    # Save theme analysis
    theme_df = pd.DataFrame(themes)
    theme_df.to_csv("data/theme_analysis.csv", index=False)

    print("\n  Enriched datasets saved to data/ directory")
    return pulse, exits, onboarding, themes


if __name__ == "__main__":
    run_sentiment_analysis()
