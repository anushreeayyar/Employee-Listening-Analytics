"""
Full Analysis Pipeline Runner
==============================
Runs all analysis modules in sequence:
1. Generate synthetic survey data
2. Run NLP sentiment analysis
3. Run topic modeling
4. Run engagement driver analysis
5. Generate AI-assisted narratives

Usage: python run_pipeline.py
Then:  streamlit run dashboard/app.py

Author: Anushree Ayyar
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("  EMPLOYEE LISTENING ANALYTICS PIPELINE")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[1/5] Generating synthetic survey data...")
    from data.generate_survey_data import main as generate_data
    generate_data()

    # Step 2: Sentiment analysis
    print("\n" + "=" * 60)
    print("[2/5] Running sentiment analysis...")
    from analysis.sentiment_analysis import run_sentiment_analysis
    pulse, exits, onboarding, themes = run_sentiment_analysis()

    # Step 3: Topic modeling
    print("\n" + "=" * 60)
    print("[3/5] Running topic modeling...")
    from analysis.topic_modeling import run_topic_modeling
    import pandas as pd
    all_texts = list(pulse["open_text_response"].dropna()) + list(exits["exit_comment"].dropna())
    topic_results = run_topic_modeling(all_texts)

    # Step 4: Driver analysis
    print("\n" + "=" * 60)
    print("[4/5] Running driver analysis...")
    from analysis.driver_analysis import run_driver_analysis
    driver_results = run_driver_analysis()

    # Step 5: Narrative generation
    print("\n" + "=" * 60)
    print("[5/5] Generating AI narratives...")
    from analysis.narrative_generator import generate_executive_summary, generate_manager_brief

    pulse_enriched = pd.read_csv("data/pulse_survey_enriched.csv")
    exits_enriched = pd.read_csv("data/exit_surveys_enriched.csv")
    onboarding_enriched = pd.read_csv("data/onboarding_surveys_enriched.csv")
    themes_df = pd.read_csv("data/theme_analysis.csv")

    exec_summary = generate_executive_summary(
        pulse_enriched, exits_enriched, onboarding_enriched, driver_results, themes_df
    )
    with open("data/executive_summary.txt", "w") as f:
        f.write(exec_summary)
    print(f"  Executive summary saved ({len(exec_summary)} chars)")

    all_briefs = ""
    for dept in pulse_enriched["department"].unique():
        all_briefs += generate_manager_brief(pulse_enriched, dept) + "\n\n"
    with open("data/manager_briefs.txt", "w") as f:
        f.write(all_briefs)
    print(f"  Manager briefs saved for {pulse_enriched['department'].nunique()} departments")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nTo launch the dashboard:")
    print("  streamlit run dashboard/app.py")
    print()


if __name__ == "__main__":
    main()
