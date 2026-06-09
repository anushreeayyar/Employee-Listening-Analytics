# Employee Listening Analytics Dashboard

**End-to-end Voice of Employee (VOE) analytics platform** demonstrating survey design, NLP sentiment analysis, topic modeling, engagement driver analysis, AI-assisted narrative generation, and interactive data storytelling.

Built by **Anushree Ayyar** as a portfolio project for Employee Listening & Insights roles.

---

## What This Project Demonstrates

| Capability | Implementation |
|---|---|
| **Survey Data Simulation** | Realistic synthetic dataset with 850 employees, lifecycle surveys, demographic fields, and open-text responses |
| **NLP Sentiment Analysis** | VADER sentiment scoring on open-text responses with theme extraction and department-level breakdowns |
| **Topic Modeling** | Latent Dirichlet Allocation (LDA) to discover latent themes from employee feedback without predefined categories |
| **Driver Analysis** | Multiple linear regression identifying which experience dimensions most impact overall engagement |
| **AI-Assisted Insights** | Auto-generated executive narratives, manager action briefs, and priority recommendations |
| **Interactive Dashboard** | Streamlit app with Plotly visualizations, department filters, and manager action planning tools |

---

## Project Structure

```
employee-listening-analytics/
├── data/
│   ├── generate_survey_data.py      # Synthetic data generator
│   ├── pulse_survey_q1_2026.csv     # Raw pulse survey (722 responses)
│   ├── exit_surveys_2026.csv        # Exit survey data (38 responses)
│   ├── onboarding_surveys_2026.csv  # 30/90-day onboarding (278 responses)
│   ├── pulse_survey_enriched.csv    # Sentiment-enriched pulse data
│   └── ...                          # Analysis outputs
├── analysis/
│   ├── sentiment_analysis.py        # VADER NLP pipeline
│   ├── topic_modeling.py            # LDA topic extraction
│   ├── driver_analysis.py           # Regression-based driver analysis
│   └── narrative_generator.py       # AI narrative generation
├── dashboard/
│   └── app.py                       # Streamlit interactive dashboard
├── run_pipeline.py                  # Full pipeline runner
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/employee-listening-analytics.git
cd employee-listening-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (generates data + runs all analyses)
python run_pipeline.py

# 4. Launch the interactive dashboard
streamlit run dashboard/app.py
```

---

## Dashboard Features

The Streamlit dashboard includes 6 interactive views:

1. **Overview** — Key metrics (engagement, eNPS, response rate), department heatmap, lifecycle survey snapshot
2. **Sentiment Analysis** — NLP sentiment distribution, department-level sentiment, theme extraction, sample responses
3. **Topic Modeling** — LDA-discovered themes, topic prevalence, sample assignments with confidence scores
4. **Driver Analysis** — Priority matrix (impact vs. score), correlation analysis, department-level drivers
5. **Manager Hub** — Department-specific insights, radar chart comparison, AI-suggested action plans with checkboxes
6. **AI Narrative** — Auto-generated executive summary synthesizing all analysis modules

---

## Key Findings (Sample Output)

- **Recognition** is the #1 priority focus area (score: 69/100, high impact on engagement)
- **Growth Opportunities** drives 35% of voluntary attrition (exit survey analysis)
- **Manager Support** is a bright spot (82/100) — best practices worth scaling
- **Psychological Safety** improved 6pts, linked to team norms initiative
- Sales and Finance departments have the lowest engagement and need targeted action plans

---

## Technical Stack

- **Python 3.10+**
- **VADER Sentiment** — lexicon-based sentiment analysis tuned for social/survey text
- **scikit-learn** — LDA topic modeling, TF-IDF, linear regression
- **Streamlit** — interactive web dashboard
- **Plotly** — data visualizations (bar, radar, heatmap, scatter, pie)
- **pandas / numpy** — data manipulation and statistical analysis

---

## Methodology

### Sentiment Analysis
VADER (Valence Aware Dictionary and sEntiment Reasoner) classifies open-text responses into 5 sentiment tiers. Theme extraction uses keyword-based matching with sentiment scoring per theme.

### Topic Modeling
Latent Dirichlet Allocation (LDA) with 6 topics, using bag-of-words with survey-specific stop words removed. Bigrams are included to capture phrases like "career growth" and "work life."

### Driver Analysis
Multiple linear regression with standardized coefficients (beta weights) determines relative importance of each experience dimension. Priority scoring combines statistical impact with current performance gaps.

### Narrative Generation
Template-based generation with data-driven fill-ins. In production, this would integrate with an LLM API for context-aware narrative adaptation.

---

## Extending This Project

Ideas for enhancement:
- Integrate a real LLM (Claude API) for dynamic narrative generation
- Add longitudinal trend analysis across multiple survey waves
- Implement BERTopic for neural topic modeling
- Add predictive attrition modeling using engagement + exit data
- Connect to a live survey platform API (Qualtrics, Culture Amp, Glint)

---

## Author

**Anushree Ayyar**
- Portfolio project demonstrating Employee Listening & Analytics capabilities
- Skills: Survey Design, People Analytics, NLP, Data Storytelling, AI-Enabled Insights, Program Management
