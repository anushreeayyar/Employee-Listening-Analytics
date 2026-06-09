"""
Employee Listening & Insights Dashboard
=========================================
Interactive Streamlit dashboard for exploring employee survey data,
sentiment analysis, topic modeling, and engagement driver analysis.

Author: Anushree Ayyar
Portfolio Project for Employee Listening & Analytics Lead

Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Employee Listening & Insights Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── STYLING ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #312e81, #4f46e5);
        padding: 24px 32px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
    }
    .main-header h1 { margin: 0; font-size: 28px; font-weight: 800; }
    .main-header p { margin: 4px 0 0 0; opacity: 0.8; font-size: 14px; }
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .insight-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
    }
    .alert-box {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, "data")

    data = {}
    files = {
        "pulse": "pulse_survey_enriched.csv",
        "exits": "exit_surveys_enriched.csv",
        "onboarding": "onboarding_surveys_enriched.csv",
        "drivers": "driver_analysis_results.csv",
        "themes": "theme_analysis.csv",
        "topics": "topic_summary.csv",
        "topic_assignments": "topic_assignments.csv",
        "correlations": "correlation_analysis.csv",
        "dept_drivers": "department_drivers.csv",
    }
    for key, filename in files.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            data[key] = pd.DataFrame()

    # Load narrative
    narrative_path = os.path.join(data_dir, "executive_summary.txt")
    if os.path.exists(narrative_path):
        with open(narrative_path) as f:
            data["narrative"] = f.read()
    else:
        data["narrative"] = ""

    return data

data = load_data()
pulse = data["pulse"]
exits = data["exits"]
onboarding = data["onboarding"]

DIMENSIONS = [
    "manager_support", "growth_opportunities", "work_life_balance",
    "recognition", "team_collaboration", "purpose_mission",
    "psychological_safety", "tools_resources", "compensation_fairness",
    "dei_belonging"
]

DIM_LABELS = {
    "manager_support": "Manager Support", "growth_opportunities": "Growth Opportunities",
    "work_life_balance": "Work-Life Balance", "recognition": "Recognition",
    "team_collaboration": "Team Collaboration", "purpose_mission": "Purpose & Mission",
    "psychological_safety": "Psychological Safety", "tools_resources": "Tools & Resources",
    "compensation_fairness": "Compensation Fairness", "dei_belonging": "DEI & Belonging",
}

COLORS = px.colors.qualitative.Set2

# ─── HEADER ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>📊 Employee Listening & Insights Platform</h1>
    <p>NovaTech Inc. — Q1 2026 Pulse Survey | End-to-end survey analytics, NLP insights, and action enablement</p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔍 Filters")

    dept_filter = st.multiselect(
        "Department",
        options=sorted(pulse["department"].unique()) if "department" in pulse.columns else [],
        default=[],
        help="Filter data by department"
    )

    level_filter = st.multiselect(
        "Level",
        options=sorted(pulse["level"].unique()) if "level" in pulse.columns else [],
        default=[]
    )

    tenure_filter = st.multiselect(
        "Tenure",
        options=["0-6 months", "6-12 months", "1-2 years", "2-5 years", "5-10 years", "10+ years"],
        default=[]
    )

    location_filter = st.multiselect(
        "Location",
        options=sorted(pulse["location"].unique()) if "location" in pulse.columns else [],
        default=[]
    )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Portfolio Project**
    by Anushree Ayyar

    Demonstrates end-to-end employee
    listening analytics capabilities:

    - Survey data simulation
    - NLP sentiment analysis
    - Topic modeling (LDA)
    - Engagement driver analysis
    - AI-assisted narrative generation
    - Interactive dashboard
    """)

# Apply filters
filtered = pulse.copy()
if dept_filter:
    filtered = filtered[filtered["department"].isin(dept_filter)]
if level_filter:
    filtered = filtered[filtered["level"].isin(level_filter)]
if tenure_filter:
    filtered = filtered[filtered["tenure"].isin(tenure_filter)]
if location_filter:
    filtered = filtered[filtered["location"].isin(location_filter)]

# ─── TABS ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview",
    "💬 Sentiment Analysis",
    "🔬 Topic Modeling",
    "🎯 Driver Analysis",
    "👤 Manager Hub",
    "📝 AI Narrative"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### Key Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        eng = filtered["overall_engagement"].mean() if len(filtered) > 0 else 0
        st.metric("Overall Engagement", f"{eng:.1f}", delta=f"{eng - 72:.1f} vs baseline")
    with col2:
        if "enps_category" in filtered.columns and len(filtered) > 0:
            enps = ((filtered["enps_category"] == "Promoter").mean() -
                    (filtered["enps_category"] == "Detractor").mean()) * 100
            st.metric("eNPS", f"{enps:.0f}", delta=f"{enps - (-11):.0f} vs Q4")
        else:
            st.metric("eNPS", "N/A")
    with col3:
        st.metric("Responses", f"{len(filtered):,}", delta=f"{len(filtered)/850*100:.0f}% rate")
    with col4:
        has_text = filtered[filtered.get("open_text_response", pd.Series(dtype=str)).fillna("") != ""]
        st.metric("Open Text Responses", f"{len(has_text):,}", delta=f"{len(has_text)/max(len(filtered),1)*100:.0f}%")
    with col5:
        st.metric("Departments", f"{filtered['department'].nunique()}" if "department" in filtered.columns else "0")

    st.markdown("---")

    # Engagement by department
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Engagement by Department")
        if "department" in filtered.columns:
            dept_eng = filtered.groupby("department")["overall_engagement"].mean().sort_values(ascending=True).reset_index()
            dept_eng.columns = ["Department", "Engagement"]

            fig = px.bar(
                dept_eng, x="Engagement", y="Department", orientation="h",
                color="Engagement",
                color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
                range_color=[55, 85],
            )
            fig.add_vline(x=filtered["overall_engagement"].mean(), line_dash="dash",
                         line_color="gray", annotation_text="Company Avg")
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0),
                            coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### eNPS Distribution")
        if "enps_category" in filtered.columns:
            enps_dist = filtered["enps_category"].value_counts().reset_index()
            enps_dist.columns = ["Category", "Count"]
            color_map = {"Promoter": "#22c55e", "Passive": "#f59e0b", "Detractor": "#ef4444"}
            fig = px.pie(enps_dist, values="Count", names="Category",
                        color="Category", color_discrete_map=color_map,
                        hole=0.4)
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

    # Dimension scores heatmap
    st.markdown("#### Engagement Dimensions by Department")
    available_dims = [d for d in DIMENSIONS if d in filtered.columns]
    if available_dims and "department" in filtered.columns:
        heatmap_data = filtered.groupby("department")[available_dims].mean()
        heatmap_data.columns = [DIM_LABELS.get(d, d) for d in heatmap_data.columns]

        fig = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale="RdYlGn",
            zmin=55, zmax=85,
            text_auto=".0f",
            aspect="auto",
        )
        fig.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0),
                         xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # Lifecycle snapshot
    st.markdown("#### Employee Lifecycle Survey Snapshot")
    lifecycle_cols = st.columns(3)
    with lifecycle_cols[0]:
        st.markdown("**Onboarding (30-day)**")
        onb30 = onboarding[onboarding["survey_type"] == "onboarding_30day"] if "survey_type" in onboarding.columns else pd.DataFrame()
        if len(onb30) > 0:
            st.metric("Satisfaction", f"{onb30['onboarding_satisfaction'].mean():.0f}/100")
            st.metric("Responses", f"{len(onb30)}")
    with lifecycle_cols[1]:
        st.markdown("**Onboarding (90-day)**")
        onb90 = onboarding[onboarding["survey_type"] == "onboarding_90day"] if "survey_type" in onboarding.columns else pd.DataFrame()
        if len(onb90) > 0:
            st.metric("Satisfaction", f"{onb90['onboarding_satisfaction'].mean():.0f}/100")
            st.metric("Responses", f"{len(onb90)}")
    with lifecycle_cols[2]:
        st.markdown("**Exit Surveys**")
        if len(exits) > 0:
            st.metric("Satisfaction", f"{exits['overall_satisfaction'].mean():.0f}/100")
            st.metric("Responses", f"{len(exits)}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### NLP Sentiment Analysis")
    st.markdown("VADER sentiment analysis applied to open-text survey responses to classify employee sentiment and extract themes.")

    col_s1, col_s2 = st.columns([2, 1])

    with col_s1:
        st.markdown("#### Sentiment Distribution")
        if "category" in filtered.columns:
            has_resp = filtered[filtered["category"] != "no_response"]
            sent_dist = has_resp["category"].value_counts().reset_index()
            sent_dist.columns = ["Sentiment", "Count"]
            color_map = {"very_positive": "#059669", "positive": "#34d399",
                        "neutral": "#9ca3af", "negative": "#f59e0b", "very_negative": "#ef4444"}
            fig = px.bar(sent_dist, x="Sentiment", y="Count", color="Sentiment",
                        color_discrete_map=color_map)
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
                            showlegend=False, xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    with col_s2:
        st.markdown("#### Sentiment Summary")
        if "category" in filtered.columns:
            has_resp = filtered[filtered["category"] != "no_response"]
            pos_pct = ((has_resp["category"] == "very_positive") | (has_resp["category"] == "positive")).mean() * 100
            neg_pct = ((has_resp["category"] == "very_negative") | (has_resp["category"] == "negative")).mean() * 100
            avg_compound = has_resp["compound"].mean()

            st.metric("Avg Compound Score", f"{avg_compound:.3f}")
            st.metric("% Positive", f"{pos_pct:.0f}%")
            st.metric("% Negative", f"{neg_pct:.0f}%")
            st.metric("Total Analyzed", f"{len(has_resp)}")

    # Sentiment by department
    st.markdown("#### Sentiment by Department")
    if "compound" in filtered.columns and "department" in filtered.columns:
        has_resp = filtered[filtered["category"] != "no_response"]
        dept_sent = has_resp.groupby("department")["compound"].agg(["mean", "count"]).reset_index()
        dept_sent.columns = ["Department", "Avg Sentiment", "Responses"]
        dept_sent = dept_sent.sort_values("Avg Sentiment")

        fig = px.bar(dept_sent, x="Avg Sentiment", y="Department", orientation="h",
                    color="Avg Sentiment", color_continuous_scale="RdYlGn",
                    range_color=[-0.5, 1.0],
                    text=dept_sent["Avg Sentiment"].apply(lambda x: f"{x:+.3f}"))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
                         coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # Theme analysis
    st.markdown("#### Extracted Themes from Open-Text Responses")
    themes = data.get("themes", pd.DataFrame())
    if len(themes) > 0:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Mention Frequency", "Avg Sentiment by Theme"))

        fig.add_trace(
            go.Bar(x=themes["mention_count"], y=themes["theme"], orientation="h",
                   marker_color="#4f46e5", name="Mentions"),
            row=1, col=1
        )
        colors = ["#22c55e" if s > 0 else "#ef4444" for s in themes["avg_sentiment"]]
        fig.add_trace(
            go.Bar(x=themes["avg_sentiment"], y=themes["theme"], orientation="h",
                   marker_color=colors, name="Sentiment"),
            row=1, col=2
        )
        fig.update_layout(height=350, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Sample comments
    st.markdown("#### Sample Open-Text Responses")
    if "clean_text" in filtered.columns:
        has_text_df = filtered[filtered["clean_text"].fillna("") != ""][["department", "level", "category", "compound", "clean_text"]]
        has_text_df.columns = ["Department", "Level", "Sentiment", "Score", "Response"]
        st.dataframe(has_text_df.head(20), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: TOPIC MODELING
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### Topic Modeling (LDA)")
    st.markdown("Latent Dirichlet Allocation discovers hidden themes in employee feedback without predefined categories.")

    topics = data.get("topics", pd.DataFrame())
    topic_assignments = data.get("topic_assignments", pd.DataFrame())

    if len(topics) > 0:
        st.markdown("#### Discovered Topics")
        for _, row in topics.iterrows():
            with st.expander(f"🔬 **{row['label']}** — {row['prevalence_pct']}% of responses"):
                st.markdown(f"**Top Keywords:** {row['top_words']}")
                st.markdown(f"**Prevalence:** {row['prevalence_pct']}% of responses have this as a significant theme")

        # Topic prevalence chart
        st.markdown("#### Topic Prevalence")
        fig = px.bar(topics, x="prevalence_pct", y="label", orientation="h",
                    color="prevalence_pct", color_continuous_scale="Viridis",
                    labels={"prevalence_pct": "% of Responses", "label": "Topic"})
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                         coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    if len(topic_assignments) > 0:
        st.markdown("#### Topic Distribution")
        topic_dist = topic_assignments["topic_label"].value_counts().reset_index()
        topic_dist.columns = ["Topic", "Count"]
        fig = px.pie(topic_dist, values="Count", names="Topic", hole=0.3,
                    color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Topic Assignments (Sample)")
        sample = topic_assignments[["text", "topic_label", "topic_confidence"]].head(15)
        sample.columns = ["Response", "Dominant Topic", "Confidence"]
        st.dataframe(sample, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: DRIVER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("### Engagement Driver Analysis")
    st.markdown("Statistical analysis identifying which experience dimensions have the greatest impact on overall engagement.")

    drivers = data.get("drivers", pd.DataFrame())
    correlations = data.get("correlations", pd.DataFrame())

    if len(drivers) > 0:
        st.markdown("#### Priority Matrix: Impact vs. Current Score")

        fig = px.scatter(
            drivers, x="current_score", y="relative_importance_pct",
            size="priority_score", color="priority_score",
            color_continuous_scale="YlOrRd",
            text="dimension",
            labels={
                "current_score": "Current Score (lower = more opportunity)",
                "relative_importance_pct": "Impact on Engagement (%)",
                "priority_score": "Priority"
            },
            size_max=30,
        )
        fig.update_traces(textposition="top center", textfont_size=10)
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0))

        # Add quadrant lines
        fig.add_hline(y=drivers["relative_importance_pct"].mean(), line_dash="dash",
                     line_color="gray", opacity=0.5)
        fig.add_vline(x=drivers["current_score"].mean(), line_dash="dash",
                     line_color="gray", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)

        st.info("💡 **How to read this:** Dimensions in the **upper-left quadrant** (high impact, low score) are your highest priority focus areas.")

        # Driver ranking table
        st.markdown("#### Driver Rankings")
        driver_display = drivers[["dimension", "current_score", "relative_importance_pct", "correlation_with_engagement", "priority_score"]].copy()
        driver_display.columns = ["Dimension", "Score", "Impact (%)", "Correlation", "Priority"]
        driver_display = driver_display.sort_values("Priority", ascending=False)
        st.dataframe(driver_display, use_container_width=True, hide_index=True)

    # Correlation chart
    if len(correlations) > 0:
        st.markdown("#### Correlation with Overall Engagement")
        correlations_sorted = correlations.sort_values("correlation", ascending=True)
        fig = px.bar(correlations_sorted, x="correlation", y="dimension", orientation="h",
                    color="correlation", color_continuous_scale="Blues",
                    labels={"correlation": "Pearson r", "dimension": ""})
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0),
                         coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Department drivers
    dept_drivers = data.get("dept_drivers", pd.DataFrame())
    if len(dept_drivers) > 0:
        st.markdown("#### Top Engagement Driver by Department")
        st.dataframe(dept_drivers.sort_values("avg_engagement"), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: MANAGER HUB
# ═══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("### Manager Action Planning Hub")
    st.markdown("Personalized insights and AI-generated action recommendations for people managers.")

    if "department" in filtered.columns:
        selected_dept = st.selectbox(
            "Select your department:",
            sorted(filtered["department"].unique())
        )

        dept_data = filtered[filtered["department"] == selected_dept]

        if len(dept_data) > 0:
            st.markdown(f"#### {selected_dept} — Team Insights")

            mc1, mc2, mc3 = st.columns(3)
            dept_eng = dept_data["overall_engagement"].mean()
            company_eng = filtered["overall_engagement"].mean()
            with mc1:
                st.metric("Team Engagement", f"{dept_eng:.0f}",
                          delta=f"{dept_eng - company_eng:.0f} vs company")
            with mc2:
                if "enps_category" in dept_data.columns:
                    dept_enps = ((dept_data["enps_category"] == "Promoter").mean() -
                                (dept_data["enps_category"] == "Detractor").mean()) * 100
                    st.metric("Team eNPS", f"{dept_enps:.0f}")
            with mc3:
                st.metric("Response Rate", f"{len(dept_data)/filtered['department'].value_counts().get(selected_dept, 1)*100:.0f}%")

            # Radar chart: team vs company
            available_dims = [d for d in DIMENSIONS if d in dept_data.columns]
            if available_dims:
                team_scores = [dept_data[d].mean() for d in available_dims]
                company_scores = [filtered[d].mean() for d in available_dims]
                dim_names = [DIM_LABELS.get(d, d) for d in available_dims]

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=team_scores + [team_scores[0]],
                    theta=dim_names + [dim_names[0]],
                    fill="toself", fillcolor="rgba(79, 70, 229, 0.15)",
                    line=dict(color="#4f46e5", width=2),
                    name="Your Team"
                ))
                fig.add_trace(go.Scatterpolar(
                    r=company_scores + [company_scores[0]],
                    theta=dim_names + [dim_names[0]],
                    fill="none",
                    line=dict(color="#94a3b8", width=1.5, dash="dash"),
                    name="Company Avg"
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(range=[40, 100])),
                    height=450, margin=dict(l=40, r=40, t=30, b=30),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15),
                )
                st.plotly_chart(fig, use_container_width=True)

            # AI action recommendations
            st.markdown("#### 🤖 AI-Suggested Focus Areas")

            dim_scores = {d: dept_data[d].mean() for d in available_dims}
            weakest = sorted(dim_scores.items(), key=lambda x: x[1])[:3]

            action_library = {
                "recognition": [
                    "Start each team meeting with a 2-min kudos moment",
                    "Send personalized appreciation messages after project milestones",
                    "Nominate a team member for a company-wide recognition award this quarter",
                ],
                "growth_opportunities": [
                    "Schedule career development 1:1s with each direct report this month",
                    "Identify one stretch assignment per team member",
                    "Share internal job postings proactively and encourage exploration",
                ],
                "tools_resources": [
                    "Run a 15-min team retro specifically focused on tooling pain points",
                    "Submit a prioritized tooling improvement request to IT",
                    "Allocate 10% of sprint capacity for developer experience improvements",
                ],
                "work_life_balance": [
                    "Audit meeting load — cancel or shorten 2 recurring meetings this month",
                    "Establish 'no-meeting' focus time blocks for the team",
                    "Model healthy boundaries by not sending after-hours messages",
                ],
                "compensation_fairness": [
                    "Have transparent conversations about comp philosophy and review cycle",
                    "Advocate for market adjustments for key talent with HR",
                    "Ensure total rewards (equity, benefits, perks) are well-understood",
                ],
                "manager_support": [
                    "Increase 1:1 cadence to weekly for any reports flagging concerns",
                    "Ask each report: 'What's one thing I could do differently?'",
                    "Share your own development goals to build trust and vulnerability",
                ],
                "psychological_safety": [
                    "Adopt team norms that explicitly encourage dissent and questions",
                    "Respond to mistakes with curiosity, not blame",
                    "Share a recent mistake of your own in a team setting",
                ],
            }

            for dim, score in weakest:
                label = DIM_LABELS.get(dim, dim)
                actions = action_library.get(dim, [
                    f"Discuss {label} openly in your next team meeting",
                    "Ask each team member for one specific improvement suggestion",
                    "Commit to one visible change within 30 days",
                ])

                with st.container():
                    st.markdown(f"""
                    <div class="alert-box">
                        <strong>⚠️ {label}</strong> — Score: {score:.0f}/100
                        (Company avg: {filtered[dim].mean():.0f})
                    </div>
                    """, unsafe_allow_html=True)
                    for a in actions:
                        st.checkbox(a, key=f"{selected_dept}_{dim}_{a[:20]}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: AI NARRATIVE
# ═══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.markdown("### AI-Generated Executive Summary")
    st.markdown("Automatically generated narrative report synthesizing survey findings, NLP analysis, and statistical insights.")

    narrative = data.get("narrative", "")
    if narrative:
        st.code(narrative, language=None)
    else:
        st.info("Run the narrative generator to produce the executive summary: `python analysis/narrative_generator.py`")

    st.markdown("---")
    st.markdown("#### How This Works")
    st.markdown("""
    The AI narrative generator combines outputs from multiple analysis modules:

    1. **Quantitative metrics** — engagement scores, eNPS, response rates
    2. **NLP sentiment analysis** — VADER compound scores and sentiment categories
    3. **Topic modeling** — LDA-extracted themes from open-text responses
    4. **Driver analysis** — regression-based impact weights and priority scoring
    5. **Template-based generation** — structured narrative with data-driven fill-ins

    In a production environment, this would integrate with an LLM API (e.g., Claude)
    to generate more nuanced, context-aware narratives that adapt to each audience.
    """)

# ─── FOOTER ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 12px;">
    Portfolio Project by <strong>Anushree Ayyar</strong> | Employee Listening & Insights Lead<br>
    Built with Python, VADER, scikit-learn (LDA), Streamlit, Plotly<br>
    Skills demonstrated: Survey Design, People Analytics, NLP, Topic Modeling, Data Storytelling, AI-Enabled Insights, Manager Enablement
</div>
""", unsafe_allow_html=True)
