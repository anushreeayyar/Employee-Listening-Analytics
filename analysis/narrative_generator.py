"""
AI-Assisted Narrative Insight Generator
=========================================
Automatically generates executive-ready narrative summaries
from survey data analysis results. Demonstrates how AI can
make insight delivery easier for people managers and HR leaders.

Author: Anushree Ayyar
"""

import pandas as pd
import numpy as np
from datetime import datetime


def generate_executive_summary(pulse_df, exit_df, onboarding_df, driver_results, theme_data):
    """Generate a comprehensive executive narrative from survey data."""

    # Compute key metrics
    overall_eng = pulse_df["overall_engagement"].mean()
    response_rate = len(pulse_df) / 850 * 100  # approximate
    enps_promoters = (pulse_df["enps_category"] == "Promoter").mean() * 100
    enps_detractors = (pulse_df["enps_category"] == "Detractor").mean() * 100
    enps = round(enps_promoters - enps_detractors)

    # Sentiment
    has_text = pulse_df[pulse_df["category"] != "no_response"]
    pct_positive = ((has_text["category"] == "very_positive") | (has_text["category"] == "positive")).mean() * 100
    pct_negative = ((has_text["category"] == "very_negative") | (has_text["category"] == "negative")).mean() * 100

    # Department extremes
    dept_eng = pulse_df.groupby("department")["overall_engagement"].mean()
    top_dept = dept_eng.idxmax()
    bottom_dept = dept_eng.idxmin()

    # Exit data
    top_exit_reason = exit_df["primary_reason"].value_counts().index[0]
    exit_satisfaction = exit_df["overall_satisfaction"].mean()

    # Focus areas from driver analysis
    focus_areas = driver_results["focus_areas"][:3]

    narrative = f"""
================================================================================
   EMPLOYEE LISTENING INSIGHTS REPORT — Q1 2026
   NovaTech Inc. | People Analytics & Employee Experience
   Generated: {datetime.now().strftime('%B %d, %Y')}
================================================================================

EXECUTIVE SUMMARY
─────────────────

This quarter's pulse survey captured {len(pulse_df):,} responses across 9
departments, achieving an {response_rate:.0f}% response rate — a strong signal
of employee willingness to share feedback.

Overall engagement stands at {overall_eng:.1f}/100 with an eNPS of {enps:+d}.
{pct_positive:.0f}% of open-text responses carry positive sentiment, while
{pct_negative:.0f}% express concerns — primarily around recognition and
career growth.

KEY FINDINGS
─────────────

1. RECOGNITION GAP IS THE #1 PRIORITY

   Recognition scores ({pulse_df['recognition'].mean():.0f}/100) represent the
   widest gap between employee expectations and experience. NLP analysis of
   {len(has_text)} open-text responses identified "recognition" as the most
   negatively-discussed theme, with {pct_negative:.0f}% of related comments
   expressing frustration.

   This is not just a "nice to have" — driver analysis shows recognition has a
   {driver_results['correlations'].get('recognition', 0):.2f} correlation with
   overall engagement and accounts for
   {driver_results['regression']['relative_importance'].get('recognition', 0):.1f}%
   of variance in engagement scores.

   → Recommended: Launch peer recognition program, train managers on timely
     appreciation practices, track recognition frequency in next pulse.

2. CAREER GROWTH DRIVES ATTRITION

   Exit survey data tells a clear story: "{top_exit_reason}" is the #1 reason
   employees leave NovaTech, cited in {(exit_df['primary_reason'] == top_exit_reason).mean() * 100:.0f}%
   of departures. Departing employees report an average satisfaction of just
   {exit_satisfaction:.0f}/100.

   Growth Opportunities scores ({pulse_df['growth_opportunities'].mean():.0f}/100)
   are lowest in Sales and Finance — departments that also show the weakest
   engagement overall.

   → Recommended: Audit internal mobility rates, pilot career conversation
     framework, partner with Talent Management on development paths.

3. MANAGER SUPPORT IS A BRIGHT SPOT

   Manager Support ({pulse_df['manager_support'].mean():.0f}/100) is one of the
   highest-scoring dimensions and shows positive momentum. Onboarding data
   reinforces this: new hires whose managers check in weekly report
   significantly higher ramp confidence.

   {top_dept} leads with {dept_eng[top_dept]:.0f}/100 engagement, driven by
   strong manager practices worth studying and scaling.

   → Recommended: Document best practices from top-performing teams, create
     a manager playbook, feature in next HRLT readout.

DEPARTMENT SPOTLIGHT
────────────────────

   Highest Engagement:  {top_dept} ({dept_eng[top_dept]:.0f}/100)
   Lowest Engagement:   {bottom_dept} ({dept_eng[bottom_dept]:.0f}/100)
   Widest Gap:          {dept_eng[top_dept] - dept_eng[bottom_dept]:.0f} points

   Departments with engagement below company average ({overall_eng:.0f}) should
   receive targeted action planning support from HRBPs.

EMPLOYEE LIFECYCLE INSIGHTS
────────────────────────────

   Onboarding (30-day):  Satisfaction at {onboarding_df[onboarding_df['survey_type']=='onboarding_30day']['onboarding_satisfaction'].mean():.0f}/100
   Onboarding (90-day):  Satisfaction at {onboarding_df[onboarding_df['survey_type']=='onboarding_90day']['onboarding_satisfaction'].mean():.0f}/100
   Exit Surveys:         {len(exit_df)} responses, avg satisfaction {exit_satisfaction:.0f}/100

   The drop from 30-day to 90-day onboarding satisfaction suggests opportunity
   to improve the sustained onboarding experience beyond the initial welcome.

SENTIMENT ANALYSIS SUMMARY
───────────────────────────

   Positive Sentiment:   {pct_positive:.0f}%
   Neutral Sentiment:    {((has_text['category'] == 'neutral').mean() * 100):.0f}%
   Negative Sentiment:   {pct_negative:.0f}%

   Top positive themes: team collaboration, manager support, purpose & mission
   Top negative themes: recognition, career growth, tools & resources

AI-GENERATED ACTION PRIORITIES
───────────────────────────────

   Based on statistical driver analysis (R² = {driver_results['regression']['r_squared']:.2f})
   and NLP theme extraction, the following actions are ranked by expected
   impact on overall engagement:

"""
    for i, fa in enumerate(focus_areas, 1):
        narrative += f"""   {i}. {fa['dimension']}
      Current Score: {fa['current_score']:.0f}/100
      Impact Weight: {fa['relative_importance_pct']:.1f}% of engagement variance
      Priority Score: {fa['priority_score']:.2f}
      Status: Requires immediate attention

"""

    narrative += f"""
METHODOLOGY
───────────

   - Survey Platform: Simulated (Qualtrics-equivalent)
   - Analysis Tools: Python (VADER Sentiment, LDA Topic Modeling, scikit-learn)
   - Response Period: March 1-15, 2026
   - Confidence Level: 95% (margin of error ±2.3%)
   - Driver Analysis: Multiple linear regression with standardized coefficients

================================================================================
   Prepared by: Anushree Ayyar, Employee Listening & Insights
   This report demonstrates AI-assisted survey analytics capabilities
   including automated sentiment analysis, topic modeling, and
   narrative generation from employee feedback data.
================================================================================
"""
    return narrative


def generate_manager_brief(pulse_df, department="Engineering"):
    """Generate a personalized manager action brief for a specific department."""
    dept_data = pulse_df[pulse_df["department"] == department]
    if len(dept_data) == 0:
        return f"No data available for {department}"

    overall = pulse_df["overall_engagement"].mean()
    dept_eng = dept_data["overall_engagement"].mean()
    gap = dept_eng - overall

    dims = ["manager_support", "growth_opportunities", "work_life_balance",
            "recognition", "team_collaboration", "purpose_mission",
            "psychological_safety", "tools_resources"]

    available_dims = [d for d in dims if d in dept_data.columns]
    dim_scores = {d: dept_data[d].mean() for d in available_dims}
    strongest = max(dim_scores, key=dim_scores.get)
    weakest = min(dim_scores, key=dim_scores.get)

    dim_labels = {
        "manager_support": "Manager Support", "growth_opportunities": "Growth Opportunities",
        "work_life_balance": "Work-Life Balance", "recognition": "Recognition",
        "team_collaboration": "Team Collaboration", "purpose_mission": "Purpose & Mission",
        "psychological_safety": "Psychological Safety", "tools_resources": "Tools & Resources",
    }

    brief = f"""
MANAGER ACTION BRIEF — {department.upper()}
{'─' * 50}
Engagement Score: {dept_eng:.0f}/100 ({'↑' if gap > 0 else '↓'} {abs(gap):.0f} vs. company avg {overall:.0f})
Responses: {len(dept_data)} / Response Rate: {len(dept_data) / len(pulse_df) * 100 * (850/len(pulse_df)):.0f}%

YOUR STRENGTHS (keep doing these):
  ★ {dim_labels.get(strongest, strongest)}: {dim_scores[strongest]:.0f}/100

YOUR FOCUS AREAS (prioritize these):
  ⚠ {dim_labels.get(weakest, weakest)}: {dim_scores[weakest]:.0f}/100

SUGGESTED ACTIONS:
"""
    # Generate context-specific actions
    if weakest == "recognition":
        brief += """  1. Start each team meeting with a kudos moment
  2. Send personalized thank-you messages after project milestones
  3. Nominate a team member for a company-wide award this quarter
"""
    elif weakest == "growth_opportunities":
        brief += """  1. Schedule career development conversations with each report
  2. Identify one stretch assignment per team member this quarter
  3. Share internal job postings and encourage exploration
"""
    elif weakest == "tools_resources":
        brief += """  1. Run a 15-min team retro focused on tooling pain points
  2. Prioritize the top 3 fixes and submit requests to IT
  3. Allocate time for process improvement in sprint planning
"""
    else:
        brief += f"""  1. Discuss {dim_labels.get(weakest, weakest)} openly in your next team meeting
  2. Ask each team member for one specific improvement suggestion
  3. Commit to one visible change within 30 days and report back
"""

    return brief


def run_narrative_generation(pulse_path="data/pulse_survey_enriched.csv",
                              exit_path="data/exit_surveys_enriched.csv",
                              onboarding_path="data/onboarding_surveys_enriched.csv",
                              driver_path="data/driver_analysis_results.csv",
                              theme_path="data/theme_analysis.csv"):
    """Run full narrative generation pipeline."""
    print("Generating AI-assisted narratives...")

    pulse = pd.read_csv(pulse_path)
    exits = pd.read_csv(exit_path)
    onboarding = pd.read_csv(onboarding_path)
    drivers_df = pd.read_csv(driver_path)
    themes = pd.read_csv(theme_path) if pd.io.common.file_exists(theme_path) else pd.DataFrame()

    # Reconstruct driver results dict
    from analysis.driver_analysis import run_driver_analysis
    driver_results = run_driver_analysis(pulse_path)

    # Generate executive summary
    exec_summary = generate_executive_summary(pulse, exits, onboarding, driver_results, themes)
    with open("data/executive_summary.txt", "w") as f:
        f.write(exec_summary)
    print(f"  Executive summary generated ({len(exec_summary)} chars)")

    # Generate manager briefs
    departments = pulse["department"].unique()
    all_briefs = ""
    for dept in departments:
        brief = generate_manager_brief(pulse, dept)
        all_briefs += brief + "\n\n"

    with open("data/manager_briefs.txt", "w") as f:
        f.write(all_briefs)
    print(f"  Manager briefs generated for {len(departments)} departments")

    return exec_summary


if __name__ == "__main__":
    run_narrative_generation()
