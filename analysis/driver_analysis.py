"""
Engagement Driver Analysis Module
===================================
Identifies key drivers of overall engagement using:
- Correlation analysis
- Multiple linear regression
- Relative importance (contribution %)

Author: Anushree Ayyar
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


DIMENSION_LABELS = {
    "manager_support": "Manager Support",
    "growth_opportunities": "Growth Opportunities",
    "work_life_balance": "Work-Life Balance",
    "recognition": "Recognition & Appreciation",
    "team_collaboration": "Team Collaboration",
    "purpose_mission": "Purpose & Mission",
    "psychological_safety": "Psychological Safety",
    "tools_resources": "Tools & Resources",
    "compensation_fairness": "Compensation Fairness",
    "dei_belonging": "DEI & Belonging",
}

DIMENSIONS = list(DIMENSION_LABELS.keys())


def compute_correlations(df):
    """Compute correlation of each dimension with overall engagement."""
    correlations = {}
    for dim in DIMENSIONS:
        if dim in df.columns:
            corr = df[dim].corr(df["overall_engagement"])
            correlations[dim] = round(corr, 3)

    sorted_corr = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
    return sorted_corr


def run_regression(df):
    """Run multiple linear regression to identify key drivers."""
    available = [d for d in DIMENSIONS if d in df.columns]
    X = df[available].dropna()
    y = df.loc[X.index, "overall_engagement"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    # Standardized coefficients (beta weights)
    betas = dict(zip(available, [round(c, 4) for c in model.coef_]))

    # Relative importance (proportion of R² explained by each predictor)
    r_squared = model.score(X_scaled, y)
    abs_betas = np.abs(model.coef_)
    relative_importance = abs_betas / abs_betas.sum()
    importance = dict(zip(available, [round(ri * 100, 1) for ri in relative_importance]))

    return {
        "r_squared": round(r_squared, 4),
        "betas": betas,
        "relative_importance": importance,
    }


def segment_analysis(df, segment_col):
    """Run driver analysis for each segment."""
    results = {}
    for segment_val in df[segment_col].unique():
        subset = df[df[segment_col] == segment_val]
        if len(subset) < 30:
            continue
        reg = run_regression(subset)
        results[segment_val] = {
            "n": len(subset),
            "avg_engagement": round(subset["overall_engagement"].mean(), 1),
            "top_drivers": dict(sorted(reg["relative_importance"].items(), key=lambda x: x[1], reverse=True)[:5]),
        }
    return results


def identify_focus_areas(correlations, regression_results, df):
    """Identify priority focus areas based on high impact + low score."""
    focus_areas = []
    for dim in DIMENSIONS:
        if dim not in df.columns:
            continue
        score = df[dim].mean()
        importance = regression_results["relative_importance"].get(dim, 0)
        correlation = correlations.get(dim, 0)

        # Priority = high importance + low score
        priority_score = importance * (100 - score) / 100
        gap_to_benchmark = round(score - 72, 1)  # Using 72 as a general benchmark

        focus_areas.append({
            "dimension": DIMENSION_LABELS.get(dim, dim),
            "dimension_key": dim,
            "current_score": round(score, 1),
            "correlation_with_engagement": correlation,
            "relative_importance_pct": importance,
            "priority_score": round(priority_score, 2),
            "gap_to_benchmark": gap_to_benchmark,
        })

    return sorted(focus_areas, key=lambda x: x["priority_score"], reverse=True)


def run_driver_analysis(pulse_path="data/pulse_survey_enriched.csv"):
    """Run full driver analysis pipeline."""
    print("Running engagement driver analysis...")

    df = pd.read_csv(pulse_path)
    available_dims = [d for d in DIMENSIONS if d in df.columns]
    print(f"  {len(df)} responses, {len(available_dims)} dimensions analyzed")

    # 1. Correlations
    correlations = compute_correlations(df)
    print(f"\n  Correlation with overall engagement:")
    for dim, corr in correlations.items():
        label = DIMENSION_LABELS.get(dim, dim)
        bar = "#" * int(abs(corr) * 40)
        print(f"    {label:30s} r={corr:+.3f}  {bar}")

    # 2. Regression
    reg = run_regression(df)
    print(f"\n  Regression model R² = {reg['r_squared']:.3f}")
    print(f"\n  Relative importance (% of variance explained):")
    sorted_imp = sorted(reg["relative_importance"].items(), key=lambda x: x[1], reverse=True)
    for dim, imp in sorted_imp:
        label = DIMENSION_LABELS.get(dim, dim)
        bar = "#" * int(imp * 2)
        print(f"    {label:30s} {imp:5.1f}%  {bar}")

    # 3. Focus areas
    focus = identify_focus_areas(correlations, reg, df)
    print(f"\n  Priority Focus Areas (high impact x low score):")
    for i, fa in enumerate(focus[:5], 1):
        print(f"    {i}. {fa['dimension']:30s} score={fa['current_score']:.0f}  importance={fa['relative_importance_pct']:.1f}%  priority={fa['priority_score']:.2f}")

    # 4. Segment analysis by department
    dept_drivers = segment_analysis(df, "department")
    print(f"\n  Department-level driver analysis:")
    for dept, info in sorted(dept_drivers.items(), key=lambda x: x[1]["avg_engagement"]):
        top_driver = list(info["top_drivers"].keys())[0]
        label = DIMENSION_LABELS.get(top_driver, top_driver)
        print(f"    {dept:25s} engagement={info['avg_engagement']:.0f}  #1 driver: {label}")

    # Save results
    focus_df = pd.DataFrame(focus)
    focus_df.to_csv("data/driver_analysis_results.csv", index=False)

    corr_df = pd.DataFrame([
        {"dimension": DIMENSION_LABELS.get(d, d), "correlation": c}
        for d, c in correlations.items()
    ])
    corr_df.to_csv("data/correlation_analysis.csv", index=False)

    dept_df = pd.DataFrame([
        {"department": dept, "avg_engagement": info["avg_engagement"],
         "n": info["n"], "top_driver": DIMENSION_LABELS.get(list(info["top_drivers"].keys())[0], "")}
        for dept, info in dept_drivers.items()
    ])
    dept_df.to_csv("data/department_drivers.csv", index=False)

    print(f"\n  Results saved to data/")
    return {"correlations": correlations, "regression": reg, "focus_areas": focus, "department_drivers": dept_drivers}


if __name__ == "__main__":
    run_driver_analysis()
