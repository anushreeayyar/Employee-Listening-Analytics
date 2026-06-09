"""
Synthetic Employee Survey Data Generator
=========================================
Generates realistic employee lifecycle survey data including:
- Demographic fields (department, tenure, level, location)
- Quantitative engagement scores across multiple dimensions
- Open-text responses with realistic employee feedback
- Multiple survey types (pulse, onboarding, exit)

Author: Anushree Ayyar
"""

import pandas as pd
import numpy as np
import random
import json
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

NUM_EMPLOYEES = 850
SURVEY_PERIOD = "Q1 2026"

DEPARTMENTS = {
    "Engineering": {"headcount": 320, "eng_mean": 74, "eng_std": 12},
    "Product Management": {"headcount": 65, "eng_mean": 80, "eng_std": 10},
    "Sales": {"headcount": 160, "eng_mean": 67, "eng_std": 14},
    "Marketing": {"headcount": 55, "eng_mean": 72, "eng_std": 11},
    "Customer Success": {"headcount": 95, "eng_mean": 76, "eng_std": 11},
    "People & Culture": {"headcount": 35, "eng_mean": 83, "eng_std": 8},
    "Finance": {"headcount": 45, "eng_mean": 69, "eng_std": 13},
    "Legal & Compliance": {"headcount": 25, "eng_mean": 71, "eng_std": 10},
    "IT & Infrastructure": {"headcount": 50, "eng_mean": 70, "eng_std": 12},
}

LEVELS = ["IC1", "IC2", "IC3", "IC4", "IC5", "Manager", "Sr. Manager", "Director", "VP"]
LEVEL_WEIGHTS = [0.05, 0.15, 0.25, 0.22, 0.13, 0.10, 0.05, 0.03, 0.02]

LOCATIONS = ["San Jose, CA", "Raleigh, NC", "Boulder, CO", "Remote - US", "Bangalore, India", "London, UK"]
LOCATION_WEIGHTS = [0.30, 0.20, 0.10, 0.20, 0.12, 0.08]

TENURE_BRACKETS = ["0-6 months", "6-12 months", "1-2 years", "2-5 years", "5-10 years", "10+ years"]
TENURE_WEIGHTS = [0.08, 0.07, 0.15, 0.30, 0.25, 0.15]

DIMENSIONS = [
    "manager_support", "growth_opportunities", "work_life_balance",
    "recognition", "team_collaboration", "purpose_mission",
    "psychological_safety", "tools_resources", "compensation_fairness",
    "dei_belonging"
]

# ─── OPEN-TEXT RESPONSE TEMPLATES ─────────────────────────────────────────────

POSITIVE_COMMENTS = {
    "manager_support": [
        "My manager is incredibly supportive and always makes time for 1:1s. I feel heard and valued.",
        "I appreciate how my manager advocates for my career development and gives me stretch opportunities.",
        "My skip-level really listens and follows through on feedback. Best manager I've had.",
        "Regular coaching conversations with my manager have helped me grow significantly this quarter.",
        "My manager creates a safe space where I can share concerns without fear of judgment.",
    ],
    "team_collaboration": [
        "Our team works really well together. Cross-functional projects have become much smoother.",
        "The collaboration culture here is what keeps me engaged. We genuinely support each other.",
        "Love how our team celebrates wins together and learns from setbacks without blame.",
        "Working across teams used to be painful but the new processes have made it much better.",
        "I feel like I belong on this team. People respect different perspectives and working styles.",
    ],
    "purpose_mission": [
        "I'm energized by the company's mission. I can see how my work directly impacts customers.",
        "Our leadership does a great job connecting daily work to the bigger picture.",
        "Knowing that our technology helps organizations manage their data better gives me real purpose.",
        "The company's focus on innovation and customer impact is what drew me here.",
    ],
    "psychological_safety": [
        "I feel comfortable sharing dissenting opinions in meetings without fear of retaliation.",
        "Our team norms have really improved the culture of open dialogue and trust.",
        "I can be myself at work and that makes a huge difference in my engagement.",
    ],
}

NEGATIVE_COMMENTS = {
    "growth_opportunities": [
        "I don't see a clear path for career advancement. Promotion criteria feel opaque and inconsistent.",
        "There aren't enough opportunities for internal mobility. I feel stuck in my current role.",
        "I've been in the same role for 3 years with no visibility into what's next for me.",
        "Career development feels like an afterthought. We need structured growth frameworks.",
        "The company talks about growth but doesn't invest in it. No training budget, no mentorship program.",
        "I've asked my manager about promotion multiple times but never get a clear answer on what I need to do.",
        "Watching external hires come in at higher levels while internal talent gets overlooked is demoralizing.",
    ],
    "recognition": [
        "Great work often goes unnoticed unless you're on a high-visibility project or in leadership's inner circle.",
        "We don't have a culture of recognition. Hitting targets is just expected, never celebrated.",
        "I consistently go above and beyond but rarely receive acknowledgment from leadership.",
        "Recognition is inconsistent across teams. Some managers are great at it, others never mention it.",
        "The annual awards feel performative. Day-to-day appreciation matters more and it's missing.",
        "I delivered a critical project under tight deadlines and didn't even get a thank you email.",
    ],
    "work_life_balance": [
        "The new hybrid policy feels like a step backward. Trust should go both ways.",
        "Workload has increased significantly but headcount hasn't. Something has to give.",
        "I'm regularly working evenings and weekends to meet unrealistic deadlines.",
        "Work-life balance is talked about but not practiced. There's an unspoken expectation to always be on.",
        "Meeting overload is killing productivity. I need focus time to actually do deep work.",
    ],
    "tools_resources": [
        "We spend too much time on workarounds for outdated internal tools.",
        "The tech stack is falling behind. We need investment in modern tooling to stay competitive.",
        "Our internal systems are clunky and slow. It impacts my daily productivity.",
        "I've flagged tooling issues for months. Nothing changes. It feels like shouting into the void.",
    ],
    "compensation_fairness": [
        "Compensation feels below market, especially compared to what competitors are offering.",
        "Pay equity across similar roles and levels seems inconsistent.",
        "The compensation review process lacks transparency. I don't understand how decisions are made.",
    ],
}

NEUTRAL_COMMENTS = [
    "Things are generally fine. Nothing extraordinary but nothing terrible either.",
    "The company is going through a lot of changes. I'm cautiously optimistic.",
    "Some things are improving, others still need work. It's a mixed bag.",
    "I like my team but have concerns about the broader organizational direction.",
    "Day-to-day work is good but I wish leadership communicated more clearly about strategy.",
]

EXIT_COMMENTS = [
    "Leaving primarily because of limited growth opportunities. I love the team but need to advance my career.",
    "Found a role that offers better compensation and a clearer path to leadership.",
    "The constant reorgs and shifting priorities made it hard to feel stable. Looking for more consistency.",
    "My manager was great but the lack of investment in my department's tools and resources became frustrating.",
    "I didn't feel recognized for my contributions. Moving to a company with a stronger appreciation culture.",
    "Work-life balance deteriorated over the past year. The new hybrid policy was the final straw.",
    "The company has a great mission but execution has been inconsistent. Need a fresh start.",
    "Compensation didn't keep up with the market. I was offered 25% more elsewhere for a similar role.",
]

ONBOARDING_COMMENTS = [
    "Onboarding was well-structured. My buddy was incredibly helpful in getting me up to speed.",
    "I felt welcomed from day one. My manager had a 30-60-90 plan ready which helped a lot.",
    "The onboarding materials were outdated. Some links didn't work and processes had changed.",
    "My role expectations were unclear in the first month. I wish there was more structure.",
    "Great onboarding experience overall. The team culture is warm and collaborative.",
    "It took me longer than expected to get access to all the tools I needed. IT onboarding needs improvement.",
    "My manager checked in regularly during my first 90 days which made a huge difference.",
    "The onboarding buddy program is a great concept but my buddy was too busy to actually help.",
]


def generate_employee_base():
    """Generate base employee records with demographics."""
    employees = []
    emp_id = 1001

    for dept, config in DEPARTMENTS.items():
        for _ in range(config["headcount"]):
            level = np.random.choice(LEVELS, p=LEVEL_WEIGHTS)
            tenure = np.random.choice(TENURE_BRACKETS, p=TENURE_WEIGHTS)
            location = np.random.choice(LOCATIONS, p=LOCATION_WEIGHTS)

            employees.append({
                "employee_id": f"EMP-{emp_id}",
                "department": dept,
                "level": level,
                "tenure": tenure,
                "location": location,
            })
            emp_id += 1

    return pd.DataFrame(employees)


def generate_engagement_scores(df):
    """Generate correlated engagement dimension scores per employee."""
    scores = []

    for _, row in df.iterrows():
        dept_config = DEPARTMENTS[row["department"]]
        base = np.random.normal(dept_config["eng_mean"], dept_config["eng_std"])

        # Tenure effect: newer employees slightly more positive (honeymoon)
        tenure_adj = {"0-6 months": 5, "6-12 months": 3, "1-2 years": 0,
                      "2-5 years": -1, "5-10 years": -2, "10+ years": -3}
        base += tenure_adj.get(row["tenure"], 0)

        # Level effect: managers/directors slightly higher engagement
        level_adj = {"IC1": -2, "IC2": -1, "IC3": 0, "IC4": 1, "IC5": 2,
                     "Manager": 3, "Sr. Manager": 4, "Director": 5, "VP": 6}
        base += level_adj.get(row["level"], 0)

        dim_scores = {}
        for dim in DIMENSIONS:
            noise = np.random.normal(0, 8)
            # Add dimension-specific effects
            dim_adj = 0
            if dim == "recognition" and row["department"] in ["Engineering", "Marketing"]:
                dim_adj = -8
            if dim == "growth_opportunities" and row["department"] in ["Sales", "Finance"]:
                dim_adj = -6
            if dim == "team_collaboration" and row["department"] in ["Engineering", "Customer Success"]:
                dim_adj = 5
            if dim == "manager_support" and row["department"] == "People & Culture":
                dim_adj = 6

            score = np.clip(base + noise + dim_adj, 1, 100)
            dim_scores[dim] = round(score)

        dim_scores["overall_engagement"] = round(np.mean(list(dim_scores.values())))
        scores.append(dim_scores)

    return pd.DataFrame(scores)


def generate_open_text(row):
    """Generate realistic open-text survey response based on scores."""
    comments = []
    overall = row["overall_engagement"]

    # Find strongest and weakest dimensions
    dim_scores = {d: row[d] for d in DIMENSIONS}
    weakest = min(dim_scores, key=dim_scores.get)
    strongest = max(dim_scores, key=dim_scores.get)

    # Add positive comment if strong area exists
    if dim_scores[strongest] >= 75 and strongest in POSITIVE_COMMENTS:
        comments.append(random.choice(POSITIVE_COMMENTS[strongest]))

    # Add negative comment if weak area exists
    if dim_scores[weakest] <= 60 and weakest in NEGATIVE_COMMENTS:
        comments.append(random.choice(NEGATIVE_COMMENTS[weakest]))

    # Sometimes add neutral comment
    if 60 < overall < 75 and random.random() < 0.3:
        comments.append(random.choice(NEUTRAL_COMMENTS))

    # Some employees skip open text
    if random.random() < 0.15:
        return ""

    return " ".join(comments) if comments else ""


def generate_pulse_survey(employees_df, scores_df):
    """Generate the main quarterly pulse survey dataset."""
    df = pd.concat([employees_df, scores_df], axis=1)

    # Simulate response rate (~85%)
    responded = df.sample(frac=0.85, random_state=42).copy()
    responded["survey_type"] = "pulse_q1_2026"
    responded["survey_date"] = "2026-03-15"

    # eNPS question (0-10)
    responded["enps_score"] = responded["overall_engagement"].apply(
        lambda x: np.clip(round(x / 10 + np.random.normal(0, 1)), 0, 10)
    )
    responded["enps_category"] = responded["enps_score"].apply(
        lambda x: "Promoter" if x >= 9 else ("Passive" if x >= 7 else "Detractor")
    )

    # Generate open-text
    responded["open_text_response"] = responded.apply(generate_open_text, axis=1)

    return responded


def generate_exit_surveys(employees_df):
    """Generate exit survey data for departed employees."""
    num_exits = 38
    exits = employees_df.sample(n=num_exits, random_state=99).copy()

    exits["survey_type"] = "exit"
    exits["survey_date"] = [
        (datetime(2026, 1, 1) + timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d")
        for _ in range(num_exits)
    ]
    exits["overall_satisfaction"] = np.clip(
        np.random.normal(52, 15, num_exits).astype(int), 10, 85
    )
    exits["would_recommend"] = exits["overall_satisfaction"].apply(
        lambda x: "Yes" if x > 65 else ("Maybe" if x > 45 else "No")
    )
    exits["primary_reason"] = np.random.choice(
        ["Growth/Career", "Compensation", "Work-Life Balance", "Management", "Company Direction", "Relocation"],
        size=num_exits, p=[0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
    )
    exits["exit_comment"] = [random.choice(EXIT_COMMENTS) for _ in range(num_exits)]

    return exits


def generate_onboarding_surveys(employees_df):
    """Generate 30-day and 90-day onboarding surveys for new hires."""
    new_hires = employees_df[employees_df["tenure"].isin(["0-6 months", "6-12 months"])].copy()
    num = len(new_hires)

    surveys = []
    for day_mark in [30, 90]:
        subset = new_hires.sample(frac=0.90, random_state=day_mark).copy()
        subset["survey_type"] = f"onboarding_{day_mark}day"
        subset["survey_date"] = [
            (datetime(2026, 1, 1) + timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d")
            for _ in range(len(subset))
        ]
        subset["role_clarity"] = np.clip(np.random.normal(72 if day_mark == 30 else 78, 14, len(subset)).astype(int), 10, 100)
        subset["manager_checkin_frequency"] = np.random.choice(
            ["Weekly", "Bi-weekly", "Monthly", "Rarely"], size=len(subset), p=[0.35, 0.30, 0.25, 0.10]
        )
        subset["ramp_confidence"] = np.clip(np.random.normal(68 if day_mark == 30 else 75, 12, len(subset)).astype(int), 10, 100)
        subset["onboarding_satisfaction"] = np.clip(np.random.normal(74, 13, len(subset)).astype(int), 10, 100)
        subset["onboarding_comment"] = [random.choice(ONBOARDING_COMMENTS) for _ in range(len(subset))]
        surveys.append(subset)

    return pd.concat(surveys, ignore_index=True)


def main():
    print("Generating synthetic employee survey data...")

    # Step 1: Create employee base
    employees = generate_employee_base()
    print(f"  Created {len(employees)} employee records across {len(DEPARTMENTS)} departments")

    # Step 2: Generate engagement scores
    scores = generate_engagement_scores(employees)

    # Step 3: Generate pulse survey
    pulse = generate_pulse_survey(employees, scores)
    print(f"  Pulse survey: {len(pulse)} responses ({len(pulse)/len(employees)*100:.0f}% response rate)")

    # Step 4: Generate exit surveys
    exits = generate_exit_surveys(employees)
    print(f"  Exit surveys: {len(exits)} responses")

    # Step 5: Generate onboarding surveys
    onboarding = generate_onboarding_surveys(employees)
    print(f"  Onboarding surveys: {len(onboarding)} responses")

    # Save datasets
    pulse.to_csv("data/pulse_survey_q1_2026.csv", index=False)
    exits.to_csv("data/exit_surveys_2026.csv", index=False)
    onboarding.to_csv("data/onboarding_surveys_2026.csv", index=False)
    employees.to_csv("data/employee_demographics.csv", index=False)

    print(f"\nDatasets saved to data/ directory:")
    print(f"  - pulse_survey_q1_2026.csv ({len(pulse)} rows)")
    print(f"  - exit_surveys_2026.csv ({len(exits)} rows)")
    print(f"  - onboarding_surveys_2026.csv ({len(onboarding)} rows)")
    print(f"  - employee_demographics.csv ({len(employees)} rows)")

    # Quick summary stats
    print(f"\nSummary Statistics:")
    print(f"  Overall Engagement Mean: {pulse['overall_engagement'].mean():.1f}")
    print(f"  eNPS: {((pulse['enps_category']=='Promoter').mean() - (pulse['enps_category']=='Detractor').mean()) * 100:.0f}")
    print(f"  Open-text response rate: {(pulse['open_text_response'] != '').mean() * 100:.0f}%")
    print(f"  Exit primary reasons: {exits['primary_reason'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
