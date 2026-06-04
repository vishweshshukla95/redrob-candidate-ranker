#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Ranker
Ranks 100,000 candidates for Senior AI Engineer role at Redrob AI.
Usage: python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import json
import csv
import argparse
from datetime import datetime, date
from tqdm import tqdm

# ─────────────────────────────────────────────
# 1.  JD CONSTANTS  (tuned to the actual JD)
# ─────────────────────────────────────────────

# Skills the JD says are REQUIRED — heavy weight
REQUIRED_SKILLS = {
    "embeddings", "sentence-transformers", "vector database", "vector db",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "hybrid search", "retrieval", "ranking", "python",
    "ndcg", "mrr", "map", "evaluation", "a/b testing", "information retrieval",
    "semantic search", "dense retrieval", "bm25", "reranking", "re-ranking",
    "rag", "retrieval augmented generation", "bgе", "e5", "openai embeddings",
}

# Skills that are NICE TO HAVE — lower weight
NICE_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "finetuning", "llm", "llms",
    "learning to rank", "xgboost", "distributed systems", "inference optimization",
    "open source", "nlp", "transformers", "pytorch", "tensorflow",
    "recommendation system", "search", "machine learning", "ml",
}

# Skills that are DISQUALIFIERS — penalize heavily
DISQUALIFIER_SKILLS = {
    "computer vision", "speech recognition", "robotics", "tts",
    "image classification", "object detection", "pose estimation",
    "asr", "ocr", "image segmentation",
}

# Title keywords that signal a GOOD fit
GOOD_TITLE_KEYWORDS = [
    "ai engineer", "ml engineer", "machine learning engineer",
    "search engineer", "nlp engineer", "applied scientist",
    "data scientist", "senior engineer", "staff engineer",
    "ranking engineer", "retrieval engineer", "research engineer",
]

# Title keywords that signal a BAD fit
BAD_TITLE_KEYWORDS = [
    "marketing", "sales", "hr", "recruiter", "finance", "accountant",
    "designer", "product manager", "project manager", "business analyst",
    "computer vision", "robotics", "speech", "customer support",
    "customer success", "content writer", "copywriter", "social media",
    "seo", "growth", "operations", "supply chain", "legal", "attorney",
    "teacher", "educator", "consultant", "analyst",
]

# Companies that are PURE consulting — penalize
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
}

# Preferred India locations
PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "ncr", "gurugram", "gurgaon",
    "hyderabad", "mumbai", "bangalore", "bengaluru",
}

# ─────────────────────────────────────────────
# 2.  SCORING FUNCTIONS
# ─────────────────────────────────────────────

def score_skills(candidate):
    """
    Score based on skills match with JD.
    - Required skills: +3 pts each (with proficiency bonus)
    - Nice-to-have skills: +1 pt each
    - Disqualifier skills: -4 pts each
    - Honeypot detection: penalize expert skills with 0 duration
    Returns: float 0.0 – 1.0
    """
    skills = candidate.get("skills", [])
    score = 0.0
    honeypot_penalty = 0.0

    proficiency_mult = {"beginner": 0.5, "intermediate": 0.8, "advanced": 1.0, "expert": 1.2}

    for skill in skills:
        name = skill.get("name", "").lower().strip()
        proficiency = skill.get("proficiency", "intermediate")
        duration = skill.get("duration_months", 0) or 0
        endorsements = skill.get("endorsements", 0) or 0
        mult = proficiency_mult.get(proficiency, 0.8)

        # Honeypot detection: expert with 0 duration and 0 endorsements is suspicious
        if proficiency == "expert" and duration == 0 and endorsements == 0:
            honeypot_penalty += 2.0

        # Check required skills
        matched_required = any(req in name for req in REQUIRED_SKILLS)
        if matched_required:
            score += 3.0 * mult
            # Bonus for duration of use
            if duration >= 24:
                score += 1.0
            elif duration >= 12:
                score += 0.5

        # Check nice-to-have skills
        matched_nice = any(nice in name for nice in NICE_SKILLS)
        if matched_nice and not matched_required:
            score += 1.0 * mult

        # Penalize disqualifier skills
        matched_disq = any(disq in name for disq in DISQUALIFIER_SKILLS)
        if matched_disq:
            score -= 4.0

    # Apply honeypot penalty
    score -= honeypot_penalty

    # Also check skill assessment scores
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    for skill_name, skill_score in assessment_scores.items():
        if any(req in skill_name.lower() for req in REQUIRED_SKILLS):
            score += (skill_score / 100) * 2.0

    # Normalize to 0-1 (cap at reasonable max)
    return max(0.0, min(1.0, score / 35.0))


def score_career(candidate):
    """
    Score based on career history — looks at actual work done, not just titles.
    - Product company experience: bonus
    - Relevant descriptions (shipping ranking/retrieval systems): bonus
    - Pure consulting career: penalty
    - Research-only background: penalty
    - Title match: bonus/penalty
    Returns: float 0.0 – 1.0
    """
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    score = 0.0

    current_title = profile.get("current_title", "").lower()
    years_exp = profile.get("years_of_experience", 0) or 0

    # Years of experience scoring (JD wants 5-9 years)
    if 5 <= years_exp <= 9:
        score += 3.0
    elif 4 <= years_exp <= 11:
        score += 2.0
    elif 3 <= years_exp <= 13:
        score += 1.0
    else:
        score += 0.0

    # Current title scoring
    if any(kw in current_title for kw in GOOD_TITLE_KEYWORDS):
        score += 2.0
    if any(kw in current_title for kw in BAD_TITLE_KEYWORDS):
        score -= 10.0

    # Career history analysis
    consulting_count = 0
    product_company_count = 0
    relevant_description_score = 0.0

    # Keywords that show real ML/AI work in descriptions
    strong_keywords = [
        "embedding", "vector", "retrieval", "ranking", "search",
        "recommendation", "nlp", "fine-tun", "deploy", "production",
        "a/b test", "evaluation", "index", "latency", "inference",
    ]

    for job in career:
        company = job.get("company", "").lower()
        title = job.get("title", "").lower()
        description = job.get("description", "").lower()
        industry = job.get("industry", "").lower()
        duration = job.get("duration_months", 0) or 0
        company_size = job.get("company_size", "")

        # Check for consulting company
        is_consulting = any(c in company for c in CONSULTING_COMPANIES)
        if is_consulting:
            consulting_count += 1
        else:
            product_company_count += 1

        # Reward for relevant work in descriptions
        for kw in strong_keywords:
            if kw in description:
                # Weight by duration (longer = more experience)
                weight = min(duration / 24.0, 2.0)
                relevant_description_score += 0.5 * weight

        # Reward for title match in career
        if any(kw in title for kw in GOOD_TITLE_KEYWORDS):
            score += 1.0
        if any(kw in title for kw in BAD_TITLE_KEYWORDS):
            score -= 2.0

        # Penalize if purely research (academic) role
        if "research" in title and "engineer" not in title and "applied" not in title:
            score -= 1.0

    # Penalize pure consulting careers (all jobs at consulting firms)
    if consulting_count > 0 and product_company_count == 0:
        score -= 6.0
    elif consulting_count > 0:
        score -= consulting_count * 1.0

    # Add relevant description score
    score += min(relevant_description_score, 4.0)

    # Normalize to 0-1
    return max(0.0, min(1.0, score / 18.0))


def score_location(candidate):
    """
    Score based on location preference.
    JD prefers: Pune, Noida, Delhi NCR, Hyderabad, Mumbai, Bangalore
    Returns: float 0.0 – 1.0
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)

    # Must be India or willing to relocate (JD says no visa sponsorship)
    if country == "india":
        if any(city in location for city in PREFERRED_LOCATIONS):
            return 1.0
        else:
            # India but not preferred city
            if willing_to_relocate:
                return 0.7
            else:
                return 0.4
    else:
        # Outside India
        if willing_to_relocate:
            return 0.3
        else:
            return 0.0


def score_education(candidate):
    """
    Score based on education quality.
    Returns: float 0.0 – 1.0
    """
    education = candidate.get("education", [])
    if not education:
        return 0.3  # No education data — neutral

    best_score = 0.0
    tier_scores = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.25, "unknown": 0.3}

    for edu in education:
        tier = edu.get("tier", "unknown")
        field = edu.get("field_of_study", "").lower()
        degree = edu.get("degree", "").lower()

        tier_score = tier_scores.get(tier, 0.3)

        # Bonus for relevant field
        if any(f in field for f in ["computer science", "cs", "ai", "machine learning",
                                     "data science", "statistics", "mathematics",
                                     "information technology", "software"]):
            tier_score = min(1.0, tier_score + 0.1)

        # Bonus for higher degree
        if any(d in degree for d in ["phd", "ph.d", "m.tech", "m.s.", "ms", "mtech", "masters"]):
            tier_score = min(1.0, tier_score + 0.1)

        best_score = max(best_score, tier_score)

    return best_score


def score_behavioral_signals(candidate):
    """
    Score based on Redrob platform behavioral signals.
    Key signals: recency of activity, open to work, response rate,
    notice period, github activity, profile completeness.
    Returns: float 0.0 – 1.0
    """
    signals = candidate.get("redrob_signals", {})
    score = 0.0

    # 1. Open to work flag — very important
    if signals.get("open_to_work_flag", False):
        score += 2.0

    # 2. Last active date — recency matters a lot
    last_active_str = signals.get("last_active_date", "")
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            today = date.today()
            days_inactive = (today - last_active).days
            if days_inactive <= 7:
                score += 3.0
            elif days_inactive <= 30:
                score += 2.5
            elif days_inactive <= 90:
                score += 1.5
            elif days_inactive <= 180:
                score += 0.5
            else:
                score -= 1.0  # Inactive for 6+ months = not really available
        except (ValueError, TypeError):
            pass

    # 3. Recruiter response rate
    response_rate = signals.get("recruiter_response_rate", 0) or 0
    score += response_rate * 2.0

    # 4. Notice period (JD prefers sub-30 days)
    notice_period = signals.get("notice_period_days", 90) or 90
    if notice_period <= 15:
        score += 2.0
    elif notice_period <= 30:
        score += 1.5
    elif notice_period <= 60:
        score += 0.5
    elif notice_period > 90:
        score -= 0.5

    # 5. GitHub activity score (important for an AI Engineer role)
    github_score = signals.get("github_activity_score", -1)
    if github_score >= 0:  # -1 means no GitHub linked
        score += (github_score / 100) * 2.0

    # 6. Profile completeness
    completeness = signals.get("profile_completeness_score", 0) or 0
    score += (completeness / 100) * 1.0

    # 7. Interview completion rate (reliability signal)
    interview_rate = signals.get("interview_completion_rate", 0) or 0
    score += interview_rate * 1.0

    # 8. Saved by recruiters recently (social proof)
    saved = signals.get("saved_by_recruiters_30d", 0) or 0
    score += min(saved / 10.0, 1.0)

    # 9. Verified identity (trust signal)
    if signals.get("verified_email", False):
        score += 0.3
    if signals.get("verified_phone", False):
        score += 0.3

    # Normalize to 0-1
    return max(0.0, min(1.0, score / 15.0))


def is_honeypot(candidate):
    """
    Detect honeypot candidates with impossible/suspicious profiles.
    Returns: True if likely honeypot
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # Check 1: expert skills with zero duration and zero endorsements (multiple)
    suspicious_skills = 0
    for skill in skills:
        if (skill.get("proficiency") == "expert" and
                (skill.get("duration_months") or 0) == 0 and
                (skill.get("endorsements") or 0) == 0):
            suspicious_skills += 1
    if suspicious_skills >= 3:
        return True

    # Check 2: years at company that didn't exist that long
    years_exp = profile.get("years_of_experience", 0) or 0
    for job in career:
        start_date_str = job.get("start_date", "")
        if start_date_str:
            try:
                start_year = int(start_date_str[:4])
                current_year = date.today().year
                company_age = current_year - start_year
                duration_years = (job.get("duration_months") or 0) / 12
                # If duration claimed is way more than company could exist
                if duration_years > company_age + 2:
                    return True
            except (ValueError, TypeError):
                pass

    # Check 3: impossible years of experience (e.g., started work before age 10)
    if years_exp > 35:
        return True

    return False


def compute_total_score(candidate):
    """
    Combine all component scores into a final weighted score.
    Weights tuned to match JD priorities.
    """
    # Skip honeypots entirely
    if is_honeypot(candidate):
        return 0.0

    skills_score    = score_skills(candidate)
    career_score    = score_career(candidate)
    location_score  = score_location(candidate)
    education_score = score_education(candidate)
    signals_score   = score_behavioral_signals(candidate)

    # Weighted combination (must sum to 1.0)
    total = (
        0.30 * skills_score    +
        0.30 * career_score    +
        0.15 * signals_score   +
        0.15 * location_score  +
        0.10 * education_score
    )

    return round(total, 6)


# ─────────────────────────────────────────────
# 3.  REASONING GENERATOR
# ─────────────────────────────────────────────

def generate_reasoning(candidate, rank):
    """
    Generate a 1-2 sentence reasoning for each candidate.
    Uses actual profile data — no hallucination.
    """
    profile   = candidate.get("profile", {})
    signals   = candidate.get("redrob_signals", {})
    skills    = candidate.get("skills", [])
    career    = candidate.get("career_history", [])

    title     = profile.get("current_title", "Unknown")
    company   = profile.get("current_company", "Unknown")
    years     = profile.get("years_of_experience", 0)
    location  = profile.get("location", "Unknown")
    country   = profile.get("country", "")

    # Top skills (required ones that match)
    top_skills = []
    for s in skills:
        name = s.get("name", "").lower()
        if any(req in name for req in REQUIRED_SKILLS):
            top_skills.append(s.get("name"))
        if len(top_skills) >= 3:
            break

    # Behavioral highlights
    open_to_work = signals.get("open_to_work_flag", False)
    notice       = signals.get("notice_period_days", None)
    github       = signals.get("github_activity_score", -1)
    response_r   = signals.get("recruiter_response_rate", 0)
    last_active  = signals.get("last_active_date", "")

    # Build reasoning
    part1 = f"{title} at {company} with {years:.0f} years of experience"
    if location:
        loc_str = location if not country or country.lower() == "india" else f"{location}, {country}"
        part1 += f", based in {loc_str}"
    part1 += "."

    highlights = []
    if top_skills:
        highlights.append(f"relevant skills include {', '.join(top_skills)}")
    if open_to_work:
        highlights.append("actively open to work")
    if notice is not None and notice <= 30:
        highlights.append(f"short notice period ({notice} days)")
    if github >= 50:
        highlights.append(f"strong GitHub activity (score: {github:.0f})")
    if response_r >= 0.7:
        highlights.append(f"high recruiter response rate ({response_r:.0%})")

    # Add concerns for lower ranks
    concerns = []
    if rank > 50:
        if years < 4 or years > 12:
            concerns.append(f"experience ({years:.0f} yrs) outside preferred 5-9 yr range")
        last_active_days = 999
        if last_active:
            try:
                la = datetime.strptime(last_active, "%Y-%m-%d").date()
                last_active_days = (date.today() - la).days
            except (ValueError, TypeError):
                pass
        if last_active_days > 180:
            concerns.append("inactive for 6+ months")

    part2 = ""
    if highlights:
        part2 = "Positives: " + "; ".join(highlights) + "."
    if concerns:
        part2 += (" Concerns: " + "; ".join(concerns) + ".") if part2 else ("Concerns: " + "; ".join(concerns) + ".")

    reasoning = (part1 + " " + part2).strip()

    # Clean up for CSV (remove quotes and newlines)
    reasoning = reasoning.replace('"', "'").replace('\n', ' ').replace('\r', '')
    return reasoning


# ─────────────────────────────────────────────
# 4.  MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=100, help="Number of top candidates to output")
    args = parser.parse_args()

    print(f"Loading candidates from: {args.candidates}")
    candidates = []

    # Support both .jsonl and .jsonl.gz
    if args.candidates.endswith(".gz"):
        import gzip
        open_fn = lambda: gzip.open(args.candidates, "rt", encoding="utf-8")
    else:
        open_fn = lambda: open(args.candidates, "r", encoding="utf-8")

    with open_fn() as f:
        for line in tqdm(f, desc="Reading candidates", unit=" candidates"):
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"Loaded {len(candidates):,} candidates")
    print("Scoring candidates...")

    # Score all candidates
    scored = []
    for cand in tqdm(candidates, desc="Scoring", unit=" candidates"):
        cid   = cand.get("candidate_id", "")
        score = compute_total_score(cand)
        scored.append((cid, score, cand))

    # Sort by score descending
    scored.sort(key=lambda x: (-x[1], x[0]))

    # Take top N
    top_candidates = scored[:args.top_n]

    print(f"Writing top {args.top_n} candidates to: {args.out}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, (cid, score, cand) in enumerate(top_candidates, start=1):
            reasoning = generate_reasoning(cand, rank)
            writer.writerow([cid, rank, f"{score:.6f}", reasoning])

    print(f"\nDone! Submission saved to: {args.out}")
    print(f"Top 5 candidates:")
    for rank, (cid, score, cand) in enumerate(top_candidates[:5], start=1):
        title = cand.get("profile", {}).get("current_title", "N/A")
        print(f"  #{rank}: {cid} | score={score:.4f} | {title}")


if __name__ == "__main__":
    main()