#!/usr/bin/env python3
"""
build.py — Enriches submission.csv with candidate profile data
so the website can display names, titles, skills, locations etc.

Usage: python build.py
Output: submission_enriched.csv  (used by index.html)
"""

import json
import csv
from tqdm import tqdm

CANDIDATES_FILE = "candidates.jsonl"
SUBMISSION_FILE = "submission.csv"
OUTPUT_FILE     = "submission_enriched.csv"

# Required skills to highlight in the website
REQUIRED_SKILLS = {
    "embeddings", "sentence-transformers", "vector database", "vector db",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "hybrid search", "retrieval", "ranking", "python",
    "rag", "semantic search", "dense retrieval", "bm25", "reranking",
}

def get_top_skills(skills):
    """Return top matching skills as pipe-separated string."""
    matched = []
    for s in skills:
        name = s.get("name", "")
        if any(req in name.lower() for req in REQUIRED_SKILLS):
            matched.append(name)
        if len(matched) >= 4:
            break
    # If no required skills found, just return top 3 skills
    if not matched:
        matched = [s.get("name","") for s in skills[:3]]
    return "|".join(matched)

def score_skills_component(candidate):
    """Quick skills score for display bars."""
    skills = candidate.get("skills", [])
    score = 0.0
    for s in skills:
        name = s.get("name", "").lower()
        if any(req in name for req in REQUIRED_SKILLS):
            score += 1.0
    return min(1.0, score / 8.0)

def score_career_component(candidate):
    """Quick career score for display bars."""
    years = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    if 5 <= years <= 9:
        return 1.0
    elif 4 <= years <= 11:
        return 0.75
    elif 3 <= years <= 13:
        return 0.5
    return 0.25

def score_signals_component(candidate):
    """Quick signals score for display bars."""
    signals = candidate.get("redrob_signals", {})
    score = 0.0
    if signals.get("open_to_work_flag"): score += 0.4
    rr = signals.get("recruiter_response_rate", 0) or 0
    score += rr * 0.3
    gh = signals.get("github_activity_score", -1)
    if gh >= 0: score += (gh / 100) * 0.3
    return min(1.0, score)

def score_location_component(candidate):
    """Quick location score for display bars."""
    profile  = candidate.get("profile", {})
    signals  = candidate.get("redrob_signals", {})
    location = profile.get("location", "").lower()
    country  = profile.get("country", "").lower()
    relocate = signals.get("willing_to_relocate", False)
    preferred = {"pune","noida","delhi","ncr","gurugram","gurgaon","hyderabad","mumbai","bangalore","bengaluru"}
    if country == "india":
        if any(c in location for c in preferred): return 1.0
        return 0.7 if relocate else 0.4
    return 0.3 if relocate else 0.0

def main():
    print("Loading candidates...")
    candidate_map = {}
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading", unit=" candidates"):
            line = line.strip()
            if line:
                try:
                    c = json.loads(line)
                    candidate_map[c["candidate_id"]] = c
                except:
                    continue

    print(f"Loaded {len(candidate_map):,} candidates")
    print("Reading submission...")

    with open(SUBMISSION_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Enriching {len(rows)} submission rows...")

    enriched = []
    for row in rows:
        cid = row["candidate_id"]
        c   = candidate_map.get(cid, {})
        p   = c.get("profile", {})
        sig = c.get("redrob_signals", {})
        skills = c.get("skills", [])

        enriched.append({
            "candidate_id":       cid,
            "rank":               row["rank"],
            "score":              row["score"],
            "reasoning":          row.get("reasoning", ""),
            "name":               p.get("anonymized_name", ""),
            "current_title":      p.get("current_title", ""),
            "current_company":    p.get("current_company", ""),
            "location":           p.get("location", ""),
            "country":            p.get("country", ""),
            "years_of_experience":p.get("years_of_experience", ""),
            "open_to_work":       str(sig.get("open_to_work_flag", False)).lower(),
            "notice_period_days": sig.get("notice_period_days", ""),
            "github_activity_score": sig.get("github_activity_score", -1),
            "recruiter_response_rate": sig.get("recruiter_response_rate", ""),
            "top_skills":         get_top_skills(skills),
            "skills_score":       round(score_skills_component(c), 4),
            "career_score":       round(score_career_component(c), 4),
            "signals_score":      round(score_signals_component(c), 4),
            "location_score":     round(score_location_component(c), 4),
        })

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(enriched[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(enriched)

    print(f"\nDone! Enriched file saved as: {OUTPUT_FILE}")
    print(f"Open index.html in your browser to see the dashboard.")

if __name__ == "__main__":
    main()
