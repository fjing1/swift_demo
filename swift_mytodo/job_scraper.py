"""Job scraper (free stack).

Replaces the paid Apify "google-jobs-scraper" actor with the free, open-source
python-jobspy library (https://github.com/Bunsly/JobSpy), which scrapes Indeed,
Google, LinkedIn, Glassdoor and ZipRecruiter, plus google_jobs.py which hits
Google's own careers site directly (jobspy's "google" site scrapes Google
*search* results, not careers.google.com, so Google/DeepMind/Waymo postings
were otherwise missing). Results are written to the local document store
instead of AWS OpenSearch.

Run directly (recommended free path, e.g. from cron):
    python job_scraper.py
"""
import os

import pandas as pd
from jobspy import scrape_jobs

import google_jobs
import utils

JOB_TITLES = ["Software Engineer", "Site Reliability Engineer"]
LOCATIONS = ["Austin, TX", "Remote"]

# Which job boards to scrape (comma-separated env override).
SITES = os.environ.get("JOBSPY_SITES", "indeed,google").split(",")
RESULTS_WANTED = int(os.environ.get("JOBSPY_RESULTS", "20"))
HOURS_OLD = int(os.environ.get("JOBSPY_HOURS_OLD", "72"))
COUNTRY = os.environ.get("JOBSPY_COUNTRY", "USA")

JOBS_INDEX = "jobs"

# Google's careers site has no "Remote" location filter; search US-wide instead
# and rely on match scoring/description to surface remote-friendly postings.
GOOGLE_LOCATIONS = {"Remote": "United States"}


def get_google_job_data(job_title, location):
    """Search careers.google.com directly and return a tidy DataFrame."""
    df = google_jobs.scrape_google_jobs(
        job_title, location=GOOGLE_LOCATIONS.get(location, location)
    )
    if df.empty:
        return df
    df["query"] = job_title
    df["search_location"] = location
    return df


def get_job_data(job_title, location):
    """Scrape jobs for one title/location and return a tidy DataFrame."""
    df = scrape_jobs(
        site_name=SITES,
        search_term=job_title,
        google_search_term=f"{job_title} jobs near {location}",
        location=location,
        results_wanted=RESULTS_WANTED,
        hours_old=HOURS_OLD,
        country_indeed=COUNTRY,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["query"] = job_title
    df["search_location"] = location
    # Expose the listing URL as `applyLink` so the matching prompt can cite it.
    if "job_url" in df.columns:
        df["applyLink"] = df["job_url"]
    if "job_type" in df.columns:
        job_type = df["job_type"].fillna("").str.lower()
        is_fulltime_permanent = job_type.str.contains("fulltime") & ~job_type.str.contains(
            "contract|temporary|parttime", regex=True
        )
        df = df[is_fulltime_permanent]
    return df


def scrape_all():
    """Scrape every title x location combination and save to the local store."""
    position_df = pd.DataFrame()

    for location in LOCATIONS:
        print(location)
        for jt in JOB_TITLES:
            print(jt)
            d = get_job_data(jt, location)
            print(f"  -> {len(d)} jobs")
            position_df = pd.concat([position_df, d], ignore_index=True)

            g = get_google_job_data(jt, location)
            print(f"  -> {len(g)} Google jobs")
            position_df = pd.concat([position_df, g], ignore_index=True)
        print("=" * 30)

    print("Saving scraped jobs to local store")
    utils.save_documents(JOBS_INDEX, position_df, dedupe_on="applyLink")
    return position_df


if __name__ == "__main__":
    scrape_all()
