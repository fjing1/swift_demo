"""Scraper for Google's own careers site (careers.google.com), free/direct.

jobspy doesn't cover Google's career page (it scrapes Google *search* results
instead), so job postings straight from Google/DeepMind/Waymo/etc. never show
up via job_scraper.py. This module hits the public, unauthenticated HTML the
careers site serves (no API key, no headless browser needed) and pulls the
job data out of the embedded `AF_initDataCallback({key: 'ds:1', ...})` blob,
which is the same JSON the page's JS would otherwise render client-side.

Run directly:
    python google_jobs.py "site reliability engineer" --location "Austin, TX"
"""
import argparse
import json
import re

import pandas as pd
import requests

SEARCH_URL = "https://www.google.com/about/careers/applications/jobs/results/"
USER_AGENT = "Mozilla/5.0 (compatible; job-scan/1.0)"


def _extract_ds_block(html, key):
    """Pull the JSON payload out of `AF_initDataCallback({key: '<key>', ..., data: [...]})`."""
    marker = f"key: '{key}'"
    idx = html.find(marker)
    if idx == -1:
        return None
    start = html.find("data:", idx)
    if start == -1:
        return None
    start += len("data:")
    depth = 0
    end = None
    for i in range(start, len(html)):
        c = html[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    return json.loads(html[start:end])


def _job_to_row(job, query, location):
    job_id, title, apply_link = job[0], job[1], job[2]
    company = job[7] if len(job) > 7 else "Google"
    locations = job[9] if len(job) > 9 else None
    location_str = ", ".join(loc[0] for loc in locations) if locations else ""
    description = ""
    if len(job) > 3 and isinstance(job[3], list) and len(job[3]) > 1:
        description = re.sub(r"<[^>]+>", " ", job[3][1] or "").strip()
    return {
        "site": "google_careers",
        "job_url": apply_link,
        "applyLink": apply_link,
        "title": title,
        "company": company,
        "location": location_str,
        "job_type": "fulltime",
        "description": description,
        "query": query,
        "search_location": location,
        "id": f"google_{job_id}",
    }


def scrape_google_jobs(query, location=None, max_pages=3):
    """Search careers.google.com for `query` (optionally near `location`).

    Returns a tidy DataFrame in the same shape job_scraper.py produces, so
    results can be concatenated straight into the shared local jobs store.
    """
    rows = []
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    for page in range(1, max_pages + 1):
        params = {"q": query}
        if location:
            params["location"] = location
        if page > 1:
            params["page"] = page

        resp = session.get(SEARCH_URL, params=params, timeout=20)
        resp.raise_for_status()

        jobs = _extract_ds_block(resp.text, "ds:1")
        if not jobs or not jobs[0]:
            break

        rows.extend(_job_to_row(job, query, location) for job in jobs[0])

        if len(jobs[0]) < 20:
            break  # short of a full page -> no more results

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search careers.google.com directly.")
    parser.add_argument("query", help='Search term, e.g. "site reliability engineer".')
    parser.add_argument("--location", help='e.g. "Austin, TX" or "United States".')
    parser.add_argument("--max-pages", type=int, default=3)
    cli = parser.parse_args()

    df = scrape_google_jobs(cli.query, location=cli.location, max_pages=cli.max_pages)
    print(f"Found {len(df)} jobs")
    for _, row in df.iterrows():
        print(f"- {row['title']} ({row['location']}) -> {row['job_url']}")
