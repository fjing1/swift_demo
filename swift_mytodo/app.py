"""Chalice scheduled wrapper around the free job scraper.

The actual scraping/storage logic lives in job_scraper.py (free stack: JobSpy +
local document store). This file just wraps it in a Chalice schedule if you want
to deploy it to AWS Lambda. For a fully free / local setup, prefer running
`python job_scraper.py` from cron instead of deploying this Lambda.
"""
from chalice import Chalice, Rate

from job_scraper import scrape_all

app = Chalice(app_name='mytodo')
app.debug = True


@app.schedule(Rate(24, unit=Rate.HOURS))
def scrape_jobs_daily(event):
    scrape_all()
