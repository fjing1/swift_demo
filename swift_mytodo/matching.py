"""Matching ETL (free stack).

  - get job + resume data from the local document store (was: OpenSearch)
  - rank the best job matches per resume with a local LLM RAG chain (was: GPT-3.5)
  - email the recommendations to each job seeker via plain SMTP (was: SendGrid)

Trigger: run on a schedule, or after resumes are preprocessed.
    python matching.py
"""
import os
import re
import ast
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader

import utils

JOBS_INDEX = "jobs"
RESUMES_INDEX = "resumes"


def get_latest_jobs(limit=100):
    """Get the latest job data from the local store."""
    jobs_df = utils.load_documents(JOBS_INDEX)
    if jobs_df.empty:
        return jobs_df
    latest_jobs_df = jobs_df.head(limit)
    print(f'Got latest {len(latest_jobs_df)} jobs from local store.')
    return latest_jobs_df


def extract_email_regex(list_of_strings):
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    for string in list_of_strings:
        if re.search(email_pattern, string):
            return re.search(email_pattern, string).group()


def matching(loader, resume_summary):
    """
    input:
    loader - loader for the job data (CSVLoader).
    resume_summary: list of summary lines for one resume.

    Given each resume_summary, provide job recommendations via the LLM.

    output:
    (resume_email, job_list)
    """
    resume_summary_formatted = "\n".join(resume_summary)
    print(f"--------------------Formatted Resume Summary:--------------------\n {resume_summary_formatted} \n ************************")

    rag_chain = utils.rag_invoke(loader)

    joblink = rag_chain.invoke("  Given a summary from a resume represented by" + resume_summary_formatted +
    '''identify and rank the TOP 3 job postings that are the best match based on the following criteria:
Exact Match on Job Title: The job title in the job posting must match 100% with the job title or role being sought as mentioned in the {resume_summary}. This is a critical requirement, and only job postings with a title that exactly matches should be considered for the TOP 3.
Recency of the Job Posting: Preference is given to the most recent job listings found in their metadata.
Similarity in Job Skills: The job listing must have skill requirements that closely match the skills listed in the {resume_summary}.
Matching Desired Job Location: The location mentioned in the job posting should match the desired job location specified in the {resume_summary}.
For each of the TOP 3 matches, structure the output as follows, providing the job title and the company name, followed by the phrase "Apply on:" and then the direct link to the job application or job posting (applyLink):
1. Job Title at Company Name - Apply on: [link]
2. Job Title at Company Name - Apply on: [link]
3. Job Title at Company Name - Apply on: [link] '''
    )
    job_list = joblink.split('\n')
    print(f'\nCHECK {len(job_list)} job_list: \n{job_list}\n')

    resume_email = extract_email_regex(resume_summary)
    print(f'\nCHECK resume_email: {resume_email}\n')
    return resume_email, job_list


def send_job_recommendations(to_email, job_links, smtp_user, smtp_password,
                             smtp_host="smtp.gmail.com", smtp_port=587):
    """Send recommendations over plain SMTP (free; e.g. Gmail app password).

    Set smtp_user / smtp_password in credential.txt. For Gmail, create an
    app password at https://myaccount.google.com/apppasswords.
    """
    if not to_email:
        print("No recipient email found in resume summary; skipping send.")
        return
    if not smtp_user or not smtp_password:
        print("SMTP credentials missing; skipping send. "
              "Add smtp_user / smtp_password to credential.txt.")
        return

    job_links_html = job_links.replace('\n', '<br>')  # HTML line breaks
    html_content = (
        "<p>Hi there,</p>"
        "<p>We've found some job opportunities that match your resume well:</p>"
        "<ul>" + job_links_html + "</ul>"
        "<p>Best of luck with your job search!</p>"
        "<p>Best regards,</p>"
        "<p>SWIFTHIRE</p>"
    )

    message = MIMEMultipart('alternative')
    message['Subject'] = 'SWIFTHIRE Job Recommendations for You'
    message['From'] = smtp_user
    message['To'] = to_email
    message.attach(MIMEText(html_content, 'html'))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [to_email], message.as_string())
        print(f"\n\n---------------------Email sent to {to_email}!-----------------------\n")
    except Exception as e:
        print(f"An error occurred: {e}")


def _as_summary_list(value):
    """Stored summaries round-trip through CSV as a string; parse back to a list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        return [value]
    return [str(value)]


if __name__ == "__main__":
    # get job data
    latest_jobs_df = get_latest_jobs()
    os.makedirs('./data/placeholder_resume_job', exist_ok=True)
    latest_jobs_df.to_csv('./data/placeholder_resume_job/latest_jobs_df.csv', index=False)

    # get resume summaries (written by resume_summary.py)
    all_resume_df = utils.load_documents(RESUMES_INDEX)

    # run LLM matching between resume and jobs
    job_loader = CSVLoader(file_path='./data/placeholder_resume_job/latest_jobs_df.csv')

    smtp_user = utils.load_specific_api_key(key_name='smtp_user')
    smtp_password = utils.load_specific_api_key(key_name='smtp_password')

    for raw_summary in all_resume_df['summary'].tolist():
        resume_summary = _as_summary_list(raw_summary)
        resume_email, joblinks = matching(job_loader, resume_summary)
        joblinks_clickable = '\n'.join(joblinks)
        send_job_recommendations(resume_email, joblinks_clickable, smtp_user, smtp_password)
