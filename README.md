# Smart Hire Search
large language model based project for a client improve current hiring process 

# Target users
The system is aimed at serving two primary user groups:
- Recruiters: Seeking efficient tools for managing and matching a high volume of candidates and job openings and ensuring high-quality matches to reduce time-to-hire.
- Job Seekers: Desiring timely responses and higher engagement rates from recruiters.

# Project Motivation:
A recruitment agency already has a pool of candidates while fetching numerous job openings from sites like LinkedIn, Indeed, and Glassdoor. The agency is seeking to enhance the efficiency of its recruitment team by providing high-quality candidates to improve the match to job openings, thereby optimizing the recruitment process.
- Volume Overload: Too many candidates and jobs.
- Matching Efficiency: Hard to find the perfect fit.
- Varied Job Sources: Inconsistent job information.
- Low Response Rates: Job seekers often receive little to no feedback.

# Metrics

## For Recruiters:
- Time-to-Screen
  - Sum of all screening times / Number of applications screened
  - Lower: indicates more efficient screening,  recruitment team is quickly evaluating incoming applications
- Interview Success Rate 
  - Number of Candidates Interviewed / Number of Candidates Matched
  - Higher: indicates an effective matching process
- Number of new Job Seeker

## For Job Seekers:
- Application Response Rate:
  - Number of applications responded to / Total number of applications
  - Higher: indicates better responsiveness
- Time-to-Respond:
  - Sum of all response times / Number of applications responded to
  - Lower: indicates faster communication
- Interview Rate:
  - Number of interviews secured / Total number of applications 
  - Higher: indicates better match quality and effective communication

## Secondary Metrics - Time-to-Hire:
- Average Time-to-Hire
  - Sum of all time-to-hire / Number of hires 
  - Lower: more efficient hiring process
- Hiring Success Rate:
  - Number of Candidates Hired / Number of Candidates Interviewed 
  - Higher: indicates an effective filtering during interviews
- Number of Candidates Hired
  - Higher: indicates an effective filtering during interviews
- First-Year Retention Rate:
  - Number of hires still employed after one year / Total number of hires one year ago 
  - Higher: effective recruitment and onboarding


# Data

## Data Collection: 
 - Job Descriptions: 
   - Source: Scrape data from job sites like LinkedIn 
   - Method: Utilize a free, open-source web scraper (**JobSpy**) to automate the collection of job descriptions. 

- Resumes and User Preferences: 
  - Source: Directly from job seekers through a web frontend 
  - Method: Provide a user-friendly interface for job seekers to submit their resumes and preferences. 

## Data Storage: 
- Location: Store the collected data in a cloud-based environment, utilizing **AWS services**. 

## Databases:
- SQL Database: Use for structured data, ensuring efficient querying and management. 
- **Elasticsearch** (NoSQL): Employ for unstructured or semi-structured data like job descriptions and resumes, benefiting from its full-text search capabilities and scalability. 


## ETL/ELT Pipeline:
- Job Description Pipeline: Schedule **API calls** to collect job descriptions, perform transformations (flattening, cleanup), and save the processed data into the database. 
- Resume Pipeline: Triggered by user upload, summarize resumes, store raw data in **S3**, and processed data in the database. 
- Matching Algorithm Pipeline: Extract and process data from resumes and job descriptions using a **GPT3.5** language model, and then send notifications based on matches. This can be triggered either on a schedule or initiated by the user. 



# Solution
- **Automated Screening**: Implement systems with LLMs to automatically screen resumes and portfolios, shortlisting candidates by their qualifications, experience, and role suitability. 
  - Job-candidate matching algorithms can incorporate factors like working style, preferred company culture, and career aspirations. 
- **Learning from Feedback**: Implement LLMs that can learn from successful hires and feedback from employers and job seekers to continually improve the matching accuracy. 



# Architecture Diagram
![Architecture Diagram](https://github.com/user-attachments/assets/eaf450c7-866c-470e-aec0-b866263c0287)

- Candidate Onboarding: Job seekers kick things off by uploading their resumes to our platform.
- Job Openings Collection: Meanwhile, our system is continuously gathering the latest job openings from top sites.
- Profile Processing: The uploaded resumes undergo an analysis where key details are extracted and readied for the next step.
- Smart Matching: Our algorithm then gets to work, smartly matching candidate profiles to the best-suited job vacancies.
- Candidate Notification: Job seekers receive personalized emails with their matched job recommendations.

Future Direction (not included in the demo) 
- Feedback Loop: Candidates click and apply, and their actions, as well as direct feedback, are fed back into the system. 
- Reporting and Metrics: We closely monitor how our matches are doing—application rates, job seeker contentment, and more.
- Continuous Refinement: The insights we gain lead us to fine-tune our matching algorithm, aiming for even happier job seekers next time around.


 
# User Journey 
Users submit their resumes through our website, and we send job recommendations via **email** based on their resumes.
<img width="1001" alt="email_jobrecommendation" src="https://github.com/user-attachments/assets/7277856f-d414-4e77-8e73-d82f35a31f70">



# Cost Estimation
We assume 100 job seekers, 1 job description. The average tokens for a job seeker resume profile is 1000 tokens, job description is 2000 tokens. Thus, one match cost 100 * 1000 + 2000 = 102k tokens. We approximately to 100k tokens for GPT3.5 API, as we searched the cost will be around the cost is  USD $0.01 per 1,000 token. Therefore, for 100,000 input tokens, the cost would be $1.00. 
We estimate everyday, we have an average 1000 new job posts. Thus the daily cost is 1000$. 
For the output, we expect ranking results per job post.
Output token cost is $0.01 per 1,000 token. Output cost per job post is less than 1000 token, but for convenience, we say it cost 0.01$ 


# Free / Self-Hosted Stack

The pipeline originally relied on several paid services. This branch swaps each one
for a free, mostly local alternative so the whole system can run on your own machine
at no cost:

| Original (paid)        | Free replacement                                   | Used in |
|------------------------|----------------------------------------------------|---------|
| OpenAI GPT-3.5         | **Ollama** local LLM (e.g. `llama3.1`)             | `utils.get_llm` |
| OpenAI embeddings      | **HuggingFace** `sentence-transformers` (local)    | `utils.get_embeddings` |
| Apify Google-jobs actor| **JobSpy** (`python-jobspy`) — Indeed/Google/etc.  | `job_scraper.py` |
| AWS OpenSearch         | **Local CSV document store** (`./data/local_store`)| `utils.save_documents` / `load_documents` |
| AWS S3 (resume bucket) | **Local folder** (`./data/resumes`)                | `web.py` |
| SendGrid               | **SMTP** via stdlib `smtplib` (e.g. Gmail)         | `matching.py` |

FAISS (vector search) was already free and is unchanged. AWS Lambda/Chalice and
DynamoDB are optional — for a fully free setup, run `python job_scraper.py` from
cron instead of deploying the Lambda.

## Setup

```bash
cd swift_mytodo
pip install -r requirements.txt

# Local LLM (one-time)
brew install ollama        # or see https://ollama.com
ollama serve &
ollama pull llama3.1

# Email credentials for sending recommendations
cp credential.txt.example credential.txt   # then add a Gmail app password
```

Optional environment overrides: `LLM_MODEL`, `OLLAMA_BASE_URL`, `EMBEDDING_MODEL`,
`LOCAL_STORE_DIR`, `RESUME_DIR`, `JOBSPY_SITES`, `JOBSPY_COUNTRY`. To use Gemini or
Groq (also free tiers) instead of Ollama, swap the single return line in
`utils.get_llm()`.

## Run the pipeline

```bash
python job_scraper.py                 # 1. scrape jobs -> ./data/local_store/jobs.csv
streamlit run web.py                  # 2. job seekers upload resumes -> ./data/resumes
python resume_summary.py              # 3. summarize resumes -> ./data/local_store/resumes.csv
python matching.py                    # 4. match + email recommendations
```





 

