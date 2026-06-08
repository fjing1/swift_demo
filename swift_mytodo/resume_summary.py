"""Resume preprocessing ETL (free stack).

  - get resumes from the local resume folder (was: AWS S3)
  - summarize each resume with a local LLM RAG chain (was: OpenAI GPT-3.5)
  - save the processed summaries to the local document store (was: OpenSearch)

Trigger: run after new resumes are uploaded via web.py.
    python resume_summary.py
"""
import os
import glob

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader

import utils

RESUMES_INDEX = "resumes"


def resume_summary(pdf):
    """
    input:
    pdf: resume path

    Build a RAG chain over the resume PDF and ask the LLM for a structured summary.

    output:
    resume summary text
    """
    print(f'Summary Resume: {pdf}\n')
    loader = PyPDFLoader(pdf)
    rag_chain = utils.rag_invoke(loader)

    summary = rag_chain.invoke("""
    Structure background by filling in the section below from the Resume
    - Name (with first and last name):
    - Email (contact info):
    - Best Matching Job Positions (for example "Data Engineer", "Product Manager", etc) rank from best fit to less fit:
    - Top Skills he/she has (tile to 20):
    - Industries (Like Retail, Tech, Consulting, Government, Finance, etc.):
    - Years of Experience (estimate if not specified. for example: don't print Internship experience since 2021, print 3 years. ):
    - Location (please be valid City, Country, if it is not valid one just don't included, for example: Meta (Ads Infra and Ranking) is not a location):
    - Full Experience (including everything):

    """
    )
    return summary


def extract_personal_preference(path):
    """
    input : path = './data/resumes/Felix_Jing_sde_toronto_fjing007@gmail.com.pdf'
    output: ", Desired Position: sde, Desired Location: toronto"

    Filename format produced by web.py is:
        {first}_{last}_{title}_{location}_{email}.pdf
    """
    parts = path.split('/')
    info_parts = parts[-1].split('_')
    position = info_parts[2] if len(info_parts) > 2 else ''
    location = info_parts[3] if len(info_parts) > 3 else ''

    personal_preference = f",\n Desired Position: {position},\n Desired Location: {location}"
    return personal_preference


def resume_summary_to_df(summary, pdf):
    """
    input:
    summary: summary of resume text
    pdf: resume path (its filename carries the personal-preference info)

    output: single-row df with a 'summary' column holding the summary lines
    """
    personal_preference = extract_personal_preference(pdf)
    input_string = summary + personal_preference
    parts = input_string.split('\n')

    df_temp = pd.DataFrame([{'summary': parts}])
    return df_temp


def llm_resume_preprocess(local_resume_path=utils.RESUME_DIR):
    """
    input:
    local_resume_path: folder holding uploaded resume PDFs

    output: df with every resume's summary
    """
    pattern = os.path.join(local_resume_path, '*.pdf')
    resume_files = glob.glob(pattern)
    print(f"Found {len(resume_files)} Resumes:", resume_files)

    all_resume_df = pd.DataFrame()
    for pdf in resume_files:
        summary = resume_summary(pdf)
        resume_df_temp = resume_summary_to_df(summary, pdf)
        all_resume_df = pd.concat([all_resume_df, resume_df_temp], ignore_index=True)

    print(f'Processed all {len(all_resume_df)} resumes. Done.\n')
    return all_resume_df


if __name__ == "__main__":
    all_resume_df = llm_resume_preprocess()

    os.makedirs('./data/placeholder_resume_job', exist_ok=True)
    all_resume_df.to_csv('./data/placeholder_resume_job/all_resume_df.csv', index=False)

    # overwrite: reprocess the full local resume set on each run
    utils.save_documents(RESUMES_INDEX, all_resume_df, mode='overwrite')
