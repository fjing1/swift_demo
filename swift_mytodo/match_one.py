"""Match a single resume against the scraped jobs and print the top matches.

A terminal-only version of the matching pipeline: no Streamlit upload and no
email. It reads one resume PDF, ranks the best jobs from the local store, and
prints the result.

Two ranking modes:
  - default  : summarize the resume with the local LLM, then rank with the same
               RAG prompt the bulk pipeline uses (needs Ollama running).
  - --no-llm : rank purely by embedding similarity (no LLM needed; uses the
               free local HuggingFace embeddings, so it runs anywhere).

Examples:
    python match_one.py ../Fei_Jing_Resume_2026.pdf --title "site reliability engineer" --location Austin
    python match_one.py ../Fei_Jing_Resume_2026.pdf --no-llm

Prerequisites:
  - Jobs already scraped into the local store:  python job_scraper.py
  - For the default (LLM) mode: Ollama running with utils.LLM_MODEL pulled.
"""
import os
import argparse

from langchain_community.document_loaders.csv_loader import CSVLoader

import utils
import matching
import resume_summary

JOBS_CSV = "./data/placeholder_resume_job/match_one_jobs.csv"


def _load_jobs(limit):
    """Load scraped jobs from the local store; return None if there are none."""
    jobs_df = matching.get_latest_jobs(limit=limit)
    if jobs_df.empty:
        print("No jobs in the local store. Run `python job_scraper.py` first.")
        return None
    return jobs_df


def _llm_matches(pdf_path, title, location, limit):
    """Rank with the local LLM + RAG prompt (same logic as matching.py)."""
    # 1. summarize the resume with the local LLM
    summary = resume_summary.resume_summary(pdf_path)

    # 2. append any preferences passed on the command line. The bulk pipeline
    #    reads these from the filename; here we take them explicitly so the
    #    resume file can be named anything.
    preference = ""
    if title:
        preference += f",\n Desired Position: {title}"
    if location:
        preference += f",\n Desired Location: {location}"
    resume_lines = (summary + preference).split("\n")

    # 3. load scraped jobs and hand them to the existing matcher
    jobs_df = _load_jobs(limit)
    if jobs_df is None:
        return None

    os.makedirs(os.path.dirname(JOBS_CSV), exist_ok=True)
    jobs_df.to_csv(JOBS_CSV, index=False)

    loader = CSVLoader(file_path=JOBS_CSV)
    _email, job_list = matching.matching(loader, resume_lines)
    return job_list


def _embedding_matches(pdf_path, limit, top_n):
    """Rank purely by embedding similarity between resume text and each job.

    No LLM required: embeds the resume and every job with the local
    sentence-transformers model and ranks by cosine similarity.
    """
    import numpy as np

    resume_text = utils.read_resume(pdf_path)

    jobs_df = _load_jobs(limit)
    if jobs_df is None:
        return None

    def job_text(row):
        parts = [str(row.get(c, "")) for c in ("title", "company", "location", "description")]
        return " | ".join(p for p in parts if p and p.lower() != "nan")

    docs = [job_text(row) for _, row in jobs_df.iterrows()]

    embeddings = utils.get_embeddings()
    job_vecs = np.asarray(embeddings.embed_documents(docs))
    resume_vec = np.asarray(embeddings.embed_query(resume_text))

    # cosine similarity (rows are L2-normalized for MiniLM, but normalize anyway)
    denom = np.linalg.norm(job_vecs, axis=1) * np.linalg.norm(resume_vec) + 1e-10
    sims = (job_vecs @ resume_vec) / denom

    top_idx = np.argsort(sims)[::-1][:top_n]
    results = []
    for rank, idx in enumerate(top_idx, 1):
        row = jobs_df.iloc[idx]
        link = row.get("applyLink") or row.get("job_url") or ""
        results.append(
            f"{rank}. {row.get('title', '?')} at {row.get('company', '?')} "
            f"[score {sims[idx]:.3f}] - Apply on: {link}"
        )
    return results


def match_one(pdf_path, title=None, location=None, limit=100, use_llm=True, top_n=3):
    """Return a list of top-match lines for one resume, or None if unavailable."""
    if not os.path.exists(pdf_path):
        print(f"Resume PDF not found: {pdf_path}")
        return None

    if use_llm:
        return _llm_matches(pdf_path, title, location, limit)
    return _embedding_matches(pdf_path, limit, top_n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print the top job matches for a single resume PDF."
    )
    parser.add_argument("resume", help="Path to the resume PDF.")
    parser.add_argument("--title", help="Desired job title, e.g. 'data engineer'.")
    parser.add_argument("--location", help="Desired location, e.g. 'Toronto'.")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max number of scraped jobs to rank against (default 100).")
    parser.add_argument("--top-n", type=int, default=3,
                        help="How many matches to show in --no-llm mode (default 3).")
    parser.add_argument("--no-llm", dest="use_llm", action="store_false",
                        help="Rank by embedding similarity only (no Ollama needed).")
    cli = parser.parse_args()

    matches = match_one(
        cli.resume,
        title=cli.title,
        location=cli.location,
        limit=cli.limit,
        use_llm=cli.use_llm,
        top_n=cli.top_n,
    )

    if matches:
        print("\n================ TOP MATCHES ================")
        for line in matches:
            print(line)
