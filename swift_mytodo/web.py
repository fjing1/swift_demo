"""Resume uploader (free stack).

Streamlit form for job seekers. Resumes are saved to a local folder instead of
an AWS S3 bucket. The filename encodes the seeker's preferences and is later
parsed by resume_summary.py.

Run with:
    streamlit run web.py
"""
import os

import streamlit as st

# Keep this in sync with utils.RESUME_DIR (same default + env override) without
# importing the heavier LLM stack into the uploader.
RESUME_DIR = os.environ.get("RESUME_DIR", "./data/resumes")


def save_resume_locally(uploaded_file, resume_dir, filename):
    """Save the uploaded PDF to the local resume folder."""
    os.makedirs(resume_dir, exist_ok=True)
    dest = os.path.join(resume_dir, filename)
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    print(f"Saved resume to {dest}")
    return dest


def main(resume_dir):
    st.title('Resume Uploader')
    st.write('Upload your resume in PDF format and we will store it locally for matching!')

    first_name = st.text_input('First Name')
    last_name = st.text_input('Last Name')
    target_title = st.text_input('Target title, such as software developer, data scientist').lower()
    target_location = st.text_input("Target cities, such as Toronto, Vancouver, Montreal, etc").lower()
    email = st.text_input("Enter the email you want to receive job apply links").lower()
    # TODO:
    # salary expectation
    # highest education?
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    if uploaded_file is not None and first_name and last_name:
        if st.button('Submit'):
            # filename format consumed by resume_summary.extract_personal_preference:
            # {first}_{last}_{title}_{location}_{email}.pdf
            s3_filename = f"{first_name}_{last_name}_{target_title}_{target_location}_{email}.pdf"
            save_resume_locally(uploaded_file, resume_dir, s3_filename)
            st.success("Your resume has been uploaded successfully!")


if __name__ == "__main__":
    main(RESUME_DIR)
