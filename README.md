# 📄 Resume Matcher

A web app that compares a resume with a job description and shows a match percentage along with matched and missing keywords.

🔗 **Live Demo:** https://sanika-resume-matcher.streamlit.app/

## Features
- Match score between resume and job description (TF-IDF + cosine similarity)
- Matched keywords
- Missing keywords to add to your resume

## Tech Stack
Python, Streamlit, scikit-learn

## Run Locally
    pip install -r requirements.txt
    streamlit run app.py