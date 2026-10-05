import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Matcher", page_icon="📄", layout="centered")

st.title("📄 Resume Matcher")
st.caption("Compare your resume with a job description and see how well they match.")

resume_text = st.text_area(
    "Your Resume",
    height=250,
    placeholder="Paste your resume text here...",
)

jd_text = st.text_area(
    "Job Description",
    height=250,
    placeholder="Paste the job description here...",
)


def calculate_match(resume, jd):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume, jd])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(score * 100, 1)


def find_keywords(resume, jd):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit([jd])
    jd_words = set(vectorizer.get_feature_names_out())

    resume_vectorizer = TfidfVectorizer(stop_words="english")
    resume_vectorizer.fit([resume])
    resume_words = set(resume_vectorizer.get_feature_names_out())

    matched = sorted(jd_words & resume_words)
    missing = sorted(jd_words - resume_words)
    return matched, missing


if st.button("Analyze Match", type="primary"):
    if not resume_text.strip() or not jd_text.strip():
        st.warning("Please paste both your resume and the job description.")
    else:
        match_score = calculate_match(resume_text, jd_text)
        matched, missing = find_keywords(resume_text, jd_text)

        st.subheader("Match Score")
        st.metric(label="Resume vs Job Description", value=f"{match_score}%")
        st.progress(min(int(match_score), 100))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Matched Keywords")
            if matched:
                st.write(", ".join(matched))
            else:
                st.write("No matching keywords found.")

        with col2:
            st.subheader("❌ Missing Keywords")
            if missing:
                st.write(", ".join(missing))
            else:
                st.write("Great, nothing is missing!")