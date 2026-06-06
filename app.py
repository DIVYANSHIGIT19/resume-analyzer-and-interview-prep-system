import streamlit as st
import PyPDF2
import re
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= DB =================
conn = sqlite3.connect("resume_data.db", check_same_thread=False)
cursor = conn.cursor()

# ===== OLD TABLE (UNCHANGED) =====
cursor.execute("""
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    score REAL,
    skills TEXT,
    missing TEXT,
    date TEXT
)
""")

# ===== NEW TABLE (ADDED ONLY) =====
cursor.execute("""
CREATE TABLE IF NOT EXISTS quiz_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    question TEXT,
    answer TEXT,
    score REAL,
    date TEXT
)
""")

conn.commit()

# ================= FUNCTIONS =================

def extract_text(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def clean(text):
    return re.sub(r'\s+', ' ', text.lower())

def similarity(resume, jd):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    tfidf = vec.fit_transform([clean(resume), clean(jd)])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100

def extract_skills(text):
    skills_db = [
        "python","java","sql","machine learning","deep learning",
        "html","css","javascript","react","node","django","flask",
        "data analysis","excel","power bi"
    ]
    text = clean(text)
    return list(set([s for s in skills_db if s in text]))

def keyword_gap(resume, jd):
    jd_words = set(clean(jd).split())
    res_words = set(clean(resume).split())
    return list(jd_words - res_words)[:10]

def final_score(resume, jd):
    sim = similarity(resume, jd)
    penalty = len(keyword_gap(resume, jd)) * 1.5
    length_score = min(len(resume.split()) / 600, 1) * 10
    return round(max(0, min(100, sim * 0.85 + length_score - penalty)), 2)

def feedback(score):
    if score < 40:
        return "❌ Weak Resume"
    elif score < 70:
        return "⚠️ Medium Resume"
    else:
        return "✅ Strong Resume"

def generate_questions(skills):
    qs = []
    for s in skills:
        qs.append(f"Explain {s}")
        qs.append(f"Real-world use of {s}")
    return qs[:6]

def save_to_db(name, score, skills, missing):
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO results VALUES (NULL,?,?,?,?,?)",
        (name, score, ", ".join(skills), ", ".join(missing), date)
    )
    conn.commit()

# ================= AI EVALUATOR (NEW) =================

def evaluate_answer(question, answer, skills):
    answer = clean(answer)
    score = 50

    for s in skills:
        if s in answer:
            score += 15

    if len(answer.split()) < 5:
        score -= 20

    if any(word in answer for word in question.lower().split()):
        score += 10

    return max(0, min(100, score))

# ================= UI =================
st.set_page_config(page_title="AI Resume System", layout="wide")

st.markdown("""
<h1 style='text-align:center;color:#00ff88'>
🚀 AI Resume Analyzer and Interview prep system
</h1>
""", unsafe_allow_html=True)

name = st.text_input("Enter Your Name")

tabs = st.tabs([
    "📄 Analyzer",
    "🧠 Skills Detected",
    "🎯 Questions Generated",
    "🧠 Practice Mode",
    "📊 Dashboard"
])

# ================= TAB 1 =================
with tabs[0]:
    file = st.file_uploader("Upload Resume", type=["pdf"], key="u1")
    jd = st.text_area("Paste Job Description")

    if st.button("Analyze Resume"):
        if file and jd and name:

            text = extract_text(file)

            sim = similarity(text, jd)
            score = final_score(text, jd)
            skills = extract_skills(text)
            missing = keyword_gap(text, jd)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Match %", f"{sim:.2f}")

            with col2:
                st.metric("Score", f"{score}/100")

            with col3:
                st.metric("Missing Keywords", len(missing))

            st.subheader("📊 Score Graph")
            fig, ax = plt.subplots()
            ax.bar(["Score"], [score])
            ax.set_ylim(0, 100)
            st.pyplot(fig)

            st.info(feedback(score))

            save_to_db(name, score, skills, missing)

# ================= TAB 2 =================
with tabs[1]:
    file = st.file_uploader("Upload Resume for Skills Detection", type=["pdf"], key="u2")

    if file:
        text = extract_text(file)
        skills = extract_skills(text)

        st.subheader("🧠 Detected Skills")
        st.write(skills)

# ================= TAB 3 =================
with tabs[2]:
    file = st.file_uploader("Upload Resume for Questions", type=["pdf"], key="u3")

    if file:
        text = extract_text(file)
        skills = extract_skills(text)

        st.subheader("🎯 Interview Questions")

        questions = generate_questions(skills)

        for q in questions:
            st.write("👉", q)

# ================= TAB 4 (LEETCODE PRACTICE MODE) =================
with tabs[3]:
    st.subheader("🧠 LeetCode Style Practice Mode")

    file = st.file_uploader("Upload Resume for Practice", type=["pdf"], key="u4")

    if file:
        text = extract_text(file)
        skills = extract_skills(text)

        questions = generate_questions(skills)

        for i, q in enumerate(questions):

            st.markdown(f"### Q{i+1}: {q}")

            ans = st.text_area(f"Your Answer {i+1}", key=f"a{i}")

            if st.button(f"Submit Q{i+1}", key=f"b{i}"):

                score = evaluate_answer(q, ans, skills)

                st.success(f"Score: {score}/100")

                cursor.execute("""
                    INSERT INTO quiz_results VALUES (NULL,?,?,?,?,?)
                """, (
                    name,
                    q,
                    ans,
                    score,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
                conn.commit()

# ================= TAB 5 (DUAL RANKING SYSTEM) =================
with tabs[4]:

    st.subheader("📊 Resume Analyzer Ranking ")

    cursor.execute("SELECT name, score FROM results")
    data = cursor.fetchall()

    if data:
        df = pd.DataFrame(data, columns=["Name","Score"])
        df = df.sort_values("Score", ascending=False)

        st.dataframe(df)
        st.bar_chart(df.set_index("Name"))
    else:
        st.info("No resume data yet")

    st.divider()

    st.subheader("🧠 Practice Mode Ranking ")

    cursor.execute("""
        SELECT name, AVG(score)
        FROM quiz_results
        GROUP BY name
    """)

    data2 = cursor.fetchall()

    if data2:
        df2 = pd.DataFrame(data2, columns=["Name","Avg Score"])
        df2 = df2.sort_values("Avg Score", ascending=False)

        st.dataframe(df2)
        st.bar_chart(df2.set_index("Name"))
    else:
        st.info("No practice data yet")   