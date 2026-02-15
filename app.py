import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# ---------------- CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #667eea, #764ba2);
}
.card {
    background-color: white;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.2);
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #2c3e50;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 20px;
}
.stTextArea textarea {
    border-radius: 10px;
    font-size: 16px;
}
.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #43cea2, #185a9d);
    color: white;
    font-size: 20px;
    border-radius: 12px;
    padding: 10px;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
}
.result-real {
    background-color: #eafaf1;
    padding: 20px;
    border-radius: 10px;
    color: #1e8449;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}
.result-fake {
    background-color: #fdecea;
    padding: 20px;
    border-radius: 10px;
    color: #922b21;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}
.footer {
    text-align: center;
    color: white;
    margin-top: 30px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

true["label"] = 1
fake["label"] = 0

data = pd.concat([true, fake])

X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

# ---------------- UI ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI based system to detect fake and real news</div>', unsafe_allow_html=True)

user_input = st.text_area("✍️ Enter News Text:")

if st.button("🔍 Check News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news text!")
    else:
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]

        if prediction == 1:
            st.markdown('<div class="result-real">✅ This is REAL News</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-fake">❌ This is FAKE News</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


