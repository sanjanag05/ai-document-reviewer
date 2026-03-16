import streamlit as st
import pandas as pd
import PyPDF2
from docx import Document
import json
import os
from groq import Groq

st.set_page_config(
    page_title="AI Document Review Tool",
    page_icon="📄",
    layout="wide"
)


client = Groq(api_key=os.environ.get("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"))

st.title("AI Document Review Tool")


# =========================
# CATEGORY INPUT
# =========================

category_input = st.text_input(
    "Enter categories separated by comma",
    placeholder="Hot, Relevant, Not Relevant"
)

categories = [c.strip() for c in category_input.split(",") if c.strip()]

# =========================
# PROMPT
# =========================

prompt = st.text_area(
    "Enter AI review instructions",
    placeholder="Mark document Hot if it mentions charger or battery explosion. Relevant if customer complaint. Otherwise Not Relevant."
)

# =========================
# TEXT EXTRACTION
# =========================

def extract_text(file):

    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        return text

    if file.name.endswith(".docx"):
        doc = Document(file)
        return " ".join([p.text for p in doc.paragraphs])

    return ""

# =========================
# AI CLASSIFICATION
# =========================

def classify(text):

    if not prompt or not categories:
        return "Uncategorized", 0, ""

    instruction = f"""
You are a document review classifier.

Read the document and classify it according to the instructions.

Instructions:
{prompt}

Allowed categories:
{", ".join(categories)}

Return the result in this JSON format:

{{
"category": "<one category>",
"confidence": "<number between 0 and 1>",
"reason": "<short explanation>"
}}

Document:
{text}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": instruction}],
        temperature=0
    )

    result = response.choices[0].message.content

    try:
        data = json.loads(result)
        return data["category"], data["confidence"], data["reason"]
    except:
        return "Uncategorized", 0, ""

# =========================
# SESSION STORAGE
# =========================

if "results" not in st.session_state:
    st.session_state.results = []

# =========================
# INPUT SECTION
# =========================

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload document or dataset",
        type=["txt", "csv", "pdf", "docx"]
    )

with col2:
    text_input = st.text_area("Paste document text")

# =========================
# ANALYZE BUTTON
# =========================

if st.button("Analyze Document"):

    if uploaded_file:

        if uploaded_file.name.endswith(".csv"):

            df_docs = pd.read_csv(uploaded_file)

            for _, row in df_docs.iterrows():

                text = str(row.iloc[1])

                prediction, confidence, reason = classify(text)

                st.session_state.results.append({
                    "ArtifactID": row.iloc[0],
                    "Document": text[:120],
                    "Prediction": prediction,
                    "Confidence": confidence,
                    "Reason": reason
                })

        else:

            text = extract_text(uploaded_file)

            prediction, confidence, reason = classify(text)

            st.session_state.results.append({
                "ArtifactID": "",
                "Document": text[:120],
                "Prediction": prediction,
                "Confidence": confidence,
                "Reason": reason
            })

    elif text_input:

        prediction, confidence, reason = classify(text_input)

        st.session_state.results.append({
            "ArtifactID": "",
            "Document": text_input[:120],
            "Prediction": prediction,
            "Confidence": confidence,
            "Reason": reason
        })

    else:
        st.warning("Please upload a document or paste text.")

# =========================
# DASHBOARD
# =========================

st.divider()
st.subheader("Review Dashboard")

if st.session_state.results:

    df = pd.DataFrame(st.session_state.results)

    st.metric("Total Documents", len(df))

    # Category metrics
    if categories:
        metric_cols = st.columns(len(categories))

        for i, category in enumerate(categories):
            count = len(df[df["Prediction"] == category])
            metric_cols[i].metric(category, count)

    # Dynamic bar chart
    counts = df["Prediction"].value_counts().reindex(categories, fill_value=0)
    st.bar_chart(counts)

    # Results table
    st.dataframe(df, use_container_width=True)

    # CSV Export
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Reviewed Documents",
        data=csv,
        file_name="review_results.csv",
        mime="text/csv"
    )

else:
    st.info("No documents analyzed yet.")