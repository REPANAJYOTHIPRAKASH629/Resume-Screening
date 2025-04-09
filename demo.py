import streamlit as st
import PyPDF2
from docx import Document
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from streamlit_webrtc import webrtc_streamer
import re
import av
import time
import os

# ================== 🔐 INIT: Google Gemini Chat Model ====================
def initialize_chat_model():
    with open("key.txt", "r") as f:
        GOOGLE_API_KEY = f.read().strip()

    chat_model = ChatGoogleGenerativeAI(
        google_api_key=GOOGLE_API_KEY,
        model="gemini-1.5-pro-latest",
        temperature=0.4,
        max_tokens=2000
    )
    return chat_model

chat_model = initialize_chat_model()

chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
        You are a helpful assistant for job readiness analysis.
        You must follow all user instructions strictly.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{human_input}")
])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
output_parser = StrOutputParser()

chain = RunnablePassthrough.assign(
    chat_history=RunnableLambda(lambda human_input: memory.load_memory_variables(human_input)['chat_history'])
) | chat_prompt_template | chat_model | output_parser

# ================== 📄 FILE TEXT EXTRACTOR ====================
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    return ""

# ================== 🧠 MCQ PARSER ====================
def parse_mcq(mcq_text):
    questions = re.split(r'\d+\.\s+', mcq_text)[1:]
    parsed = []
    for q in questions:
        lines = q.strip().split('\n')
        question = lines[0].strip()
        options = {}
        for line in lines[1:]:
            match = re.match(r'\s*[-\u2022]?\s*([A-Da-d])[\).]?\s*(.*)', line)
            if match:
                key = match.group(1).upper()
                value = match.group(2).strip()
                options[key] = value
        if options:
            parsed.append({'question': question, 'options': options})
    return parsed

# ================== 🖼️ Streamlit UI ====================
st.set_page_config(page_title="AI Job Readiness Portal", layout="wide")
st.title("🧠 AI-Powered Job Application Readiness")

resume = st.file_uploader("📄 Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
jd = st.file_uploader("📃 Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])

if resume and jd:
    resume_text = extract_text(resume)
    jd_text = extract_text(jd)
    st.success("✅ Files uploaded successfully!")

# === MCQ Section ===
if st.button("🧪 Generate MCQs") or "mcqs" in st.session_state:
    if "mcqs" not in st.session_state:
        query = {"human_input": """
        Create 30 MCQs on general programming concepts.
        Format exactly like this:
        1. What does HTML stand for?
           - A) Hyper Text Markup Language
           - B) Hyper Trainer Marking Language
           - C) High Text Machine Language
           - D) None of the above
        """}
        with st.spinner("Generating MCQs..."):
            mcq_raw = chain.invoke(query)
            mcqs = parse_mcq(mcq_raw)
            st.session_state.mcqs = mcqs
            st.session_state.q_index = 0
            st.session_state.answers = {}
            st.session_state.start_time = time.time()

    mcqs = st.session_state.mcqs
    q_index = st.session_state.q_index
    total_questions = len(mcqs)
    current_question = mcqs[q_index]

    st.markdown(f"### ❓ Question {q_index+1} / {total_questions}")
    st.markdown(f"**{current_question['question']}**")

    selected_option = st.radio(
        "Choose your answer:",
        options=list(current_question["options"].keys()),
        format_func=lambda x: f"{x}) {current_question['options'][x]}",
        key=f"question_{q_index}"
    )

    elapsed_time = int(time.time() - st.session_state.start_time)
    remaining_time = 60 - elapsed_time
    if remaining_time <= 0:
        st.warning("⏰ Time’s up! Moving to next question...")
        time.sleep(1)
        if q_index < total_questions - 1:
            st.session_state.answers[q_index] = selected_option
            st.session_state.q_index += 1
            st.session_state.start_time = time.time()
            st.rerun()
        else:
            st.session_state.answers[q_index] = selected_option
            st.success("✅ Quiz completed!")
    else:
        st.markdown(f"⏳ Time left: **{remaining_time} seconds**")

    if q_index < total_questions - 1:
        if st.button("Next"):
            st.session_state.answers[q_index] = selected_option
            st.session_state.q_index += 1
            st.session_state.start_time = time.time()
            st.rerun()
    else:
        if st.button("Submit"):
            st.session_state.answers[q_index] = selected_option
            st.success("✅ Quiz submitted successfully!")
            st.markdown("### 📋 Your Answers")
            for i, ans in st.session_state.answers.items():
                st.markdown(f"**Q{i+1}:** {st.session_state.mcqs[i]['question']}")
                st.markdown(f"- Your Answer: {ans}) {st.session_state.mcqs[i]['options'][ans]}")

# === Coding Questions Section ===
if st.button("💻 Show Coding Questions"):
    query = {"human_input": """
    Provide 2 programming/coding questions in paragraph format with sample input/output and constraints only.
    Do not provide answers.
    """}
    coding_questions = chain.invoke(query)
    st.session_state.coding_questions_raw = coding_questions

if "coding_questions_raw" in st.session_state:
    st.markdown("## 🧮 Coding Questions")
    st.markdown(st.session_state.coding_questions_raw)
    st.text_area("✍️ Your Answer", key="code_answers_area", height=300)

# === Webcam Proctoring + Feedback ===
st.markdown("## 📸 Webcam Proctoring & Behavior Analysis")
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # Behavior detection logic (placeholder)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="video", video_frame_callback=video_frame_callback, video_size=(320, 240))

# Placeholder for webcam-based analysis
feedback_query = {"human_input": """
Simulate a behavioral analysis based on webcam activity.
Evaluate:
- Eye contact
- Attention span
- Overall presentation
Give a short summary assuming average behavior.
"""}
st.subheader("🧠 AI Webcam Behavior Feedback")
st.write(chain.invoke(feedback_query))

# === Final Summary ===
st.markdown("## 📊 Final Evaluation Report")
st.markdown("Combine MCQ performance, coding responses, and webcam analysis for a full readiness score.")
