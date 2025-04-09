# Save as app.py
# Run using: streamlit run app.py

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
import whisper
import re
import av

# --- Initialize Chat Model ---
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
        You are a language model designed to follow user instructions exactly.
        Don't take action unless instructed.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{human_input}")
])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
output_parser = StrOutputParser()

chain = RunnablePassthrough.assign(
    chat_history=RunnableLambda(lambda human_input: memory.load_memory_variables(human_input)['chat_history'])
) | chat_prompt_template | chat_model | output_parser

# --- Extract Text from Files ---
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    return ""

# --- Parse MCQs ---
def parse_mcq(mcq_text):
    questions = re.split(r'\d+\.\s+', mcq_text)[1:]
    parsed = []
    for q in questions:
        parts = q.strip().split('  - ')
        parsed.append({
            'question': parts[0],
            'options': {opt[0]: opt[2:].strip() for opt in parts[1:] if len(opt) > 2}
        })
    return parsed

# --- Streamlit UI ---
st.set_page_config(page_title="AI Job Readiness Test", layout="wide")
st.title("🎯 AI-Powered Job Readiness Portal")

resume = st.file_uploader("📄 Upload your Resume (PDF/DOCX)", type=["pdf", "docx"])
jd = st.file_uploader("📄 Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])

if resume and jd:
    resume_text = extract_text(resume)
    jd_text = extract_text(jd)
    st.success("✅ Files uploaded and processed successfully!")

    if st.button("🧠 Generate 30 MCQs"):
        query = {"human_input": f"""
        Create 30 MCQs based on this resume and job description.
        Follow format:
        1. Question text...
           - A) Option
           - B) Option
           - C) Option
           - D) Option
        Resume: {resume_text}
        JD: {jd_text}
        """}
        mcq_raw = chain.invoke(query)
        mcqs = parse_mcq(mcq_raw)
        for i, q in enumerate(mcqs):
            st.markdown(f"**Q{i+1}: {q['question']}**")
            for opt in ['A', 'B', 'C', 'D']:
                if opt in q['options']:
                    st.markdown(f"- {opt}) {q['options'][opt]}")

    if st.button("💻 Generate Coding Questions"):
        query = {"human_input": f"""
        Generate 2 coding questions with sample input/output and constraints
        based on the resume and job description.
        Resume: {resume_text}
        JD: {jd_text}
        """}
        st.markdown(chain.invoke(query))

    if st.button("🗣️ Start AI Interview"):
        query = {"human_input": f"""
        Start a mock interview. Ask:
        1. Behavioral questions (3)
        2. Resume-based questions (3)
        3. JD-based technical questions (4)
        Provide one question per message.
        Resume: {resume_text}
        JD: {jd_text}
        """}
        interview_qs = chain.invoke(query)
        st.markdown(interview_qs)

# --- Proctoring via Webcam ---
st.markdown("## 📸 Webcam Proctoring")
st.markdown("Face detection and video frame monitoring during test session.")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="video", video_frame_callback=video_frame_callback)

# --- Voice Input & Transcription ---
st.markdown("## 🎙️ Audio Capture & Analysis")
uploaded_audio = st.file_uploader("Upload your interview audio response (mp3/wav)", type=["mp3", "wav"])
if uploaded_audio is not None:
    st.audio(uploaded_audio)
    model = whisper.load_model("base")
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_audio.read())
    result = model.transcribe("temp_audio.wav")
    st.subheader("📝 Transcript")
    st.write(result["text"])

    query = {"human_input": f"""
    Analyze this transcript:
    1. Communication clarity
    2. Behavioral alignment
    3. Confidence & delivery
    4. Relevant points mentioned

    Transcript: {result['text']}
    """}
    st.subheader("🧠 AI Feedback on Speech")
    st.write(chain.invoke(query))

# --- Final Report ---
st.markdown("## 📊 Final Evaluation Report")
st.markdown("This section will summarize performance across all modules.")
st.info("💡 To be implemented: Aggregate score and performance summary.")
