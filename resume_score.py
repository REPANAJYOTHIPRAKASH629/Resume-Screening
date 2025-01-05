import re
from PIL import Image
import google.generativeai as genai
import PyPDF2
import streamlit as st
from docx import Document
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder)
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_core.messages import SystemMessage  # Updated import
from langchain_google_genai import ChatGoogleGenerativeAI


# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from Word document
def extract_text_from_word(uploaded_file):
    text = ""
    doc = Document(uploaded_file)
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def parse_mcq_questions(mcq_list):
    # Split the string into individual questions
    questions = re.split(r'\d+\.\s+', mcq_list)[1:]  #  Skip the empty first element
    parsed_questions = []
    
    for q in questions:
        # Split into question and options
        parts = q.strip().split('    - ')
        question = parts[0].strip()
        options = {
            opt[0]: opt[2:].strip()
            for opt in parts[1:]
        }
        parsed_questions.append({
            'question': question,
            'options': options
        })
    
    return parsed_questions
# Function to generate MCQs using LLM
def generate_mcqs(keywords):
    # Construct the query
    query = {"human_input": f"""
You are an advanced AI model trained to generate high-quality multiple-choice questions (MCQs).
Based on the provided list of skills: {keywords}, create **exactly 10 MCQs**. Each MCQ should focus on most important concepts related to the internal topics of each skill.
For example, if the keyword is "Python," the questions should be derived from core Python concepts, like data structures, syntax, or libraries.

The MCQs should follow this structure:

1. A clear and concise important question based on a topic within the skill.
2. Four options (labeled as A, B, C, and D).
3. Only one correct answer per question, with the other options serving as plausible distractors.

Do not provide any other information, explanations, or extra text. Output **only** the 10 MCQs in proper structure, like this:

1. Question text...
   - A) Option 1
   - B) Option 2
   - C) Option 3
   - D) Option 4

2. Question text...
   - A) Option 1
   - B) Option 2
   - C) Option 3
   - D) Option 4

Continue this format for all 10 questions.
"""}

    # Invoke the language model to generate MCQs
    response = chain.invoke(query)
    memory.save_context(query, {"output": response})

    # Return the generated MCQs as a string
    return response

# Function to evaluate MCQ answers
def evaluate_mcqs(mcq_list, answers):
    query = {"human_input": f"""
    You are an advanced AI model trained to evaluate answers for high-quality multiple-choice questions (MCQs). Act as an expert professional in all relevant skills and concepts, analyzing the user's answers in detail. Follow these instructions:
    1. Evaluate the provided answers {answers} against the correct answers for the MCQs.
    2. Award 1 mark for each correct answer. Determine if each answer is correct or incorrect.
    3. For incorrect answers:
       - Analyze deeply to identify the specific concepts or subtopics within the skill where the user is struggling.
       - Provide a focused list of concepts the user needs to improve on, derived from the incorrect answers.
    4. At the end of the evaluation, output:
       - Total marks scored (out of 10).
       - A detailed and analyzed one by one list of concepts to focus on, ensuring they address the root areas of misunderstanding or lack of knowledge.
    Output **only** the following information:
    - Total marks scored: X/10
    - Concepts to focus on: [Provide an analyzed and specific list of concepts derived from incorrect answers]
    """}

    response = chain.invoke(query)
    memory.save_context(query, {"output": response})
    return response

# Function to generate Questions using LLM
def generate_questions(keywords):
    # Construct the query
    query = {"human_input": f"""
You are a highly advanced AI trained to act as a real-time interview expert. Based on the provided keywords {keywords}, identify the most relevant skills and generate exactly two coding interview questions.
These questions should adhere to the professional structure used in coding platforms like LeetCode or HackerRank. Follow these instructions:

1. Analyze the provided keywords to identify key skills and concepts.
2. Generate two easy to medium-level coding questions that align with these skills.
3. Ensure the questions are well-structured, with a clear problem statement, input format, output format, and example(s) for clarity.
4. Output the questions in the following format:

Question 1: [Title of the Question]

Problem Statement: [Provide a clear description of the problem.]

Input Format: [Specify the format of input(s).]
Output Format: [Specify the format of output(s).]
Constraints: [Mention constraints, if applicable.]

Example(s):
- Input: [Provide sample input]
- Output: [Provide corresponding output]

Question 2: [Title of the Question]

Problem Statement: [Provide a clear description of the problem.]

Input Format: [Specify the format of input(s).]
Output Format: [Specify the format of output(s).]
Constraints: [Mention constraints, if applicable.]

Example(s):
- Input: [Provide sample input]
- Output: [Provide corresponding output]
"""}

    # Invoke the language model to generate MCQs
    response = chain.invoke(query)
    memory.save_context(query, {"output": response})

    # Return the generated MCQs as a string
    return response


# Function to Interview start using LLM
def interview(job_description_keywords):
    # Construct the query
    query = {"human_input": f"""
You are a real-time expert interviewer with in-depth knowledge of various industries, job roles, and market trends.
Your task is to conduct an interview for a specific job role based on the given keywords: {job_description_keywords}.
Analyze the keywords to fully understand the role's responsibilities, required skills, and challenges. Use this understanding to ask relevant and impactful interview questions.

Rules:
1. Begin the interview with a self-introduction question to ease the candidate into the process.
2. Ask 10 highly effective, real-world interview questions tailored to the role, progressing from general to more specific and challenging.
3. Ensure the questions focus on assessing the candidate’s practical knowledge, problem-solving skills, and ability to handle real-world scenarios.
4. Incorporate situational and behavioral questions to evaluate how the candidate handles challenges and decision-making.
5. The last two questions must delve into the candidate’s past projects, focusing on:
   - The project's purpose and goals.
   - Challenges faced and how they were addressed.
   - Impact and measurable outcomes.
6. Provide one question at a time, without additional context, explanations, or formatting.
7. Questions must be clear, concise, and aligned with the job role, ensuring they reflect real-time industry expectations.

Start the interview with the first question.
"""}

    # Invoke the language model to generate MCQs
    response = chain.invoke(query)
    memory.save_context(query, {"output": response})

    # Return the generated MCQs as a string
    return response


# Initialize Google Generative AI chat model
def initialize_chat_model():
    with open("key.txt", "r") as f:
        GOOGLE_API_KEY = f.read().strip()

    chat_model = ChatGoogleGenerativeAI(
        google_api_key=GOOGLE_API_KEY,
        model="gemini-1.5-pro-latest",
        temperature=0.4,
        max_tokens=2000,
        timeout=120,
        max_retries=5,
        top_p=0.9,
        top_k=40,
        presence_penalty=0.6,
        frequency_penalty=0.3
    )
    return chat_model

chat_model = initialize_chat_model()

# Create Chat Template
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""  You are a language model designed to follow user instructions exactly as given.
            Do not take any actions or provide any information unless specifically directed by the user.
            Your role is to fulfill the user's requests precisely without deviating from the instructions provided."""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}")
    ]
)

# Initialize the Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create an Output Parser
output_parser = StrOutputParser()

# Define a chain
chain = RunnablePassthrough.assign(
            chat_history=RunnableLambda(lambda human_input: memory.load_memory_variables(human_input)['chat_history'])
        ) | chat_prompt_template | chat_model | output_parser

# Streamlit App
# Page configuration
st.set_page_config(
    page_title="TASK TITANS",  # Page title that will show in the browser tab
    page_icon="✨",             # Page icon (emoji or custom icon)
    layout="wide",             # Use wide layout for the page
    initial_sidebar_state="collapsed"  # Optional: Start with collapsed sidebar (if any)
)


st.markdown(
    """
    <style>
    .header {
        background-color: #FF6347;
        color: white;
        padding: 10px;
        text-align: center;
        font-size: 2em;
        font-weight: bold;
    }
    </style>
    
    """,
    unsafe_allow_html=True,
)

#st.markdown("## Part-1: Upload Files, Summarize, and Extract Keywords")
st.markdown("""
    <style>
    .main {
        font-family: 'Arial', sans-serif;
    }
    .header {
        font-size: 36px;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 10px;
    }
    .subheader {
        font-size: 24px;
        font-weight: 500;
        color: #333;
    }
    .button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        cursor: pointer;
        font-size: 16px;
        border-radius: 5px;
    }
    .button:hover {
        background-color: #45a049;
    }
    .section {
        margin-top: 20px;
    }
    .image {
        width: 100%;
        height: auto;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .info {
        background-color: #e7f7ff;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .question {
        background-color: #f2f2f2;
        padding: 20px;
        border-radius: 10px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)




with st.sidebar:
    st.markdown(
        """
        <style>
        .stSidebar {
            background-image: url('https://png.pngtree.com/thumb_back/fw800/background/20220617/pngtree-group-of-smiling-teenagers-over-green-background-guys-people-attractive-photo-image_18808062.jpg'); /* Replace with your image URL */
            background-size: cover;  /* Ensure the image covers the sidebar */
            background-position: center;  /* Center the image */
            background-repeat: no-repeat; /* Don't repeat the image */
            height: 100vh;  /* Make the sidebar take the full height */
        }
        .sidebar-text {
            color: white;
            padding: 20px;
            background-color: rgba(0, 0, 0, 0.6);  /* Semi-transparent background for text */
            border-radius: 10px;
        }
        </style>
        """, unsafe_allow_html=True
    )
    


# Main title and description
st.markdown("<div class='header'>Task Titans: Optimize Your Job Application</div>", unsafe_allow_html=True)
st.markdown("""
    Welcome to Task Titans! This tool allows you to upload your resume and a job description to evaluate how well your resume matches the job requirements. We will generate an ATS (Applicant Tracking System) score and also test your understanding of the job role with multiple-choice questions (MCQ), Coding and Mock Interview
""")

# App Header with Image
header_image = "https://techcrunch.com/wp-content/uploads/2015/06/interviews-e1433244493315.jpg"  # Path to your header image
st.image(header_image, use_container_width=False, width=600, caption="Task Titans: Resume and Job Description Matching")


# # File upload section
# file1 = st.file_uploader("Upload your resume (PDF or DOCX):", type=["pdf", "docx"])
# file2 = st.file_uploader("Upload the job description (PDF or DOCX):", type=["pdf", "docx"])

# Custom CSS for styling the file uploader and other buttons
st.markdown("""
    <style>
    /* General body styling */
    body {
        background-color: #f2f7fc;
        font-family: 'Arial', sans-serif;
    }

    /* Header styling for the upload section */
    .upload-section {
        background: linear-gradient(135deg, #6e7fd9, #4f9fff);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1);
        margin-bottom: 40px;
    }

    .upload-section h3 {
        font-size: 1.75rem;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Green file uploader button styling */
    .stFileUploader {
        display: inline-block;
        width: 100%;
        padding: 15px 30px;
        background-color: #28a745;  /* Green color */
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        text-align: center;
        cursor: pointer;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        transition: background-color 0.3s, transform 0.2s ease-in-out;
    }

    .stFileUploader:hover {
        background-color: #FFF000;  /* Darker green on hover */
        transform: translateY(-2px);
    }

    /* File upload instruction section */
    .upload-instruction {
        font-size: 1rem;
        color: #444;
        font-weight: 500;
        text-align: center;
        margin-top: 20px;
    }

    /* Success message styling */
    .success-message {
        color: #4CAF50;
        font-size: 1rem;
        font-weight: 500;
        margin-top: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# File 1: Resume upload
file1 = st.file_uploader("Upload your resume (PDF or DOCX):", type=["pdf", "docx"])
if file1 is not None:
    st.markdown(f'<p class="success-message">Successfully uploaded: {file1.name}</p>', unsafe_allow_html=True)

# File 2: Job description upload
file2 = st.file_uploader("Upload the job description (PDF or DOCX):", type=["pdf", "docx"])
if file2 is not None:
    st.markdown(f'<p class="success-message">Successfully uploaded: {file2.name}</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Add some instructions for users
st.markdown("""
    <div class="upload-instruction">
        <p>Ensure that both files are uploaded correctly before proceeding with further steps. Your resume should be in PDF or DOCX format, and the job description should also follow the same format.</p>
    </div>
""", unsafe_allow_html=True)




if file1 and file2:
    try:
        # Detect file type and extract text for file 1
        if file1.name.endswith('.pdf'):
            text1 = extract_text_from_pdf(file1)
        elif file1.name.endswith('.docx'):
            text1 = extract_text_from_word(file1)
        else:
            st.error("Unsupported file type for file 1")

        # Detect file type and extract text for file 2
        if file2.name.endswith('.pdf'):
            text2 = extract_text_from_pdf(file2)
        elif file2.name.endswith('.docx'):
            text2 = extract_text_from_word(file2)
        else:
            st.error("Unsupported file type for file 2")


        # Ensure session state variables are initialized

        # if "ats_score_calculated" not in st.session_state:
        #     st.session_state.ats_score_calculated = False
        if 'resume_keywords' not in st.session_state:
            st.session_state.resume_keywords = set(text1)
        if 'job_description_keywords' not in st.session_state:
            st.session_state.job_description_keywords = set(text2)        

        # Button to Calculate ATS Score
        if st.button("ATS Score"): #or st.session_state.ats_score_calculated:
            
            #st.session_state.ats_score_calculated = True
            st.markdown("### ATS Score Calculation")
            query = {"human_input": f""""
Your task is to act as a highly advanced Applicant Tracking System (ATS) that evaluates the compatibility of a candidate's resume with a given job description. You will meticulously extract and analyze all relevant keywords and information from both the resume and the job description, including but not limited to Role-Specific Keywords, Technical Skills, Certifications, Experience, Soft Skills, Job Responsibilities, Industry Keywords, Methodologies and Practices, Keywords Indicating Preferences, and Core Values.

You will then calculate an ATS score on a scale of 0-100, reflecting how well the resume matches the job description. The score should be based on the following criteria:

Keywords Matching (20%): The extent to which the resume contains the exact keywords and phrases mentioned in the job description.

Skills and Competencies (20%): The presence and relevance of skills and competencies that align with the job requirements.

Formatting (10%): The clarity and simplicity of the resume format, ensuring that the ATS can easily parse the information.

Job Title Match (10%): The similarity between the candidate's previous job titles and the job title in the description.

Experience and Education (20%): Whether the candidate's experience level and education meet the job requirements.

Customization (20%): How well the resume is tailored to the specific job description, including the use of industry-specific language and terminology.

For each criterion, provide a detailed breakdown of the match percentage, highlighting where the candidate meets the requirements and where there are gaps. Finally, provide an overall ATS score and a summary of the candidate's strengths and areas for improvement.

Ensure that the evaluation is done in real-time and with 100% accuracy, taking into account all possible factors that a traditional ATS would consider."


Job Description Keywords:
{st.session_state.job_description_keywords}

Resume Keywords:
{st.session_state.resume_keywords}
"""}

            response = chain.invoke(query)
            memory.save_context(query, {"output": response})

            st.write(response)

        if 'questions' not in st.session_state:
            # Your MCQ string goes here
            mcq_list = generate_mcqs(st.session_state.job_description_keywords)
            st.session_state.questions = parse_mcq_questions(mcq_list)
            
        if 'current_question' not in st.session_state:
            st.session_state.current_question = 0
        if 'answers' not in st.session_state:
            st.session_state.answers = []            
            
        if "mcq_button" not in st.session_state:
            st.session_state.mcq_button = False
            
        if st.button("MCQ Test") or st.session_state.mcq_button:               
            st.session_state.mcq_button = True
# Display current question number and total questions
            st.write(f"Question {st.session_state.current_question + 1} of {len(st.session_state.questions)}")
             
            # Display current question
            current_q = st.session_state.questions[st.session_state.current_question]
            st.write(current_q['question'])
            
            # Create radio buttons for options with the corrected format_func
            answer = st.radio(
                "Select your answer:",
                options=['A', 'B', 'C', 'D'],  # List of option keys
                format_func=lambda x: f"{x}) {current_q['options'].get(x, ' ')}",
                key=f"question_{st.session_state.current_question}"  # Unique key per question
            )
            
            # Navigation buttons in columns
            col1, col2 = st.columns(2)
            
            if st.session_state.current_question > 0:
                with col1:
                    if st.button("Previous"):
                        st.session_state.current_question -= 1
                        st.rerun()
            
            if st.session_state.current_question < len(st.session_state.questions) - 1:
                with col2:
                    if st.button("Next"):
                        st.session_state.answers.append(f"{st.session_state.current_question + 1}-{answer}")
                        st.session_state.current_question += 1
                        st.rerun()
            else:
                with col2:
                    if st.button("Submit"):
                        st.session_state.answers.append(f"{st.session_state.current_question + 1}-{answer}")
                        st.write("Quiz completed! Your answers:")
                        
                        
                        query = {"human_input": f"""
    You are an advanced AI model trained to evaluate answers for high-quality multiple-choice questions (MCQs). Act as an expert professional in all relevant skills and concepts, analyzing the user's answers in detail. Follow these instructions:
    1. Evaluate the provided answers : {st.session_state.answers} against the correct answers for the MCQs.
    2. Award 1 mark for each correct answer. Determine if each answer is correct or incorrect.
    3. For incorrect answers:
       - Analyze deeply to identify the specific concepts or subtopics within the skill where the user is struggling.
       - Provide a focused list of concepts the user needs to improve on, derived from the incorrect answers.
    4. At the end of the evaluation, output:
       - Total marks scored (out of 10).
       - A detailed and analyzed one by one list of concepts to focus on, ensuring they address the root areas of misunderstanding or lack of knowledge.
    Output **only** the following information:
    - Total marks scored: X/10
    - Concepts to focus on: [Provide an analyzed and specific list of concepts derived from incorrect answers]
    """}

                        response = chain.invoke(query)
                        memory.save_context(query, {"output": response})
                        st.session_state.mcq_button = False
                        #st.write(response)
                        
                        #st.write(st.session_state.answers)
                        
        if "generate_questions_button" not in st.session_state:
            st.session_state.generate_questions_button = False
                                    
        if st.button("Generate Questions") or st.session_state.generate_questions_button:
            st.session_state.generate_questions_button = True
                # Generate questions
            
            if 'questions_response' not in st.session_state:
                st.session_state.questions_response = generate_questions(st.session_state.job_description_keywords)
                        
            
            # Split questions
            code_questions = [q.strip() for q in st.session_state.questions_response.split("Question")[1:]]
            code_questions = [f"Question{q}" for q in code_questions]
            
            # Display questions and collect answers
            st.session_state.code_questions = code_questions
            st.session_state.coding_answers = [""] * len(code_questions)
            
            # Display each question with a text area for answers
            for i, question in enumerate(code_questions):
                st.markdown(f"### {question}")
                cod_answer = st.text_area(f"Your Answer for Question {i+1}", key=f"answer_{i}")
                st.session_state.coding_answers[i] = cod_answer
            
            if st.button("Submit Answers"):
                st.write("### Submitted Answers:")
                #st.write(st.session_state.coding_answers)


                query = {"human_input": f"""
Evaluate the following user responses to two coding questions:

**User Responses:**
{st.session_state.coding_answers}

**Evaluation Criteria:**

* **Perfection:** Each question carries 10 marks.
* **Assess:**
    * Correctness of the code logic and implementation.
    * Efficiency of the solution (time and space complexity).
    * Code readability, maintainability, and adherence to best practices.
    * Handling of edge cases and potential errors.

**Output:**

* **Marks:**
    * **Question 1:** [Out of 10 marks]
    * **Question 2:** [Out of 10 marks]
* **Analysis:**
    * Identify areas where the user needs to improve.
    * Suggest specific topics or concepts for further study and practice.
    * Provide constructive feedback on the user's approach and coding style.

**Note:**
* Provide a concise and informative evaluation.
* Avoid vague or generic feedback.
"""}

                response = chain.invoke(query)
                memory.save_context(query, {"output": response})
                st.session_state.generate_questions_button = False
                st.write(response)
                
        # if "Interview_questions_button" not in st.session_state:
        #     st.session_state.Interview_questions_button = False
            
        # if st.button("Interview Questions") or st.session_state.Interview_questions_button:
        #     st.session_state.Interview_questions_button = True
            
        #     if 'interview_questions' not in st.session_state:
        #     # # Your MCQ string goes here
        #         st.session_state.interview_questions = interview(st.session_state.job_description_keywords)
        #         st.write(st.session_state.interview_questions)

        #     if 'flag' not in st.session_state:
        #         st.session_state.flag = 0
            
        #     if st.session_state.flag <= 10 :
        #         # Input from the user using chat_input
        #         human_prompt = st.chat_input(" Message Pega ...")
        #         response = chain.invoke(human_prompt)
        #         memory.save_context(human_prompt, {"output": response})                
        #         st.session_state.flag += 1
            


    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload both files to proceed.")