import streamlit as st
import os
import re
import tempfile
import pdfplumber
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from transformers import AutoTokenizer
from huggingface_hub import login
import requests
import time
import json
import math
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from io import BytesIO
from dotenv import load_dotenv

st.markdown("""
<style>
.st-emotion-cache-zq5wmm.ezrtsby0
{
       visibility: hidden;     
}    
</style>
""",unsafe_allow_html=True)

load_dotenv()

API_URL = str(os.environ.get("hf_ep_url_Phi"))

           
headers = {
    "Accept": "application/json",
    "Authorization": "Bearer " + str(os.environ.get("hf_bearer_token")),
    "Content-Type": "application/json"
}

mcq_params = {
    "top_k": 50,
    "top_p": 0.9,
    "temperature": 0.95,
    "max_new_tokens": 1024,
    "clean_up_tokenization_spaces": True,
    "return_full_text": False,
    "do_sample": True 
    # "return_text": True,
    #"prefix": "##########",
    # "response_format": {
    #     "type": "json_object",
    # }
}

essay_params = {
    "top_k": 50,
    "top_p": 0.9,
    "temperature": 0.95,
    "max_new_tokens": 1024,
    "clean_up_tokenization_spaces": True,
    "return_full_text": False,
    "do_sample": True
    #"return_text": True,
    #"prefix": "##########",
    # "response_format": {
    #     "type": "json_object",
    # }
}

def query(payload):
    while True:
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()  # Raise an error for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)

def string_to_json(list):
    json_objects = []
    for i in list:
        json_pattern = re.compile(r'\{[^{}]*\}')

        json_strings = json_pattern.findall(i)

        for json_str in json_strings:
            try:
                json_obj = json.loads(json_str)
                json_objects.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON object: {e}")

    return json_objects

def sorting_mcq_json_objects(json_list, req_mcq_num, vector_db):
    sorted_objects = []
    for i in json_list:
        if len(sorted_objects) < req_mcq_num:
            if "question" in i and "options" in i and "answer" in i:
                mcq_options = ""
                for k in i["options"]:
                    mcq_options = mcq_options + str(i["options"].index(k) + 1) + "." + k + "\n"
                mcq_constructed = i["question"] + "\n" + mcq_options + "\n" + i["answer"]
                mcq_sim_search_list = vector_db.similarity_search_with_score(mcq_constructed)
                sum_similarity_score_mcq = 0.0
                doc_count = 0
                for doc, score in mcq_sim_search_list:
                    sum_similarity_score_mcq += score
                    doc_count += 1
                average_sim_score_mcq = sum_similarity_score_mcq / doc_count
                if average_sim_score_mcq <= 14.0:
                    sorted_objects.append(i)
                else:
                    continue
            elif "Question" in i and "Options" in i and "Answer" in i:
                i["question"] = i.pop("Question")
                i["options"] = i.pop("Options")
                i["answer"] = i.pop("Answer")
                sorted_objects.append(i)
        else:
            break
    return sorted_objects

def sorting_essay_json_objects(json_list, req_essay_num):
    sorted_objects = []
    for i in json_list:
        if len(sorted_objects) < req_essay_num:
            if "question" in i and "detailed_answer" in i:
                sorted_objects.append(i)
            elif "Question" in i and "Detailed_answer" in i:
                i["question"] = i.pop("Question")
                i["detailed_answer"] = i.pop("Detailed_answer")
                sorted_objects.append(i)
        else:
            break
    return sorted_objects

def generate_mcq_pairs(doc, num_pairs):

    mcq_prompt_template = f"""You are an creative expert in creating practice multiple-choice questions and answers based on study material.Your goal is to prepare a student for their exam. You do this by asking creative multple choice questions and providing answers about the text below:
     {doc}
Create {num_pairs} multiple-choice questions with four options and provide the correct answer. Question and answer should be in the json format under question,options,answer fields .Strickly follow the exact json format for all Questions and answers . When Question and answer generating, do not mention as activity 8.1 or something like that. instead explain the activity as a part of the question. You are a free entity from the text provided above.
Avoid "Which of the following..." or "What is..." formats unless necessary. Be creative, use analogies, scenarios, and imaginative setups related to the topic.
Below is the example of the json format for the essay type question and answer.:
JSON format example:
{{
  "question": "Imagine you are an astronaut observing plant growth in space. Which factor is most affected due to the absence of gravity?",
  "options": [
    "Photosynthesis rate",
    "Geotropism response",
    "Chlorophyll production",
    "Light absorption"
  ],
  "answer": "Geotropism response"
}}

QUESTIONS AND ANSWERS:
"""
    chat_prompt = [
            {"role": "system", 
             "content": (
                "You are a highly creative and intelligent assistant trained to generate practice multiple-choice questions "
                "from study material for students preparing for the G.C.E. O/L Science exam.\n\n"
                "Your goals:\n"
                "- Be creative: Generate thoughtful, imaginative, and non-obvious questions.\n"
                "- Avoid simple WH-questions (e.g., “What is…”, “Where does…”).\n"
                "- Encourage application, reasoning, and analysis.\n"
                "- Don’t copy sentences directly from the text — rephrase and reshape ideas.\n"
                "- Do not mention activity numbers or sections (e.g., “Activity 8.1”).\n"
                "- Strictly follow the given JSON format for each question.")
            },
            {"role": "user", 
             "content": mcq_prompt_template
             }
        ]
    
    if len(mcq_prompt_template) > 4000:  
            print("ERROR : ************************\nstack:\n"+"The document content is too long for the API to process."+"********************************stack end\n\n")
            return ""

    output = query({
        "inputs": chat_prompt,
        "parameters": mcq_params
    })

    print(output)
    assistant_response = output[0]['generated_text']
    print("\n\n\n####################   \n\nexecuting generate_mcq_pairs func......\n\n#####################")
    print("****generated text dict start ******")
    print( assistant_response)
    print("****generated text dict end ******")
    if isinstance(output, list) and 'generated_text' in output[0]:
        print("\n\n\n####################   \n\nexecuting generate_mcq_pairs func......\n\n#####################")
        print("#####################   \n\nRaw ouput mcq start......\n\n#####################")
        print(assistant_response)
        print("#####################   \n\nRaw ouput mcq end......\n\n#####################")
        return assistant_response
    else:
        st.error(f"type : {type(output)}****\n\n")
        st.error(f"Unexpected response format: {output}")
        return ""

def generate_essay_pairs(doc_content, num_pairs):

    essay_prompt_template = f"""You are an expert in creating practice essay type questions and answers based on study material.Your goal is to prepare a student for their exam. You do this by asking questions and providing detailed answers about the text below:
    {doc_content}
Create {num_pairs} essay type questions and provide the correct detailed answers. Question and answer should be in the json format under question,detailed_answer fields.Strickly follow the exact json format for all Questions and answers . When Question and answer generating, do not mention as activity 8.1 or something like that. instead explain the activity as a part of the question. You are a free entity from the text provided above.
Below is the example of the json format for the essay type question and answer.:
JSON format:
{{
  "question": "<Question created>",
  "detailed_answer": "<answer>"
}}

QUESTIONS AND ANSWERS:
"""

    chat_prompt = [
            {"role": "system", 
             "content": (
                "You are a highly creative and intelligent assistant trained to generate practice multiple-choice questions "
                "from study material for students preparing for the G.C.E. O/L Science exam.\n\n"
                "Your goals:\n"
                "- Be creative: Generate thoughtful, imaginative, and non-obvious questions.\n"
                "- Avoid simple WH-questions (e.g., “What is…”, “Where does…”).\n"
                "- Encourage application, reasoning, and analysis.\n"
                "- Don’t copy sentences directly from the text — rephrase and reshape ideas.\n"
                "- Do not mention activity numbers or sections (e.g., “Activity 8.1”).\n"
                "- Strictly follow the given JSON format for each question.")
            },
            {"role": "user", 
             "content": essay_prompt_template
             }
    ]
   
 
    if len(essay_prompt_template) > 4000:  
            print("ERROR : ************************\nstack:\n"+"The document content is too long for the API to process."+"********************************stack end\n\n")
            return ""

    output = query({
        "inputs": chat_prompt,
        "parameters": essay_params
    })

    assistant_response = output[0]['generated_text']
    print("\n\n\n####################   \n\nexecuting generate_essay_pairs func......\n\n#####################")
    print("****generated text dict start ******")
    print( assistant_response)
    print("****generated text dict end ******")
    if isinstance(output, list) and 'generated_text' in output[0]:
        print("\n\n\n####################   \n\nexecuting generate_essay_pairs func......\n\n#####################")
        print("#####################   \n\nRaw ouput essays start......\n\n#####################")
        print(assistant_response) # type: ignore
        print("#####################   \n\nRaw ouput essays end......\n\n#####################")
        return assistant_response
    else:
        st.error(f"type : {type(output)}****\n\n")
        st.error(f"Unexpected response format: {output}")
        return ""

def create_pdf(text, title):
    buffer = BytesIO()
    custom_width = 800  
    custom_height = 1000  
    doc = SimpleDocTemplate(buffer, pagesize=(custom_width, custom_height), rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []

    pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Bold', fontSize=14, leading=16, spaceAfter=14, spaceBefore=14, alignment=1, textColor='#000000', fontName="DejaVuSans")) # type: ignore
    styles.add(ParagraphStyle(name='UnicodeStyle', fontName='DejaVuSans', fontSize=12))

    title = Paragraph(title, styles['Bold'])

    story.append(title)
    story.append(Spacer(1, 20))

    text_lines = text.split('\n')
    for line in text_lines:
        if ")" in line:
            story.append(Spacer(1, 20)) 
            story.append(Paragraph(line+"\n", styles['UnicodeStyle']))
            story.append(Spacer(1, 5))
        else:
            story.append(Paragraph(line+"\n", styles['UnicodeStyle']))

    story.append(Spacer(1, 60))  
    doc.build(story)
    buffer.seek(0)
    return buffer

def read_pdf(file_path):
    text = ''
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
    return text

def check_pdfs(file):
    if file is not None:
        bytes_data = file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(bytes_data)
            temp_file_path = temp_file.name
        pdf_text = read_pdf(temp_file_path)
        return pdf_text
    else:
        st.warning("Please upload a file to Proceed !!", icon="🚨")
        return False

def estimate_qa_pairs(text):
    max_count = len(text) // 60
    if max_count >= 40:
        return 40
    else:
        return max_count 

def main():
    login(token=os.environ.get("hf_bearer_token"))
    st.title("Create Practice Questionnaires for G.C.E. O/L Science Education")
    st.sidebar.title("Upload the Syllabus Content")
    uploaded_file = st.sidebar.file_uploader("Upload files", accept_multiple_files=False, type=['pdf'])
    # st.session_state.file_name = uploaded_file.name if uploaded_file else None

    embeddings = HuggingFaceEmbeddings(model_name='GaniduA/bge-finetuned-olscience', model_kwargs={'device': 'cpu'})
    vectorDB = Chroma(persist_directory="v_db_ol_science", embedding_function=embeddings)

    if 'file_change' not in st.session_state:
        st.session_state.file_change = False

    if uploaded_file:
        textInPDF = check_pdfs(uploaded_file)
        if not textInPDF:
            st.warning(f"""No text found in the document. Please reupload a proper file!!""", icon="🚨")
            return

        tokenizer=AutoTokenizer.from_pretrained("Ganidu/Phi-3-Finetuned-LORA", trust_remote_code=True)
        tokens=tokenizer.tokenize(textInPDF)
        token_count=len(tokens)


        if token_count >=10000:
            st.warning(f"""WORD COUNT of this document : {token_count}. Please upload a file with word count less than 10000 !!""", icon="🚨")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=150)
        chunked_text_in_PDF = text_splitter.split_text(textInPDF)

        docs_in_vector_store = []
        if len(docs_in_vector_store) == 0:
            for chunk in chunked_text_in_PDF:
                docs_in_vector_store.extend(vectorDB.similarity_search_with_score(chunk))

        sum_similarity_score = 0.0
        doc_count = 0

        for doc, score in docs_in_vector_store:
            sum_similarity_score += score
            doc_count += 1

        average_similarity_score = sum_similarity_score / doc_count
        

        N = 3
        top_n_docs = []
        for j in docs_in_vector_store:
            top_n_docs.append(j)
        top_n_docs.sort(key=lambda x: x[1])
        top_n_docs = top_n_docs[:N]

        if average_similarity_score >= 14.0:
            st.warning("Please upload content only related to G.C.E. O/L Science Education !!", icon="🚨")
        else:
            st.success('Document Verified!', icon="✅")
            total_possible_mcq_pairs = int(sum(estimate_qa_pairs(doc.page_content) for doc, _ in top_n_docs) / N)
            total_possible_essay_pairs = total_possible_mcq_pairs // 7

            st.sidebar.write(f"Total possible MCQ pairs: {total_possible_mcq_pairs}")
            st.sidebar.write(f"Total possible essay-type pairs: {total_possible_essay_pairs}")

            num_mcq = st.sidebar.number_input("Number of Multiple Choice Questions", min_value=0, max_value=total_possible_mcq_pairs, on_change=lambda: setattr(st.session_state, 'file_change', True))
            num_essay = st.sidebar.number_input("Number of Essay Type Questions", min_value=0, max_value=total_possible_essay_pairs, on_change=lambda: setattr(st.session_state, 'file_change', True))
            generate_button = st.sidebar.button("Create QnA Pairs",on_click=lambda: setattr(st.session_state, 'file_change', True))

            st.sidebar.write(f"Total selected Q&A pairs: {num_mcq + num_essay}")

            if generate_button:
                st.session_state.file_change = False
                if num_mcq > total_possible_mcq_pairs or num_essay > total_possible_essay_pairs:
                    st.sidebar.error("Selected number of Q&A pairs exceeds the total possible Q&A pairs.")
                elif num_mcq == 0 and num_essay == 0:
                    st.sidebar.error("You haven't selected any number of MCQ or Essay type Questions to generate!!")
                else:
                    st.session_state.file_change = True
                    with st.spinner('Generating response...'):
                        print("\n\n\n####################   \n\nexecuting Generating response spinner...\n\n#####################")
                        generated_mcq_pairs = []
                        generated_essay_pairs = []
                        sorted_mcq_objs =[]
                        sorted_essay_objs = []
                        # top_doc_content = []
                        while len(sorted_mcq_objs) <= num_mcq and len(sorted_essay_objs) <= num_essay :
                            if len(sorted_mcq_objs) == num_mcq and len(sorted_essay_objs) == num_essay:
                                break
                            for doc, _ in top_n_docs:
                                # top_doc_content.append(doc.page_content)
                                if num_essay == 0:
                                    print("\n\n\n####################   \n\nexecuting num_essay = 0 ....\n\n#####################")
                                    generated_mcq_pairs.append(generate_mcq_pairs(doc.page_content, num_mcq))
                                elif num_mcq == 0:
                                    print("\n\n\n####################   \n\nexecuting num_mcq = 0 ....\n\n#####################")
                                    generated_essay_pairs.append(generate_essay_pairs(doc.page_content, num_essay))
                                else:
                                    print("\n\n\n####################   \n\n generating both mcq and essay ....\n\n#####################")
                                    generated_mcq_pairs.append(generate_mcq_pairs(doc.page_content, num_mcq))
                                    generated_essay_pairs.append(generate_essay_pairs(doc.page_content, num_essay))

                            print("\n\n\n####################   \n\n string to json -mcq  ....\n\n#####################")
                            mcq_json_objects = string_to_json(generated_mcq_pairs)
                            print("\n\n\n####################   \n\n json sorting -mcq  ....\n\n#####################")
                            sorted_mcq_objs = sorting_mcq_json_objects(mcq_json_objects, num_mcq, vectorDB)

                            print("\n\n\n####################   \n\n string to json - essay  ....\n\n#####################")
                            essay_json_objects = string_to_json(generated_essay_pairs)
                            print("\n\n\n####################   \n\n json sorting - essay  ....\n\n#####################")
                            sorted_essay_objs = sorting_essay_json_objects(essay_json_objects, num_essay)
                            # st.write("generated_mcq_objects")
                            # st.write(generated_mcq_pairs)
                            # st.write("mcq_Json_objects")
                            # st.write(mcq_json_objects)
                            # st.write("sorted_mcq_objects")
                            # st.write(sorted_mcq_objs)

                        if len(sorted_mcq_objs) !=0:
                            mcq_text = "Multiple Choice Questions\n\n\n"
                            mcq_answer_text = "Multiple Choice Questions Answers\n\n\n"
                            for mcq in sorted_mcq_objs:
                                question = mcq["question"]
                                options = ''
                                answer_index = 0
                                for option in mcq["options"]:
                                    options += "0" + str(mcq["options"].index(option) + 1) + "." + option + "\n"
                                    if re.search(re.escape(mcq["answer"]), option, re.IGNORECASE): 
                                        answer_index = mcq["options"].index(option)
                                        mcq_answer_text += "(" + str(sorted_mcq_objs.index(mcq) + 1) + ") " + str(answer_index + 1) + ". " + mcq["answer"] + "\n\n"

                                mcq_text += "(" + str(sorted_mcq_objs.index(mcq) + 1) + ") " + question + "\n" + options + "\n\n"

                            st.session_state.pdf_buffer_mcq = create_pdf(mcq_text, "Multiple Choice Questions")
                            st.session_state.pdf_buffer_mcq_answers = create_pdf(mcq_answer_text, "Multiple Choice Questions Answers")

                        if len(sorted_essay_objs) !=0:
                            essay_text = "Essay Type Questions\n\n\n"
                            essay_answer_text = "Essay Type Questions Answers\n\n\n"
                            for essay in sorted_essay_objs:
                                question = essay["question"]
                                detailed_answer = essay["detailed_answer"]
                                essay_text += "(" + str(sorted_essay_objs.index(essay) + 1) + ") " + question + "\n\n"
                                essay_answer_text += "(" + str(sorted_essay_objs.index(essay) + 1) + ") " + detailed_answer + "\n\n"

                            st.session_state.pdf_buffer_essay = create_pdf(essay_text, "Essay Type Questions")
                            st.session_state.pdf_buffer_essay_answers = create_pdf(essay_answer_text, "Essay Questions Answers")

                st.session_state.file_change = False
            if 'pdf_buffer_mcq' in st.session_state:
                st.write("Here are your generated files. Please click to download.")
                st.download_button(label="Download MCQ PDF", data=st.session_state.pdf_buffer_mcq, file_name="generated_mcq.pdf", mime="application/pdf", disabled=st.session_state.file_change)

            if 'pdf_buffer_mcq_answers' in st.session_state:
                st.download_button(label="Download MCQ - Answers - PDF", data=st.session_state.pdf_buffer_mcq_answers, file_name="generated_mcq_answers.pdf", mime="application/pdf", disabled=st.session_state.file_change)

            if 'pdf_buffer_essay' in st.session_state:
                st.download_button(label="Download Essay Questions PDF", data=st.session_state.pdf_buffer_essay, file_name="generated_eq.pdf", mime="application/pdf", disabled=st.session_state.file_change)

            if 'pdf_buffer_essay_answers' in st.session_state:
                st.download_button(label="Download Essay Questions-Answers - PDF", data=st.session_state.pdf_buffer_essay_answers, file_name="generated_eq_answers.pdf", mime="application/pdf", disabled=st.session_state.file_change)

    else:
        st.write("\n\n")
        st.info("""Instructions:
                \n• Note that you can only upload G.C.E. O/L Science education subject related Notes documents only.
                \n• Only Pdf format files are allowed.
                \n• Maximum word count for a document is 10000.
                \n• Only One Document is allowed to upload at a time. 
                \n• This application can create only maximum 40 MCQs and 5 Essay type questions only.
                \n• For a better experience keep the maximum page count of the document upto 25 pages
                \n• Questions and answers are generated from a Fine-tuned Large Language model therefore sometimes outputs might not be 100% accurate""", icon="ℹ️")

if __name__ == "__main__":
    main()
