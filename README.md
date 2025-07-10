# FYP_Scio_Learn
Created by Ganidu Bandara
for the final year project of Bachelors of Science in computer science (university of Westminster)
A questionaire generator for the local G.C.E. Ordinary Level science education subject examination.

I developed an AI-powered question paper generation system tailored for Sri Lankan G.C.E. Ordinary Level Science education. The system leverages cutting-edge LLMs (Large Language Models) and embedding models to generate exam-ready multiple-choice and essay-type questions from syllabus-based study material.

Key Highlights:

✅ Fine-tuned LLM: GaniduA/phi3-finetuned-olscience-merged

✅ Fine-tuned Embedding Model: GaniduA/bge-finetuned-olscience

✅ Manually built a custom dataset with 12,000+ context-question-answer pairs specifically designed for O/L Science, as no open-source dataset existed for this domain.

✅ Used a Retrieval-Augmented Generation (RAG) pipeline with ChromaDB for similarity search and document retrieval.

✅ Developed a full-stack system with a Streamlit-powered user interface, allowing PDF syllabus uploads, customizable question counts, and automatic PDF export of generated exam papers.

Throughout the project, I explored advanced fine-tuning, model evaluation with academic benchmarks (ARC, MMLU, SciQ, OpenBookQA), and deployed everything on Hugging Face Inference Endpoints.

🧠 This project demonstrates how domain-specific AI systems can be built from scratch, even without readily available datasets.

Check it out on Hugging Face!
LLM: https://huggingface.co/GaniduA/phi3-finetuned-olscience-merged
Embedding Model: https://huggingface.co/GaniduA/bge-finetuned-olscience
