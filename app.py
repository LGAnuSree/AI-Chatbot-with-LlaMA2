import os
import faiss
import numpy as np
import streamlit as st
import pdfplumber
import pytesseract
import whisper
import pandas as pd
import torch
import ollama  
import ffmpeg
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from docx import Document
from concurrent.futures import ThreadPoolExecutor

#Ensure efficient GPU usage
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

# oad embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS index
dimension = 384  # MiniLM embedding size
index = faiss.IndexFlatL2(dimension)

# Store document texts
document_texts = []
processed_files = set()

# Load BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Load Whisper model for video transcription
whisper_model = whisper.load_model("base")

# Configure Tesseract path (update as needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def generate_with_llama(prompt):
    try:
        response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        return f"LLaMA 2 Error: {str(e)}"

# Text Extraction Functions
def extract_text_from_pdf(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages).strip()
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

def extract_text_from_docx(docx_file):
    try:
        doc = Document(docx_file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Error extracting DOCX text: {str(e)}"

def extract_text_from_csv(csv_file):
    try:
        df = pd.read_csv(csv_file)
        return df.to_string()
    except Exception as e:
        return f"Error extracting CSV text: {str(e)}"

def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        inputs = processor(image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return f"Extracted Text: {text}\nCaption: {caption}"
    except Exception as e:
        return f"Error extracting image text: {str(e)}"

def extract_text_from_video(video_file):
    try:
        temp_audio = "temp_audio.wav"
        video_path = os.path.join("temp_videos", video_file.name)
        
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        
        ffmpeg.input(video_path).output(temp_audio, format="wav", acodec="pcm_s16le", ac=1, ar=16000).run(overwrite_output=True)
        result = whisper_model.transcribe(temp_audio)
        os.remove(temp_audio)
        os.remove(video_path)
        return result["text"]
    except Exception as e:
        return f"Error extracting video text: {str(e)}"

def summarize_text(text):
    if len(text.split()) < 50:
        return text
    return generate_with_llama(f"Summarize the following text:\n\n{text}")

def process_file(file):
    ext = file.name.split(".")[-1].lower()
    text = ""
    try:
        if ext == "pdf":
            text = extract_text_from_pdf(file)
        elif ext == "docx":
            text = extract_text_from_docx(file)
        elif ext == "csv":
            text = extract_text_from_csv(file)
        elif ext in ["jpg", "jpeg", "png"]:
            text = extract_text_from_image(file)
        elif ext in ["mp4", "avi", "mov"]:
            text = extract_text_from_video(file)
        else:
            st.error(f"Unsupported file format: {ext}")
            return None
        
        if text:
            summarized_text = summarize_text(text)
            document_texts.append(summarized_text)
            embedding = embedding_model.encode([summarized_text], convert_to_tensor=False).astype('float32')
            index.add(embedding)
            return file.name
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return None

def process_uploaded_files(files):
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_file, files)
        processed_files.update(filter(None, results))

def retrieve_and_generate(query):
    if not document_texts:
        return "No documents uploaded yet."
    
    query_embedding = embedding_model.encode([query], convert_to_tensor=False).astype('float32')
    D, I = index.search(query_embedding, k=1)
    
    if I[0][0] == -1:
        return "No relevant information found."
    
    retrieved_text = document_texts[I[0][0]]
    prompt = f"Context: {retrieved_text}\nQuestion: {query}\nAnswer:"
    return generate_with_llama(prompt)

# Streamlit UI
st.title("AI Chatbot (Built with LLaMA 2)")

uploaded_files = st.file_uploader("-->Upload files (PDF, DOCX, CSV, Images, Videos)<--", accept_multiple_files=True)
if uploaded_files:
    process_uploaded_files(uploaded_files)
    st.success("Documents processed successfully!!!")

query = st.text_input("Ask a question:")
if st.button("Submit"):
    response = retrieve_and_generate(query)
    st.write("Response:")
    st.write(response)
