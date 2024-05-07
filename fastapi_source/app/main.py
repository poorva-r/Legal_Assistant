from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from fastapi import HTTPException
from dotenv import load_dotenv
import google.generativeai as ggi
from typing import List
from langchain.embeddings import GooglePalmEmbeddings
from PyPDF2 import PdfReader
import docx



app = FastAPI()

# Mount the directory containing static files
app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")

# Load the model and other dependencies
models_folder = "models"
model_path = os.path.join(models_folder, 'model.pkl')
with open(model_path, 'rb') as model_file:
    ipc_model = pickle.load(model_file)

encoder_path = os.path.join(models_folder, 'label_encoder.pkl')
with open(encoder_path, 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

vectorizer_path = os.path.join(models_folder, 'vectorizer.pkl')
with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load environment variables from .env file
load_dotenv(".env")

# Fetch API key from environment variables
os.environ['GOOGLE_API_KEY'] = 'AIzaSyDlq6BXL8gQsFHg8JVHWb38J4lOk_BHfnA'
fetched_api_key = os.getenv("API_KEY")

# Configure the Gemini Pro API with the fetched API key
ggi.configure(api_key=fetched_api_key)

embeddings = GooglePalmEmbeddings()
# Initialize the generative model (Gemini Pro)
model = ggi.GenerativeModel("gemini-pro")

# Start a chat session with the initialized model
chat = model.start_chat()

def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Define landing page
@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("landing_page.html", {"request": request})

# Define classifier page
@app.get("/classifier", response_class=HTMLResponse)
async def classifier_page(request: Request):
    return templates.TemplateResponse("classifier_page.html", {"request": request})

# Define endpoint to handle form submission
@app.post("/classify")
async def classify(request: Request, input_text: str = Form(...)):
    # Use CountVectorizer to convert input text to numerical format
    input_numerical = vectorizer.transform([input_text])

    # Make prediction using the trained model
    prediction = ipc_model.predict(input_numerical.toarray())

    # Get top 3 predicted labels
    top_indices = np.argsort(prediction[0])[::-1][:3]
    top_labels = label_encoder.inverse_transform(top_indices)

    # Prepare the output to display
    output = []
    for label in top_labels:
        output.append(f"{label}\n\n")

    # Render the classifier page with the predictions
    return templates.TemplateResponse("classifier_page.html", {"request": request, "predictions": output})


# Define chatbot page
@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    return templates.TemplateResponse("chatbot_page.html", {"request": request})

# Define endpoint to handle chatbot messages
@app.post("/chatbot-query")
async def chatbot_message(request: Request, message: str = Form(...)):

    try:
        # Get response from the Gemini API for the user's question
        response = chat.send_message(message, stream=True)

        # Extract text from the response object and wrap lines
        response_text = "".join([word.text for word in response])

        # plain_text_response = response_text.replace('**', '').replace('*', '')

        # Return the response
        return {"message": response_text}
    except Exception as e:
        return {"error": str(e)}
    

# Define summarise page
@app.get("/summarizer", response_class=HTMLResponse)
async def summarizer_page(request: Request):
    return templates.TemplateResponse("summarize_page.html", {"request": request})

@app.post("/summarize")
async def summarize(request: Request, file: UploadFile = File(None), user_input: str = Form(None)):
    if file is not None:
        # Read text from uploaded file
        file_extension = file.filename.split(".")[-1]
        if file_extension == "pdf":
            text = extract_text_from_pdf(file.file)
        elif file_extension == "docx":
            text = extract_text_from_docx(file.file)
        else:
            text = file.file.read().decode("utf-8")
    elif user_input:
        text = user_input
    else:
        return {"error": "No input provided"}

    # Send text to model for summarization
    response = chat.send_message(text)
    response_text = response.text
    plain_text_response = response_text.replace('**', '').replace('*', '')
    # return {"summary": response.text}

    return templates.TemplateResponse("summarize_page.html", {"request": request, "summary": plain_text_response})

# Define casestudy page
@app.get("/casestudy", response_class=HTMLResponse)
async def casestudy_page(request: Request):
    return templates.TemplateResponse("casestudy_page.html", {"request": request})

# Define chat with pdf page
@app.get("/chatwithpdf", response_class=HTMLResponse)
async def chatwithpdf_page(request: Request):
    return templates.TemplateResponse("chatwithpdf_page.html", {"request": request})




