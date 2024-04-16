from fastapi import FastAPI, Request, Form
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
import markdown


app = FastAPI()


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
fetched_api_key = os.getenv("API_KEY")
print(fetched_api_key)

# Configure the Gemini Pro API with the fetched API key
ggi.configure(api_key=fetched_api_key)

# Initialize the generative model (Gemini Pro)
model = ggi.GenerativeModel("gemini-pro")

# Start a chat session with the initialized model
chat = model.start_chat()

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
        output.append(f"Description of {label}\n{label}")

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
        response_text = "\n".join([word.text for word in response])

        # Return the response
        return {"message": response_text}
    except Exception as e:
        return {"error": str(e)}

