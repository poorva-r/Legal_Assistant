from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from fastapi import HTTPException


app = FastAPI()


templates = Jinja2Templates(directory="templates")

# Load the model and other dependencies
models_folder = "models"
model_path = os.path.join(models_folder, 'model.pkl')
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

encoder_path = os.path.join(models_folder, 'label_encoder.pkl')
with open(encoder_path, 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

vectorizer_path = os.path.join(models_folder, 'vectorizer.pkl')
with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

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
    prediction = model.predict(input_numerical.toarray())

    # Get top 3 predicted labels
    top_indices = np.argsort(prediction[0])[::-1][:3]
    top_labels = label_encoder.inverse_transform(top_indices)

    # Prepare the output to display
    output = []
    for label in top_labels:
        output.append(f"Description of {label}\n{label}")

    # Render the classifier page with the predictions
    return templates.TemplateResponse("classifier_page.html", {"request": request, "predictions": output})
