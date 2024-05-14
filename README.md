# Legal Assistant

## Project Description

This repository contains the code for a Legal Assistant software application designed to aid both legal and non-legal professionals. The Legal Assistant provides several functionalities, including a general chatbot, a summarizer, a relevant case study scraper, and an IPC section classifier. These features aim to streamline various tasks within the legal domain, enhancing efficiency and accessibility.

## Technologies Used

### Languages
- Python
- HTML
- CSS
- JavaScript

### Tools
- Google Colab
- VSCode
- Jupyter Notebook

### Libraries
- FastAPI: A web framework for building APIs with Python.
- Jinja Templates: A templating engine for Python, used to generate HTML or other markup formats.
- Keras: Deep learning framework for building and training neural networks.
- TensorFlow: Open-source machine learning library.
- Streamlit: For building interactive and customizable web applications.
- Streamlit-Authenticator: Enables the creation of a login interface for Streamlit applications.
- NLTK (Natural Language Toolkit): Suite of libraries and programs for natural language processing tasks.
- Beautiful Soup: Python library for pulling data out of HTML and XML files.
- Requests: HTTP library for making requests and working with APIs.
- PyPDF2: Pure-Python PDF toolkit.
- Docx: Python library for creating and updating Microsoft Word (.docx) files.
- Scikit-learn: Machine learning tools and algorithms.
- NumPy: Library for numerical computing in Python.

### External APIs
- Google Gemini Pro Model: Utilized with its API key for the chatbot and summarization feature.

## Project Screenshots
![Screenshot 2024-05-07 160130](https://github.com/poorva-r/Legal_Assistant/assets/85826531/430bfef7-ad85-4435-ab81-ec776d45fa91)

![Screenshot 2024-05-07 215002](https://github.com/poorva-r/Legal_Assistant/assets/85826531/bd1fae28-1873-4899-80fd-dd2b121bfa51)

![Screenshot 2024-05-07 160207](https://github.com/poorva-r/Legal_Assistant/assets/85826531/265a09f9-b3c4-4c25-9875-dbb6fbf88e27)

![Screenshot 2024-05-07 160244](https://github.com/poorva-r/Legal_Assistant/assets/85826531/619dc9e1-555c-4574-8e31-c5edd6bd7071)

![Screenshot 2024-05-07 171047](https://github.com/poorva-r/Legal_Assistant/assets/85826531/4767b349-f097-4b81-b73c-4f3ba3bbbb21)

![Screenshot 2024-05-07 220950](https://github.com/poorva-r/Legal_Assistant/assets/85826531/1c08f860-df87-4078-a88e-03f2880b3f50)

![Screenshot 2024-05-07 160032](https://github.com/poorva-r/Legal_Assistant/assets/85826531/e2cf4ea7-bb09-413d-82cb-f500ba022c17)


### Installation:

1. Create a virtual environment:
   
python -m venv venv

2. Activate the virtual environment:
   
venv\Scripts\activate

3. Install the required packages from the requirements.txt file:
   
pip install -r requirements.txt

4. Verify if all packages are installed:

pip list

5. Create the .env file from the template provided and add the API key.

### Running the app:

1. Navigate to the source directory:
   
cd fastapi_source

2. Navigate to the app directory:

cd app

3. Start the FastAPI server using uvicorn:

uvicorn main:app --reload

