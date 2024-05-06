Steps:

- Create a venv:

python -m venv venv

- Activate venv:

venv\Scripts\activate

- Run the requirements.txt file
  
pip install -r requirements.txt

- Check if all are installed: pip list

- create the .env from template and add api key

- Running the app:

cd fastapi_source

cd app

uvicorn main:app --reload