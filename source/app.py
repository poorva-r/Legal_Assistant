from dotenv import load_dotenv
import streamlit as st
import os
import pickle
import numpy as np
import google.generativeai as ggi
import streamlit_authenticator as stauth

# Set up authentication
names = ["Poorva", "Vishant", "Vidhi"]
usernames = ["poorva-r", "vishant-m", "vidhi-b"]
file_path = "hashed_pw.pkl"  # Update this with the path to your hashed password file
with open(file_path, "rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "Legal Assistant", "abcdf", cookie_expiry_days=0)

# Load environment variables from .env file
load_dotenv(".env")

# Fetch API key from environment variables
fetched_api_key = os.getenv("API_KEY")

# Configure the Gemini Pro API with the fetched API key
ggi.configure(api_key=fetched_api_key)

# Initialize the generative model (Gemini Pro)
model = ggi.GenerativeModel("gemini-pro")

# Start a chat session with the initialized model
chat = model.start_chat()

# Define the path to the folder containing the models
models_folder = "models"

# Load the model
model_path = os.path.join(models_folder, 'model.pkl')
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the label encoder
encoder_path = os.path.join(models_folder, 'label_encoder.pkl')
with open(encoder_path, 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Load the CountVectorizer
vectorizer_path = os.path.join(models_folder, 'vectorizer.pkl')
with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

#Chatbot feature
def chatbot_feature():
    st.title("Legal Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Hi there! Ask me a legal question:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from the Gemini API for the user's question
        response = chat.send_message(prompt, stream=True)
        
        # Extract text from the response object and wrap lines
        response_text = "\n".join([word.text for word in response])

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response_text)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})


# Function for feature 2
def Classify():
    st.title('IPC Sections Classifier')

    input_text = st.text_input('Enter the situation:', '')

    if st.button('Classify'):
        # Use CountVectorizer to convert the input text to numerical format
        input_numerical = vectorizer.transform([input_text])

        # Make a prediction using the trained model
        prediction = model.predict(input_numerical.toarray())

        # Get the indices of the top 3 predictions in descending order of probability
        top_indices = np.argsort(prediction[0])[::-1][:3]

        # Convert the top indices to class labels
        top_labels = label_encoder.inverse_transform(top_indices)

        st.write("Most probable punishment with IPC sections:")
        for label in top_labels:
            st.write(label)

# Function for feature 3
def feature_3():
    st.title("Feature 3")
    # Add a button to print something
    if st.button("Print"):
        st.write("Printing something...")

# Main function
def main():
    name, authentication_status, username = authenticator.login("Login", "main")

    # If authentication is successful, proceed to display the application options
    if authentication_status:
        authenticator.logout("Logout", "sidebar")
        st.sidebar.title(f"Legal Assistant | Welcome {name}")
        selected_option = st.sidebar.radio("Select Option", ["Chatbot", "IPC Section Classifier", "Feature 3"])

        if selected_option == "Chatbot":
            chatbot_feature()

        elif selected_option == "IPC Section Classifier":
            Classify()

        elif selected_option == "Feature 3":
            feature_3()
    else:
        # If authentication fails or user has not logged in yet, display appropriate messages
        if authentication_status == False:
            st.error("Incorrect Username or Password")

        if authentication_status == None:
            st.warning("Please enter Username and Password")


if __name__ == "__main__":
    main()
