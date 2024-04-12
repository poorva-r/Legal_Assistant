from dotenv import load_dotenv
import streamlit as st
import os
import pickle
import google.generativeai as ggi
import streamlit_authenticator as stauth

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

# Function to get a response from the language model
def LLM_Response(question):
    # Send the user's question to the chat session and get the response
    response = chat.send_message(question, stream=True)
    return response

# Set up authentication
names = ["Poorva", "Vishant", "Vidhi"]
usernames = ["poorva-r", "vishant-m", "vidhi-b"]
file_path = "hashed_pw.pkl"  # Update this with the path to your hashed password file
with open(file_path, "rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "Legal Assistant", "abcdf", cookie_expiry_days=2)

# Function for the chatbot feature
def chatbot_feature():
    st.title("Legal Chatbot")
    # Add a text input box for the user to ask a question
    user_quest = st.text_input("Ask a legal query:")
    
    # Add a button to submit the question
    btn = st.button("Ask")
    
    # Handle button click event and user input
    if btn and user_quest:
        # Get response from the language model for the user's question
        result = LLM_Response(user_quest)
        
        # Display the response
        st.subheader("Response : ")
        for word in result:
            st.text(word.text)

# Function for feature 2
def feature_2():
    st.title("Feature 2")
    # Add a button to print something
    if st.button("Print"):
        st.write("Printing something...")

# Function for feature 3
def feature_3():
    st.title("Feature 3")
    # Add a button to print something
    if st.button("Print"):
        st.write("Printing something...")

# Main function
def main():
    name, authentication_status, username = authenticator.login("Login", "main")

    if authentication_status == False:
        st.error("Incorrect Username or Password")

    if authentication_status == None:
        st.warning("Please enter Username and Password")

    if authentication_status:
        st.sidebar.title(f"Legal Assistant | Welcome {name}")
        selected_option = st.sidebar.radio("Select Option", ["Chatbot", "Feature 2", "Feature 3"])

        if selected_option == "Chatbot":
            chatbot_feature()

        elif selected_option == "Feature 2":
            feature_2()

        elif selected_option == "Feature 3":
            feature_3()

if __name__ == "__main__":
    main()
