from dotenv import load_dotenv
import streamlit as st
import os
import pickle
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

# Function to get a response from the language model
def LLM_Response(question):
    # Send the user's question to the chat session and get the response
    response = chat.send_message(question, stream=True)
    return response

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

    # If authentication is successful, proceed to display the application options
    if authentication_status:
        authenticator.logout("Logout", "sidebar")
        st.sidebar.title(f"Legal Assistant | Welcome {name}")
        selected_option = st.sidebar.radio("Select Option", ["Chatbot", "Feature 2", "Feature 3"])

        if selected_option == "Chatbot":
            chatbot_feature()

        elif selected_option == "Feature 2":
            feature_2()

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
