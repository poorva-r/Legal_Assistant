from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as ggi

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

# Set up the Streamlit app interface
st.title("Legal Assistant")

# Add a sidebar for options
option = st.sidebar.radio("Select Option", ["Chatbot", "Feature 2", "Feature 3"])

# If Chatbot option is selected
if option == "Chatbot":

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

#If option Feaure 2 is selected
elif option == "Feature 2":
    
    st.title("Feature 2")
    # Add a button to print something
    if st.button("Print"):
        st.write("Printing something...")


#If option Feaure 3 is selected
elif option == "Feature 3":
    st.title("Feature 2")
    # Add a button to print something
    if st.button("Print"):
        st.write("Printing something...")

