import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI 
from htmlTemplates import css, bot_template, user_template
import os
import base64

# Set your name for the AIProfileVCard
name = 'Imanol Asolo'

# Function to extract text from a PDF file
def get_pdf_text(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    # Iterate through each page and extract text
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split the extracted text into manageable chunks for processing
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate a vector store using the text chunks
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPEN_AI_APIKEY"])
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversational chain using the vector store
def get_conversation_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=st.secrets["OPEN_AI_APIKEY"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input and generate responses
def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display the conversation history
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title= name, page_icon=":wave:", layout="centered")

    # Function to encode image as base64 to set as background
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    # Encode the background image
    img_base64 = get_base64_of_bin_file('logo_portfolio.jpg')

    # Set the background image using the encoded base64 string
    st.markdown(
    f"""
    <style>
    .stApp {{
        background: url('data:image/jpeg;base64,{img_base64}') no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

    # Apply custom CSS styles
    st.write(css, unsafe_allow_html=True)

    # Set the title and header of the app
    st.title("AIProfileVCard")
    st.header(name)
    st.subheader("CEO of CodeCodix")

    # Display the profile picture and description in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.image('picture_imanol.png', caption=name, width=200)
    with col2:
        description = """
        ### About Me
        I am the CEO of CodeCodix, a company dedicated to developing artificial intelligence tools. Besides leading the company, I am a proud father who loves the sea. At CodeCodix, I work as a Full Stack Developer and Scrum Master, playing multiple roles to ensure the success of our projects. My passion for technology and versatility in my skills allow me to contribute significantly to various aspects of development and team management.
        """
        st.markdown(description)

    # Display services offered and projects developed in two columns
    col1, col2 = st.columns(2)
    with col1:
        services = ["AI Tools Development", "Technology Consulting", "Project Management"]
        st.write("### Services Offered")
        for service in services:
            st.write(f"- {service}")

    with col2:
        projects = ["AI_Medicare", "Raptor_eye", "Botarmy_Hub"]
        st.write ('### Developed projects')
        for project in projects:
            st.write(f"- {project}")

    # Section for interacting with the AI chatbot
    st.write("### Chat with Me, know me and let´s contact!")
    st.info("Doesn´t matter the language, ask anything you need!")

    # Process the PDF file to be used as context for the chatbot
    pdf_path = os.path.join(os.getcwd(), "imanolpdf1.pdf")
    pdf_text = get_pdf_text(pdf_path)
    text_chunks = get_text_chunks(pdf_text)
    vector_store = get_vector_store(text_chunks)
    conversation_chain = get_conversation_chain(vector_store)

    # Store the conversation chain and history in session state
    st.session_state.conversation = conversation_chain
    st.session_state.chat_history = []

    # Input box for user questions
    user_question = st.text_input("Ask me anything:")
    if user_question:
        handle_user_input(user_question)

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
