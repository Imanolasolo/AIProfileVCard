import os
# Allow duplicate OpenMP runtimes to load (workaround for Windows libomp/libiomp conflicts).
# This is an unsafe workaround; prefer installing compatible packages or ensuring a single OpenMP runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import os
import base64

# Set your name for the AIProfileVCard
name = 'Imanol Asolo'

# Language dictionaries
TRANSLATIONS = {
    'en': {
        'title': 'AIProfileVCard',
        'ceo_title': 'CEO of CodeCodix',
        'about_me_title': '### About Me',
        'about_me_text': '''I am the CEO of CodeCodix, a company dedicated to developing artificial intelligence tools. Besides leading the company, I am a proud father who loves the sea. At CodeCodix, I work as a Full Stack Developer and Scrum Master, playing multiple roles to ensure the success of our projects. My passion for technology and versatility in my skills allows me to contribute significantly to various aspects of development and team management.''',
        'services_title': '### Services Offered',
        'services': ['AI Tools Development', 'Technology Consulting', 'Project Management'],
        'projects_title': '### Developed Projects',
        'projects': ['AI_Medicare', 'Raptor_eye', 'Botarmy_Hub'],
        'chat_title': '### Chat with Me, know me and let\'s contact!',
        'chat_info': 'No matter the language, ask anything you need!',
        'ask_placeholder': 'Ask me anything:',
        'picture_caption': name
    },
    'es': {
        'title': 'AIProfileVCard',
        'ceo_title': 'CEO de CodeCodix',
        'about_me_title': '### Sobre Mí',
        'about_me_text': '''Soy el CEO de CodeCodix, una empresa dedicada al desarrollo de herramientas de inteligencia artificial. Además de liderar la empresa, soy un padre orgulloso que ama el mar. En CodeCodix, trabajo como Full Stack Developer y Scrum Master, desempeñando múltiples roles para asegurar el éxito de nuestros proyectos. Mi pasión por la tecnología y la versatilidad en mis habilidades me permite contribuir significativamente a varios aspectos del desarrollo y la gestión de equipos.''',
        'services_title': '### Servicios Ofrecidos',
        'services': ['Desarrollo de Herramientas IA', 'Consultoría Tecnológica', 'Gestión de Proyectos'],
        'projects_title': '### Proyectos Desarrollados',
        'projects': ['AI_Medicare', 'Raptor_eye', 'Botarmy_Hub'],
        'chat_title': '### ¡Chatea conmigo, conóceme y contactemos!',
        'chat_info': '¡No importa el idioma, pregunta lo que necesites!',
        'ask_placeholder': 'Pregúntame lo que quieras:',
        'picture_caption': name
    }
}

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
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
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

    # Initialize language in session state
    if 'language' not in st.session_state:
        st.session_state.language = 'en'

    # Get current language translations
    lang = TRANSLATIONS[st.session_state.language]

    # Function to encode image as base64 to set as background
    # def get_base64_of_bin_file(bin_file):
        #with open(bin_file, 'rb') as f:
            #data = f.read()
        #return base64.b64encode(data).decode()

    ## Encode the background image
    #img_base64 = get_base64_of_bin_file('logo_portfolio.jpg')

    ## Set the background image using the encoded base64 string
    #st.markdown(
    #f"""
    #<style>
    #.stApp {{
        #background: url('data:image/jpeg;base64,{img_base64}') no-repeat center center fixed;
        #background-size: cover;
    #}}
    #</style>
    #""",
    #unsafe_allow_html=True
#)

    # Apply custom CSS styles
    st.write(css, unsafe_allow_html=True)

    # Language selector buttons next to title
    col_title, col_lang = st.columns([3, 1])
    with col_title:
        st.title(lang['title'])
    with col_lang:
        col_en, col_es = st.columns(2)
        with col_en:
            if st.button("🇬🇧 EN"):
                st.session_state.language = 'en'
                st.rerun()
        with col_es:
            if st.button("🇪🇸 ES"):
                st.session_state.language = 'es'
                st.rerun()

    st.header(name)
    st.subheader(lang['ceo_title'])

    # Display the profile picture and description in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.image('picture_imanol.png', caption=lang['picture_caption'], width=200)
    with col2:
        st.markdown(lang['about_me_title'])
        st.markdown(lang['about_me_text'])

    # Display services offered and projects developed in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.write(lang['services_title'])
        for service in lang['services']:
            st.write(f"- {service}")

    with col2:
        st.write(lang['projects_title'])
        for project in lang['projects']:
            st.write(f"- {project}")

    # Section for interacting with the AI chatbot
    st.write(lang['chat_title'])
    st.info(lang['chat_info'])

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
    user_question = st.text_input(lang['ask_placeholder'])
    if user_question:
        handle_user_input(user_question)

# Run the main function if the script is executed
if __name__ == "__main__":
    main()